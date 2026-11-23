"""Measure time to a predeclared multichain precision criterion on CPU."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

from poremc.cli import construct_case
from poremc.statistics import chain_mean_interval


SEEDS = (440713, 440731, 440747, 440753)
OBSERVE_EVERY = 200


def write_json(path, payload):
    def builtin(obj):
        if isinstance(obj,np.ndarray):
            return obj.tolist()
        if isinstance(obj,np.generic):
            return obj.item()
        raise TypeError(type(obj).__name__)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload,default=builtin,indent=2)+'\n')
    tmp.replace(path)


def read_model(path):
    return json.loads(path.read_text())


def setup(model,path,seed,mode,cache,mesh,guest_update):
    case=copy.deepcopy(model)
    sample=case['sampling']
    sample.update(driver='cpp',seed=seed,sampling_mode=mode,guest_update=guest_update,
                  reciprocal_backend='cpu')
    if mode=='exact':
        sample.pop('grid_root_bias',None)
    else:
        sample.update(host_field_cache=str(cache.resolve()),host_grid_mesh=mesh)
        sample.setdefault('grid_root_bias',.60)
    return construct_case(case,path.parent)[0]


def preload(path, cache, mesh):
    """Prepare two independent fixed-count states shared by the engine comparisons."""
    model=read_model(path)
    expected=np.asarray(model['metadata']['preload_counts'],dtype=int)
    starts=[]
    for i,seed in enumerate(SEEDS[2:]):
        with setup(model,path,seed+91001,'delayed',cache,mesh,'incremental') as sim:
            sim.set_activities(np.maximum(sim.activities*350.,1e-7))
            for attempt in range(45):
                sim.run(2000,sample_every=2000,record_energy=False)
                if np.all(sim.counts>=expected):
                    break
            if not np.all(sim.counts>=expected):
                raise RuntimeError(f'preload failed in {path.stem}: {sim.counts} vs {expected}')
            xyz=sim.entries()
            parts=[]
            for component,n in enumerate(expected):
                subset=[a for s,a in xyz if s==component]
                indices=np.random.default_rng(seed+1239).choice(len(subset),size=n,replace=False)
                parts.extend([component,subset[j].tolist()] for j in indices)
            starts.append(parts)
        print(path.stem,'preloaded seed',i+1,expected.tolist(),flush=True)
    write_json(path.with_name(path.stem+'_starts.json'),[[],[],*starts])


def trans_count(coordinates):
    x=np.asarray(coordinates)
    if len(x)==0:
        return 0
    b=np.diff(x,axis=1)
    normal1=np.cross(b[:,0,:],b[:,1,:])
    normal2=np.cross(b[:,1,:],b[:,2,:])
    cosines=np.sum(normal1*normal2,axis=1)/(np.linalg.norm(normal1,axis=1)*np.linalg.norm(normal2,axis=1))
    return int(np.sum(cosines<=-.5))


def conduct_chain(job):
    path, start, seed, mode, budgets, warmup, cache, mesh, guest_update, origin, out=job
    t0=time.perf_counter()
    model=read_model(path)
    settings=model['metadata']
    flex=settings.get('observe_trans',False)
    with setup(model,path,seed,mode,cache,mesh,guest_update) as sim:
        initial_counts=np.bincount([entry[0] for entry in start],minlength=len(sim.species)).tolist()
        for species,coords in start:
            sim.add_particle(species,coords)
        preparation=time.perf_counter()-t0
        sim.run(0,burnin=warmup,record_energy=False)
        post_burnin=time.perf_counter()
        acc_before=sim.acceleration_profile
        move_before=sim.counters
        counts=[]
        trans=[]
        stamps=[]
        profiles=[]
        past=0
        for steps in budgets:
            if flex:
                for _ in range((steps-past)//OBSERVE_EVERY):
                    r=sim.run(OBSERVE_EVERY,sample_every=OBSERVE_EVERY,record_energy=False)
                    counts.extend(r['counts'])
                    trans.append(trans_count([xyz for sid,xyz in sim.entries()]))
            else:
                r=sim.run(steps-past,sample_every=OBSERVE_EVERY,record_energy=False)
                counts.extend(r['counts'])
            stamps.append(time.perf_counter()-origin)
            profiles.append({'production_wall_s':time.perf_counter()-post_burnin,'steps':steps,
                             'acceleration':sim.acceleration_profile,'move_counters':sim.counters,
                             'long_range':sim.long_range_profile})
            past=steps
        energy=sim.total_energy()
        entries=sim.entries()
        # The whole-box Python oracle rebuilds a reciprocal environment for
        # every molecule. Compare the large-box Hamiltonian against independent
        # RASPA snapshots instead of repeating that reconstruction per chain.
        large_ewald=sim.potential.ewald is not None and len(sim.potential.host_positions)>4000 and len(entries)>32
        closure=None if large_ewald or model['sampling'].get('ewald_k_cutoff') else abs(
            energy-sim.potential.total_energy(entries))
        setup_grid=sim.acceleration_profile.get('host_grid')
    filename=f'{out.stem}_{seed}.npz'
    np.savez_compressed(out.parent/filename,counts=np.asarray(counts),trans_count=np.asarray(trans))
    return {'seed':seed,'start':'preloaded' if start else 'empty','initial_counts':initial_counts,
            'preparation_s':preparation,'elapsed_at_prefix_s':stamps,'last_energy_K':energy,
            'energy_closure_K':closure,'moves_after_warmup':move_before,'acceleration_after_warmup':acc_before,
            'profiles':profiles,'cache':setup_grid,'data':filename}


def finite(x):
    return x if math.isfinite(x) else None


def estimate(x):
    metrics=chain_mean_interval(x)
    return {key:finite(value) if isinstance(value,(float,np.floating)) else value
            for key,value in metrics.items()}


def diagnostics(counts, trans, metadata):
    out={f'count_{i}':estimate(counts[:,:,i]) for i in range(counts.shape[2])}
    if metadata.get('joint_threshold'):
        joint=np.all(counts>=np.asarray(metadata['joint_threshold']),axis=2)
        out['joint_probability']=estimate(joint)
    if metadata.get('observe_trans'):
        n=float(counts[:,:,0].mean())
        if n>0:
            p=float(trans.mean()/n)
            out['trans_fraction']=estimate(p+(trans-p*counts[:,:,0])/n)
        else:
            out['trans_fraction']={'mean':None,'mcse':None,'rank_split_rhat':None}
    return out


def summarize(chains, metadata, prefix_lengths, nominal, output, ref):
    count_arrays=[]
    trans_arrays=[]
    for chain in chains:
        with np.load(output.parent/chain['data'],allow_pickle=False) as data:
            count_arrays.append(data['counts'])
            trans_arrays.append(data['trans_count'])
    counts=np.asarray(count_arrays)
    trans=np.asarray(trans_arrays)
    checkpoints=[]
    for i,k in enumerate(prefix_lengths):
        obs=diagnostics(counts[:,:k],trans[:,:k],metadata)
        goals=metadata['precision_goals']
        passing=all(v['rank_split_rhat'] is not None and v['mcse'] is not None and
                    v['rank_split_rhat']<1.01 and v['mcse']<=goals[key]
                    for key,v in obs.items())
        checkpoints.append({'budget':nominal[i],'draws_per_chain':k,
                            'batch_wall_s':max(c['elapsed_at_prefix_s'][i] for c in chains),
                            'observables':obs,'met':passing})
    first=next((row for row in checkpoints if row['met']),None)
    empirical=Counter(map(tuple,counts.reshape(-1,counts.shape[2])))
    empirical={k:v/(counts.shape[0]*counts.shape[1]) for k,v in empirical.items()}
    result={'checkpoints':checkpoints,'qualified_wall_s':None if first is None else first['batch_wall_s'],
            'start_group_final_means':{'empty':counts[:2].mean(axis=(0,1)),
                                      'preload':counts[2:].mean(axis=(0,1))},
            'occupancy_distribution':[[*k,v] for k,v in sorted(empirical.items())],
            'precision_goals':metadata['precision_goals']}
    if ref.exists():
        with np.load(ref,allow_pickle=False) as reference:
            support=reference['counts']
            p=reference['probabilities']
        p_mean=p.mean(axis=0)
        q=np.asarray([empirical.get(tuple(k),0.) for k in support])
        result['NIST']={'reference_replica_means':p@support,
                        'empirical_mean':counts.mean(axis=(0,1)),
                        'total_variation':.5*(np.abs(p_mean-q).sum()+max(0.,1.-q.sum())),
                        'reference_replica_TV':[.5*float(np.abs(row-p_mean).sum()) for row in p]}
        if metadata.get('joint_threshold'):
            mask=np.all(support>=np.asarray(metadata['joint_threshold']),axis=1)
            result['NIST']['joint_probability_by_reference_replica']=p[:,mask].sum(axis=1)
    return result


def run_native(path, output, mode, budgets, warmup, cache, mesh, guest_update, workers, force_cold, seed_offset=0):
    model=read_model(path)
    starts=read_model(path.with_name(path.stem+'_starts.json'))
    origin=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        chains=list(pool.map(conduct_chain,[
            (path,start,seed,mode,budgets,warmup,cache/str(seed) if force_cold else cache,
             mesh,guest_update,origin,output) for start,seed in zip(starts,(s+seed_offset for s in SEEDS))]))
    ref=path.with_name(path.stem+'_nist.npz')
    report=summarize(chains,model['metadata'],[b//OBSERVE_EVERY for b in budgets],budgets,output,ref)
    report.update(case=path.stem,platform=platform.platform(),mode=mode,mesh=mesh,guest_update=guest_update,
                  force_cold=force_cold,warmup=warmup,workers=workers,chains=chains,sample_every=OBSERVE_EVERY)
    write_json(output,report)
    print(path.stem,mode,'batch_wall_s',round(report['checkpoints'][-1]['batch_wall_s'],3),
          'qualified_wall_s',report['qualified_wall_s'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--prepare-starts',action='store_true')
    p.add_argument('--mode',choices=('exact','delayed','reference-mh'),default='delayed')
    p.add_argument('--budgets',type=int,nargs='+',default=[20000,50000,120000,250000])
    p.add_argument('--warmup',type=int,default=20000)
    p.add_argument('--mesh',type=int,default=32)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--guest-update',choices=('incremental','rebuild'),default='incremental')
    p.add_argument('--force-cold',action='store_true')
    p.add_argument('--seed-offset',type=int,default=0)
    a=p.parse_args()
    if a.prepare_starts:
        preload(a.model,a.cache,a.mesh)
        return
    if not a.output:
        p.error('set --output or --prepare-starts')
    if any(s < 4*OBSERVE_EVERY or s%OBSERVE_EVERY for s in a.budgets) or a.budgets!=sorted(set(a.budgets)):
        p.error(f'budgets must be increasing multiples of {OBSERVE_EVERY}, at least {4*OBSERVE_EVERY}')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    run_native(a.model,a.output,a.mode,a.budgets,a.warmup,a.cache,a.mesh,a.guest_update,a.workers,a.force_cold,a.seed_offset)


if __name__=='__main__':
    main()
