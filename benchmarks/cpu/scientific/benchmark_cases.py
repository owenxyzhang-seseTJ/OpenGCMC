"""Reproduce the non-MFI scientific calculations with an installed OpenGCMC."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
from contextlib import ExitStack
import json
import math
from pathlib import Path
import platform
import statistics
import time

import numpy as np
from scipy.stats import t

from poremc.calibration import sample_reaction_windows
from poremc.cli import construct_case
from poremc.electrode import VOLT_TO_K, solve_projected_charge
from poremc.ensembles import GrandCanonicalReplicaExchange
from poremc.fast import FastGCMC
from poremc.free_energy import MBAR
from poremc.reaction_cache import state_energy_differences
from poremc.reactive_cli import build
from poremc.response import electrochemical_reweight
from poremc.scheduling import _publish
from poremc.statistics import chain_mean_interval, rank_split_rhat
from poremc.transport import DiffusionRedox, FARADAY
from poremc.units import KB, NA


HERE = Path(__file__).resolve().parent
SEEDS = (231631, 231653, 231683, 231701)
CONFIRMATION_SEEDS = (724241, 724277, 724309, 724333)


def model(name):
    return json.loads((HERE / 'inputs' / (name + '.json')).read_text())


def independent_interval(values):
    x = np.asarray(values, float)
    mean = float(x.mean())
    sem = float(x.std(ddof=1)/math.sqrt(x.size))
    delta = float(t.ppf(.975,x.size-1)*sem)
    return {'mean':mean,'sem':sem,'ci95':[mean-delta,mean+delta]}


def _shell_counts(frames, cell):
    all_counts = []
    for frame in frames:
        oxygens = np.array([m.positions[0] for m in frame if m.template==0])
        ions = np.array([next(m.positions[0] for m in frame if m.template==s) for s in (1,2)])
        dr = oxygens[None,:,:]-ions[:,None,:]
        dr -= cell*np.rint(dr/cell)
        all_counts.append(np.sum(np.linalg.norm(dr,axis=-1)<np.array([3.2,3.8])[:,None],axis=1))
    return np.asarray(all_counts)


def _electrode_pairs(sim, frames, charges):
    e = sim.h.atomic.electrodes
    a = e.response.A
    n = a.shape[0]
    one = np.ones((n,1))
    kkt = np.block([[a, one],[one.T,np.zeros((1,1))]])
    lower = float(.95*np.linalg.eigvalsh(a)[0])
    dense_ts, projected_ts, errors, bounds = [], [], [], []
    for index in np.linspace(0,len(frames)-1,32,dtype=int):
        frame = frames[index]
        xyz = np.concatenate([m.positions for m in frame])
        qf = np.concatenate([sim.h.templates[m.template].states[m.state].charges for m in frame])
        field = e.fields(xyz,qf).sum(axis=0)
        total = -float(qf.sum())
        ref = np.linalg.solve(kkt,np.r_[VOLT_TO_K*e.potentials-field,total])[:-1]

        def dense():
            start = time.thread_time_ns()
            x = e.response.solve(field,e.potentials,total)
            dense_ts.append(time.thread_time_ns()-start)
            return x

        def projected():
            start = time.thread_time_ns()
            x = solve_projected_charge(a.__matmul__,field,e.potentials,total,
                                      eigenvalue_lower_bound=lower,tolerance=1e-11)
            projected_ts.append(time.thread_time_ns()-start)
            return x

        if len(errors)%2:
            p = projected()
            d = dense()
        else:
            d = dense()
            p = projected()
        if not p['converged']:
            raise RuntimeError('projected solve did not converge')
        errors.append([np.max(np.abs(charges[index]-ref)),np.max(np.abs(p['q']-ref)),
                       abs(p['free_energy']-d['free_energy']),p['potential_residual_V']])
        bounds.append(p['energy_error_bound'])
    return {'snapshots':len(errors),'cached_charge_max_error_e':float(np.max(errors,axis=0)[0]),
            'projected_charge_max_error_e':float(np.max(errors,axis=0)[1]),
            'projected_energy_max_error_K':float(np.max(errors,axis=0)[2]),
            'projected_residual_max_V':float(np.max(errors,axis=0)[3]),
            'projected_variational_bound_max_K':max(bounds),
            'dense_median_us':statistics.median(dense_ts)/1000,
            'projected_median_us':statistics.median(projected_ts)/1000}


def interface_chain(job):
    seed, extended = job
    c = model('nacl_spce_constant_potential')
    c['seed'] = seed
    if extended:
        c['run'].update(steps=160000,burnin=40000,sample_every=80)
    start = time.perf_counter()
    sim = build(c)
    setup = time.perf_counter()-start
    try:
        run = sim.run(**c['run'])
        if not np.all(run['counts']==[62,1,1]):
            raise RuntimeError('unexpected interfacial composition')
        q = run['electrode_charges']
        e = sim.h.atomic.electrodes
        pattern = np.repeat([-.5,.5],len(e.positions)//2)
        area = np.prod(c['cell'][:2])
        cap = e.response.capacitance(q,pattern,sim.temperature,area)
        dv = .0001
        lo,hi = [electrochemical_reweight(run['counts'],q,np.zeros(3),shift,pattern,
                                        e.response,sim.temperature) for shift in (-dv,dv)]
        slope = ((hi['mean_electrode_charges']-lo['mean_electrode_charges'])@pattern)/(2*dv)
        short = _shell_counts(run['frames'],np.asarray(c['cell']))
        pair = _electrode_pairs(sim,run['frames'],q)
        if pair['cached_charge_max_error_e']>1e-8 or pair['projected_charge_max_error_e']>1e-8:
            raise RuntimeError('electrode solver disagrees with direct KKT')
        np.savez_compressed(HERE/f'nacl_{seed}.npz',coordination=short,charges_e=q,
                            potential_conjugate_charge_e=q@pattern,energy_K=run['energy_kelvin'])
        return {'seed':seed,'run':c['run'],'setup_seconds':setup,'production_seconds':run['wall_seconds'],
                'moves_per_second':run['moves_per_second'],'energy_recompute_error_K':run['energy_recompute_error_K'],
                'max_neutrality_error_e':float(np.max(np.abs(q.sum(axis=1)))),
                'counters':run['counters'],'capacitance':cap,'charge_slope_e_per_V':float(slope),
                'charge_slope_difference_e_per_V':float(slope-cap['dcharge_dvoltage_e_per_V']),
                'coordinate':short,'conjugate_charge':q@pattern,'solver':pair}
    finally:
        sim.h.close()


def interface_case(workers,extended=False):
    start = time.perf_counter()
    seeds = CONFIRMATION_SEEDS if extended else SEEDS
    with ProcessPoolExecutor(max_workers=workers) as pool:
        chains = list(pool.map(interface_chain,((seed,extended) for seed in seeds)))
    coord = np.asarray([x.pop('coordinate') for x in chains])
    charge = np.asarray([x.pop('conjugate_charge') for x in chains])
    return {'wall_seconds_with_analysis':time.perf_counter()-start,'workers':workers,
            'input':'inputs/nacl_spce_constant_potential.json','chains':chains,
            'Na-O':chain_mean_interval(coord[:,:,0]), 'Cl-O':chain_mean_interval(coord[:,:,1]),
            'electrode_charge_e':chain_mean_interval(charge),
            'mean_chain_capacitance_uF_cm2':independent_interval(
                [x['capacitance']['capacitance_uF_cm2'] for x in chains]),
            'median_moves_per_second':statistics.median(x['moves_per_second'] for x in chains)}


def _direct_state_energies(h,uid,states):
    i=h.molecule_index(uid)
    values=[]
    for j in states:
        m=h.molecules[i].clone()
        m.state=j
        values.append(h.propose([i],[m])['delta_energy'])
    return np.array(values)


def reaction_chain(arg):
    kind,seed,extended=arg
    c=model(f'hydrated_{kind}_calibration')
    c['seed']=seed
    cal=c['calibration'].copy()
    idx=cal.pop('edge_index')
    if extended:
        cal.update(samples_per_window=600,decorrelation_moves=20,burnin_moves=4000)
    else:
        cal.update(samples_per_window=200,decorrelation_moves=8,burnin_moves=1000)
    sim=build(c)
    try:
        states=sim.reactions[idx].bridge
        uid=cal['uid']
        if sim.h.molecules[sim.h.molecule_index(uid)].template!=sim.reactions[idx].template:
            raise RuntimeError('reaction UID and template differ')
        start=time.perf_counter()
        ans=sample_reaction_windows(sim,bridge_states=states,**cal)
        seconds=time.perf_counter()-start
        ht=sim.h
        timings=[]
        for i in range(16):
            for _ in range(30):
                sim.displace()
            methods={'cached':state_energy_differences,'direct':_direct_state_energies}
            values={}
            durations={}
            order=('direct','cached') if i%2 else ('cached','direct')
            for m in order:
                t0=time.thread_time_ns()
                values[m]=methods[m](ht,uid,states)
                durations[m]=(time.thread_time_ns()-t0)/1e3
            error=float(np.max(np.abs(values['direct']-values['cached'])))
            if error>1e-6:
                raise RuntimeError('reaction cache does not match state-wise reference')
            timings.append({'maximum_difference_K':error,'cpu_microseconds':durations})
        ends=[ht.templates[sim.reactions[idx].template].states[j].name for j in (states[0],states[-1])]
        np.savez_compressed(HERE/f'{kind}_{seed}.npz',u_kn=ans['u_kn'],n_k=ans['n_k'],
                            free_energy_K=ans['free_energy_kelvin'],overlap=ans['overlap'])
        return {'kind':kind,'seed':seed,'physical_states':ends,'options':cal,
                'production_and_equilibration_seconds':seconds,'timing_pairs':timings,
                'relative_free_energy_K':ans['free_energy_kelvin'],
                'overlap_gap':ans['overlap_spectral_gap'],'u_kn':ans['u_kn'],'n_k':ans['n_k'],
                'temperature_K':sim.temperature}
    finally:
        sim.h.close()


def reaction_case(workers,extended=False):
    start=time.perf_counter()
    seeds=CONFIRMATION_SEEDS if extended else SEEDS
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results=list(pool.map(reaction_chain,((k,s,extended) for k in ('redox','acid') for s in seeds)))
    output={}
    for kind in ('redox','acid'):
        chains=[x for x in results if x['kind']==kind]
        traces=[x.pop('u_kn') for x in chains]
        counts=np.asarray([x.pop('n_k') for x in chains])
        slices=np.r_[0,np.cumsum(counts[0])]
        total=np.concatenate([np.concatenate([u[:,i:j] for u in traces],axis=1)
                              for i,j in zip(slices[:-1],slices[1:])],axis=1)
        joined=MBAR(total,counts.sum(axis=0))
        temperature=chains[0]['temperature_K']
        factor=temperature*KB*NA/1000
        shift=independent_interval([x['relative_free_energy_K'][-1]*KB*NA/1000 for x in chains])
        pairs=[r for x in chains for r in x['timing_pairs']]
        adjacent=[]
        for w,(lo,hi) in enumerate(zip(slices[:-1],slices[1:])):
            p,q=(w,w+1) if w+1<len(slices)-1 else (w-1,w)
            adjacent.append(rank_split_rhat([u[q,lo:hi]-u[p,lo:hi] for u in traces]))
        output[kind]={'chains':chains,'pooled_delta_G_kJ_mol':float(joined.f[-1]*factor),
                      'replica_delta_G_kJ_mol':shift,'pooled_overlap_gap':joined.spectral_gap,
                      'adjacent_gap_rhat_by_window':adjacent,
                      'cache_speedup_median':statistics.median(x['cpu_microseconds']['direct']/
                                                              x['cpu_microseconds']['cached'] for x in pairs),
                      'cache_max_error_K':max(x['maximum_difference_K'] for x in pairs)}
    return {'wall_seconds':time.perf_counter()-start,'workers':workers,'cases':output}


def transport_case():
    names=('ferricyanide_chrono','ferricyanide_cv_100mVs','ferricyanide_cv_200mVs')
    results={}
    for name in names:
        c=model(name)
        base=c['transport']
        rows=[]
        for nodes in (base['cells']//2,base['cells'],2*base['cells']):
            sim=DiffusionRedox(**{**base,'cells':nodes})
            start=time.perf_counter()
            ans=sim.run(**c['protocol'])
            wall=time.perf_counter()-start
            tc,j,v=(ans[x] for x in ('time_s','faradaic_current_A','potential_V'))
            row={'cells':nodes,'wall_seconds':wall,'rhs_evaluations':ans['rhs_evaluations'],
                 'minimum_concentration_mol_m3':ans['minimum_concentration_mol_m3']}
            scale=FARADAY*sim.area_m2*sim.bulk_oxidized_mol_m3*math.sqrt(sim.diffusivity_oxidized_m2_s)
            if 'chrono' in name:
                mask=tc>0
                reference=-scale/np.sqrt(np.pi*tc[mask])
                row.update(log_slope=float(np.polyfit(np.log(tc[mask]),np.log(-j[mask]),1)[0]),
                           max_relative_error=float(np.max(np.abs(j[mask]/reference-1))),
                           final_microamp=float(j[-1]*1e6))
            else:
                period=c['protocol']['times_s'][1]
                rate=abs(v[-1]-v[np.argmin(v)])/period
                forward=np.flatnonzero(tc<=period)
                backward=np.flatnonzero(tc>=period)
                i=forward[np.argmin(j[forward])]
                k=backward[np.argmax(j[backward])]
                ref=.4463*scale*math.sqrt(FARADAY*rate/(KB*NA*sim.temperature_K))
                row.update(peak_cathodic_microamp=float(j[i]*1e6),
                           randles_sevcik_microamp=float(ref*1e6),
                           peak_rel_error=float(abs(-j[i]/ref-1)),
                           peak_separation_mV=float(1e3*(v[k]-v[i])))
            np.savez_compressed(HERE/f'{name}_{nodes}.npz',time_s=tc,current_A=j,potential_V=v)
            rows.append(row)
        results[name]=rows
    ratio=[b['peak_cathodic_microamp']/a['peak_cathodic_microamp'] for a,b in
           zip(results[names[1]],results[names[2]])]
    return {'cases':results,'peak_ratio_200_over_100':ratio,'sqrt_2':math.sqrt(2.)}


def _adsorption_run(tag,config,sampling,cache,seed,steps,burnin):
    case=copy.deepcopy(config)
    c=case['sampling']
    c.update(sampling,seed=seed,steps=steps,burnin=burnin,sample_every=40,record_energy=False)
    if c.get('sampling_mode')=='delayed':
        c['host_field_cache']=str(cache.resolve())
    start=time.perf_counter()
    sim,opts,_,_,_=construct_case(case,HERE/'inputs')
    setup=time.perf_counter()-start
    with sim:
        run=sim.run(**opts)
        wall=time.perf_counter()-start
        profile=sim.profile
        entries=sim.entries()
        closure=abs(sim.total_energy()-sim.potential.total_energy(entries))
        if closure>1e-5:
            raise RuntimeError(f'rebuilt adsorption energy differs by {closure:g} K')
        path=HERE/f'{tag}_{seed}.npz'
        np.savez_compressed(path,counts=np.asarray(run['counts']),sample_steps=np.asarray(run['sample_steps']))
    moves=run['acceptance']
    accepts=sum(v['accepted'] for v in moves.values())
    production_sec=run['production_seconds']
    return {'seed':seed,'setup_seconds':setup,'wall_seconds':wall,'production_seconds':production_sec,
            'attempts_per_second':run['attempted_moves_per_second'],'energy_closure_K':closure,
            'accepts_per_second_including_warmup':accepts/run['elapsed_seconds'],
            'final_count':run['final_counts'][0],'moves_with_warmup':moves,
            'guest_cell_updates':profile['guest_local_updates'],'host_grid':profile['acceleration'].get('host_grid'),
            'sample_file':path.name,'count_samples':[row[0] for row in run['counts']]}


def xylene_case(cache):
    configs=('nitpt_pxylene_0p1Pa','nitpt_pxylene_higher_activity')
    result={}
    for k,name in enumerate(configs):
        c=model(name)
        modes={'exact':{'sampling_mode':'exact'},
               'grid32':{'sampling_mode':'delayed','host_grid_mesh':32},
               'grid64':{'sampling_mode':'delayed','host_grid_mesh':64}}
        if k:
            modes.pop('exact')
        steps,burnin=(16000,5000) if k==0 else (52000,12000)
        runs={m:[] for m in modes}
        keys=list(modes)
        for i,seed in enumerate(SEEDS):
            offset=i%len(keys)
            for mode in keys[offset:]+keys[:offset]:
                run=_adsorption_run(f'{name}_{mode}',c,
                                    {**modes[mode],'guest_update':'incremental'},cache,seed,steps,burnin)
                runs[mode].append(run)
                print(f'{name} {mode} seed {seed}: {run["wall_seconds"]:.2f} s',flush=True)
        summaries={}
        base=runs[keys[0]]
        for m,rows in runs.items():
            summary=chain_mean_interval([r.pop('count_samples') for r in rows])
            summaries[m]={'chains':rows,'uptake_per_supercell':summary,
                          'median_attempts_s':statistics.median(r['attempts_per_second'] for r in rows),
                          'median_wall_speedup':statistics.median(r['wall_seconds']/v['wall_seconds']
                                                                 for r,v in zip(base,rows)),
                          'median_production_speedup':statistics.median(v['attempts_per_second']/r['attempts_per_second']
                                                                       for r,v in zip(base,rows)),
                          'max_energy_closure_K':max(r['energy_closure_K'] for r in rows)}
        result[name]={'conditions':c['reservoir'],'framework_atoms':len(c['host']['positions_A']),
                      'comparison_baseline':keys[0],'steps':steps,'burnin':burnin,'modes':summaries}
    return {'cases':result}


XYLENE_FUGACITIES_PA=(.006,.013,.027,.048,.073,.10,.18)


def _xylene_ladder(job):
    seed,cache=job
    c=model('nitpt_pxylene_0p1Pa')
    reservoir=c['reservoir']['activities_A3_inverse'][0]
    activities=reservoir*np.asarray(XYLENE_FUGACITIES_PA)/.10
    c['reservoir']['activities_A3_inverse']=[float(activities[0])]
    c['sampling'].update(seed=seed,sampling_mode='delayed',host_grid_mesh=32,
                         grid_root_bias=.65,guest_update='incremental',host_field_cache=str(cache))
    start=time.perf_counter()
    first,_,_,_,_=construct_case(c,HERE/'inputs')
    counts=[[] for _ in activities]
    with ExitStack() as stack:
        walkers=[stack.enter_context(first)]
        for index,z in enumerate(activities[1:],1):
            walkers.append(stack.enter_context(FastGCMC(
                first.species,first.potential,first.cell,first.temperature,[z],
                seed=seed+index,trials=first.trials,move_weights=first.move_weights,
                max_translation=first.max_translation,max_rotation=first.max_rotation,
                **first._acceleration_options())))
        rex=GrandCanonicalReplicaExchange(walkers,seed=seed+103)
        for _ in range(40):
            rex.advance(1000,sample_every=40)
        visited=[0]*len(walkers)
        round_trips=0
        for _ in range(160):
            draws=rex.advance(1000,sample_every=40)
            for record,point in zip(counts,draws):
                record.extend(row[0] for row in point['counts'])
            low,high=rex.occupants[0],rex.occupants[-1]
            if visited[high]==1:
                visited[high]=2
            if visited[low]==2:
                round_trips+=1
            visited[low]=1
    name=f'xylene_activity_ladder_{seed}.npz'
    np.savez_compressed(HERE/name,counts=np.asarray(counts),fugacity_Pa=XYLENE_FUGACITIES_PA)
    return {'seed':seed,'counts':counts,'round_trips':round_trips,
            'edge_exchange_acceptance':(rex.accepts/rex.attempts).tolist(),
            'wall_s':time.perf_counter()-start,'sample_file':name}


def xylene_exchange_case(workers,cache):
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows=list(pool.map(_xylene_ladder,((s,cache.resolve()) for s in CONFIRMATION_SEEDS)))
    counts=np.asarray([x.pop('counts') for x in rows])
    return {'points':[{'fugacity_Pa':p,'uptake_per_supercell':chain_mean_interval(counts[:,i])}
                      for i,p in enumerate(XYLENE_FUGACITIES_PA)],
            'ladders':rows,'warmup_rounds':40,'production_rounds':160,'attempts_per_round':1000,
            'wall_s':time.perf_counter()-start}


def _count_butane_conformers(entries):
    if not entries:
        return np.zeros(3,dtype=int)
    xyz=np.asarray([positions for _,positions in entries])
    a,b,c=xyz[:,0]-xyz[:,1],xyz[:,2]-xyz[:,1],xyz[:,3]-xyz[:,2]
    b/=np.linalg.norm(b,axis=1)[:,None]
    v=a-np.sum(a*b,axis=1)[:,None]*b
    w=c-np.sum(c*b,axis=1)[:,None]*b
    phi=np.arctan2(np.sum(np.cross(b,v)*w,axis=1),np.sum(v*w,axis=1))
    trans=np.abs(phi)>=2*np.pi/3
    return np.array([trans.sum(),np.sum(~trans&(phi>=0)),np.sum(~trans&(phi<0))])


def _butane_chain(config,mode,cache,seed,steps,burnin,stride):
    c=copy.deepcopy(config)
    c['sampling'].update(seed=seed,partial_regrowth=mode=='suffix')
    if c['sampling']['sampling_mode']=='delayed':
        c['sampling']['host_field_cache']=str(cache)
    start=time.perf_counter()
    sim,_,host,_,_=construct_case(c,HERE/'inputs')
    setup=time.perf_counter()-start
    frames=[]
    elapsed_kernel=0.
    with sim:
        sim.run(0,burnin=burnin,sample_every=stride,record_energy=False)
        warm_regrow=sim.counters['regrow']
        production=time.perf_counter()
        for _ in range(steps//stride):
            row=sim.run(stride,sample_every=stride,record_energy=False)
            elapsed_kernel+=row['production_seconds']
            conformers=_count_butane_conformers(sim.entries())
            if conformers.sum()!=row['counts'][0][0]:
                raise RuntimeError('conformer counts disagree with the accepted state')
            frames.append(conformers)
        analysis_wall=time.perf_counter()-production
        total_wall=time.perf_counter()-start
        entries=sim.entries()
        closure=abs(sim.total_energy()-sim.potential.total_energy(entries))
        if closure>1e-5:
            raise RuntimeError(f'butane full energy disagrees by {closure:g} K')
        regrow={k:v-warm_regrow[k] for k,v in sim.counters['regrow'].items()}
    xyz_counts=np.asarray(frames)
    sample_file=HERE/f'butane_{mode}_{seed}.npz'
    np.savez_compressed(sample_file,conformer_counts=xyz_counts,
                        steps=np.arange(burnin+stride,burnin+steps+1,stride))
    return {'seed':seed,'steps':steps,'burnin':burnin,'sample_every':stride,
            'frame_counts':xyz_counts,'host_mass_amu':host.mass_amu,
            'production_kernel_s':elapsed_kernel,'production_with_observables_s':analysis_wall,
            'setup_seconds':setup,'wall_seconds':total_wall,'regrow':regrow,
            'final_energy_closure_K':closure,'sample_file':sample_file.name}


def _summarize_butane(records):
    counts=np.asarray([r.pop('frame_counts') for r in records])
    total=counts.sum(axis=2)
    mean_n=float(total.mean())
    if mean_n<=0:
        raise RuntimeError('no observed butane loading')
    fraction=float(counts[:,:,0].mean()/mean_n)
    linearized=fraction+(counts[:,:,0]-fraction*total)/mean_n
    trans=chain_mean_interval(linearized)
    number=chain_mean_interval(total)
    wall=sum(r['wall_seconds'] for r in records)
    return {'chains':records,'trans_fraction':trans,'number_per_cell':number,
            'mean_gauche_plus_fraction':float(counts[:,:,1].mean()/mean_n),
            'mean_gauche_minus_fraction':float(counts[:,:,2].mean()/mean_n),
            'trans_ESS_per_s':trans['ess']/wall,'loading_ESS_per_s':number['ess']/wall,
            'median_kernel_attempts_s':statistics.median(r['steps']/r['production_kernel_s'] for r in records),
            'regrowth_acceptance':sum(r['regrow']['accepted'] for r in records)/
                                  sum(r['regrow']['attempted'] for r in records)}


def _butane_pair(job):
    i,seed,cache,extended=job
    config=model('flexible_butane_adsorption')
    result={}
    steps,burnin,stride=(840000,100000,240) if extended else (240000,35000,120)
    for mode in (('full','suffix') if i%2==0 else ('suffix','full')):
        result[mode]=_butane_chain(config,mode,cache,seed,steps,burnin,stride)
    if not extended:
        result['ideal']=_butane_chain(model('flexible_butane_ideal'),'ideal',cache,seed,120000,5000,100)
    print(f'butane paired seed {i+1}: {steps} production attempts per adsorption mode',flush=True)
    return result


def butane_case(cache,workers,extended=False):
    config=model('flexible_butane_adsorption')
    gas=model('flexible_butane_ideal')
    cache=cache.resolve()
    rng=np.random.SeedSequence(735179 if extended else 172934)
    seeds=[int(x) for x in rng.generate_state(6)]
    jobs=[(i,seed,cache,extended) for i,seed in enumerate(seeds)]
    if extended:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pairs=list(pool.map(_butane_pair,jobs))
    else:
        pairs=[_butane_pair(job) for job in jobs]
    results={mode:[pair[mode] for pair in pairs] for mode in pairs[0]}
    ratio=statistics.median(a['wall_seconds']/b['wall_seconds']
                            for a,b in zip(results['full'],results['suffix']))
    differences=[]
    uptake_differences=[]
    for a,b in zip(results['full'],results['suffix']):
        fa,fb=a['frame_counts'],b['frame_counts']
        differences.append(float(fb[:,0].sum()/fb.sum()-fa[:,0].sum()/fa.sum()))
        uptake_differences.append(float(fb.sum(axis=1).mean()-fa.sum(axis=1).mean()))
    return {'modes':{key:_summarize_butane(rows) for key,rows in results.items()},
            'analytical_ideal':gas['metadata'],
            'adsorption_model':config['metadata'],'reservoir':config['reservoir'],
            'median_wall_speedup_suffix_over_full':ratio,
            'paired_suffix_minus_full_trans_fraction':independent_interval(differences),
            'paired_suffix_minus_full_loading':independent_interval(uptake_differences)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',choices=('interface','reaction','transport','xylene','xylene_exchange','butane'))
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--cache',type=Path,default=HERE/'field_cache')
    parser.add_argument('--extended',action='store_true',help='use independent longer electrochemical runs')
    args=parser.parse_args()
    if args.case=='transport':
        report=transport_case()
    elif args.case=='xylene':
        report=xylene_case(args.cache)
    elif args.case=='xylene_exchange':
        report=xylene_exchange_case(args.workers,args.cache)
    elif args.case=='butane':
        report=butane_case(args.cache,args.workers,args.extended)
    else:
        report={'interface':interface_case,'reaction':reaction_case}[args.case](args.workers,args.extended)
    report['platform']=platform.platform()
    suffix='_extended' if args.extended else ''
    _publish(HERE/f'{args.case}{suffix}_benchmark.json',report)
    print(f'{args.case}: saved benchmark',flush=True)


if __name__=='__main__':
    main()
