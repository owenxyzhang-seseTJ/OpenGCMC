#!/usr/bin/env python3
"""Paired single-core PoreMC 0.6 / 0.7 full-chain measurements at the same activity.

Pass a real input JSON or --synthetic low/mid/dense/high for the supplied workload fixture.
Each run is a fresh process. For PoreMC 0.6 the file cache is disabled by default.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
MODES = ('exact','delayed32','delayed64')


def synthetic_config(label):
    import numpy as np
    sys.path.insert(0, str(ROOT / 'tools06'))
    from fixtures import workload_fixture
    mol, params, _ = workload_fixture('aromatic12')
    n, activity_v = {'low': (5,2.8), 'mid': (18,10.5), 'dense': (30,18.), 'high': (30,24.)}[label]
    # Three planes in each slit allow independent placement of thirty molecules.
    starts = []
    for i in range(n):
        pos = np.array([5+9*(i%5), 7+16*((i//5)%3), 12+24*(i//15)])
        starts.append({'species':0,'positions_A':(mol.rigid_positions+pos).tolist()})
    return dict(description='Synthetic corrugated slit with an uncalibrated rigid aromatic guest.',
                temperature_K=350., cell_A=params['cell'].tolist(),species=[mol.to_dict()],
                reservoir={'activities_A3_inverse':[activity_v/48**3]},
                potential=dict(coulomb='ewald',algorithm='cached',cutoff_A=10.,alpha=.28,kmax=[13]*3,
                               scale14_lj=.5,scale14_coulomb=.75),
                sampling=dict(trials=10,move_weights=params['weights'],max_translation=.65,max_rotation=.35),
                initial_particles=starts,host={
                    name:params[key].tolist() for name,key in [
                        ('positions_A','host_positions'),('sigma_A','host_sigma'),
                        ('epsilon_K','host_epsilon'),('charge_e','host_charge')]})


def sha_bytes(blob):
    return hashlib.sha256(blob).hexdigest()


def worker(args):
    source = ROOT if args.release=='07' else ROOT/'baseline_06'
    sys.path.insert(0,str(source))
    from poremc.cli import construct_case,clean_json
    config = synthetic_config(args.synthetic) if args.synthetic else json.loads(args.input.read_text())
    base = ROOT if args.synthetic else args.input.resolve().parent
    config=copy.deepcopy(config)
    p=config.setdefault('potential',{})
    if p.get('coulomb')!='ewald': raise ValueError('A classical Ewald input is required.')
    p.update(backend='native',threads=1)
    s=config.setdefault('sampling',{})
    for key in ('host_grid_cache','cuda_library'):
        s.pop(key,None)
    mode=args.worker_mode
    delayed=mode!='exact'
    s.update(driver='cpp', sampling_mode='delayed' if delayed else 'exact',
             reciprocal_method='direct',reciprocal_backend='cpu',ewald_k_cutoff=0.,
             ewald_half_spectrum=True,grid_root_bias=0.,seed=args.seed)
    s.setdefault('host_grid_steep','boltzmann')
    if delayed:
        s['host_grid_mesh']=int(mode[7:])
        if args.release=='07' and not args.no_cache:
            s['host_grid_cache']=str(args.cache_dir.resolve()/f'{mode}.bin')
    start=time.perf_counter()
    sim,_,host,physics,_=construct_case(config,base)
    try:
        init=time.perf_counter()-start
        initial=sim.total_energy()
        if args.burnin:
            sim.run(args.burnin,sample_every=args.burnin,record_energy=False)
        before=sim.counters
        result=sim.run(args.steps,sample_every=args.sample_every,record_energy=False)
        after=sim.counters
        elapsed=result['production_seconds']
        counters={m:{c:after[m][c]-before[m][c] for c in ('attempted','accepted')} for m in before}
        counts=[int(x[0]) for x in result['counts']]
        coords=[(int(t),x.tolist()) for t,x in sim.entries()]
        digest=sha_bytes(json.dumps(coords,separators=(',',':')).encode())
        final=sim.total_energy()
        reference=None
        if args.independent:
            # Same full NumPy finite-k reference in both releases; chunk scratch
            # independently of the native calculation to bound the reference memory.
            sys.path.insert(0,str(ROOT/'tools06'))
            from benchmark_hostgrid import endpoint_reference
            reference=endpoint_reference(sim.potential,sim.entries())
        out=dict(version=args.release,mode=mode,seed=args.seed,host_sites=len(host.positions),physical_input=physics,
                 initial_energy_K=initial,final_energy_K=final,native_sha256=sha_bytes(Path(sim._lib._name).read_bytes()),
                 initial_N=len(config.get('initial_particles',[])),mean_N=statistics.mean(counts),
                 steps=args.steps,counts=counts,counters=counters,construction_seconds=init,production_seconds=elapsed,
                 attempts_s=args.steps/elapsed,accepted_s=sum(v['accepted'] for v in counters.values())/elapsed,
                 final_positions_sha256=digest,full_reference_K=reference,
                 error_K=abs(final-reference) if reference is not None else None,
                 acceleration=sim.acceleration_profile,long_range=sim.long_range_profile)
        print(json.dumps(clean_json(out),allow_nan=False))
    finally: sim.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--input',type=Path)
    source.add_argument('--synthetic',choices=('low','mid','dense','high'))
    p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--burnin',type=int,default=1000)
    p.add_argument('--sample-every',type=int,default=10)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--seed',type=int,default=7107)
    p.add_argument('--modes',default=','.join(MODES))
    p.add_argument('--no-cache',action='store_true',help='Keep the 0.7 grid in memory only; 0.6 always uses memory only.')
    p.add_argument('--independent',action='store_true',help='Evaluate the full finite-k independent energy after timing each row.')
    p.add_argument('--cache-dir',type=Path,help='Optional persistent cache directory. Otherwise a fresh temporary directory is used for the batch.')
    p.add_argument('--output',type=Path,default=Path('comparison07.json'))
    p.add_argument('--release',choices=('06','07'),help=argparse.SUPPRESS)
    p.add_argument('--worker-mode',choices=MODES,help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.release: return worker(args)
    if not 1<=args.sample_every<=args.steps or args.repeats<1 or args.burnin<0:p.error('Invalid run length or sampling interval.')
    modes=args.modes.split(',')
    if not modes or set(modes)-set(MODES):p.error('Choose from exact,delayed32,delayed64.')
    for directory in (ROOT/'baseline_06',ROOT):
        ext='.dylib' if sys.platform=='darwin' else '.so'
        if not (directory/'poremc'/('libporemc'+ext)).is_file():
            subprocess.run([sys.executable,str(directory/'native/build.py')],cwd=directory,check=True)
    affinity=None
    if hasattr(os,'sched_getaffinity'):
        affinity=min(os.sched_getaffinity(0))
        os.sched_setaffinity(0,{affinity})
    env={**os.environ,**dict.fromkeys(('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'),'1')}
    with tempfile.TemporaryDirectory() as tmp:
        cache=(args.cache_dir or Path(tmp)).resolve()
        rows=[]
        for rep in range(args.repeats):
            for mode in (modes if rep%2==0 else reversed(modes)):
                for ver in (('06','07') if rep%2==0 else ('07','06')):
                    cmd=[sys.executable,str(Path(__file__).resolve()),'--release',ver,'--worker-mode',mode,
                         '--cache-dir',str(cache),'--steps',str(args.steps),'--burnin',str(args.burnin),
                         '--sample-every',str(args.sample_every),'--seed',str(args.seed+rep)]
                    cmd+=['--synthetic',args.synthetic] if args.synthetic else ['--input',str(args.input.resolve())]
                    if args.independent:cmd+=['--independent']
                    if args.no_cache:cmd+=['--no-cache']
                    row=json.loads(subprocess.check_output(cmd,text=True,env=env))
                    rows.append(row)
                    print(f"{ver} {mode} {row['seed']}  N={row['mean_N']:.2f}  {row['attempts_s']:.1f} moves/s",flush=True)
    summary=[]
    for mode in modes:
        old=[r for r in rows if r['version']=='06' and r['mode']==mode]
        new=[r for r in rows if r['version']=='07' and r['mode']==mode]
        ratios=[]
        for a,b in zip(old,new):
            if (a['seed']!=b['seed'] or a['physical_input']!=b['physical_input'] or
                abs(a['initial_energy_K']-b['initial_energy_K'])>1e-5):
                raise RuntimeError('Paired input mismatch.')
            ratios.append(b['attempts_s']/a['attempts_s'])
        summary.append(dict(mode=mode,old_attempts_s=statistics.median(r['attempts_s'] for r in old),
                            new_attempts_s=statistics.median(r['attempts_s'] for r in new),
                            old_accepted_s=statistics.median(r['accepted_s'] for r in old),
                            new_accepted_s=statistics.median(r['accepted_s'] for r in new),
                            old_mean_N=statistics.mean(r['mean_N'] for r in old),
                            new_mean_N=statistics.mean(r['mean_N'] for r in new),
                            paired_speedups=ratios,median_speedup=statistics.median(ratios)))
    result=dict(source='synthetic '+args.synthetic if args.synthetic else str(args.input),platform=platform.platform(),
                affinity_cpu=affinity,threads=1,production_attempts_per_rep=args.steps,burnin=args.burnin,
                repeats=args.repeats,cache_policy='0.6 off; 0.7 '+('off' if args.no_cache else 'on (raw fields)'),
                summary=summary,rows=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
