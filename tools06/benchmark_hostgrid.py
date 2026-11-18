#!/usr/bin/env python3
"""Paired 0.5/0.6 complete-loop CPU timings on disclosed synthetic fixtures.

The user's RASPA2 HPC rates are context, not measurements made by this script.
The large workload matches counts/algorithmic dimensions, not an absent CIF
or absent molecular force field. Different acceptance algorithms generate
different trajectories; throughput and accepted rates are both recorded.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def endpoint_reference(potential,entries):
    """Full NumPy finite-k reference, with bounded temporary phase storage.

    This is installed only after the timed native run.  It leaves every
    physical term and mode unchanged; it changes the reference reduction's
    allocation size, allowing concurrent long-chain validation on modest RAM.
    The original baseline files and native production path are untouched.
    """
    import numpy as np
    from poremc.electrostatics import EwaldSnapshot
    ewald=potential.ewald
    def snapshot(positions,charges):
        x=np.asarray(positions,float).reshape(-1,3);q=np.asarray(charges,float).reshape(-1)
        structure=np.zeros(len(ewald.k),dtype=complex)
        for first in range(0,len(q),128):
            stop=first+128
            structure+=(q[first:stop,None]*np.exp(1j*(x[first:stop]@ewald.k.T))).sum(axis=0)
        return EwaldSnapshot(x,q,structure,float(q.sum()))
    ewald.snapshot=snapshot
    return potential.total_energy(entries)


def worker(args):
    sys.path.insert(0, str(Path(args.source).resolve()))
    import numpy as np
    import poremc
    from poremc.fast import FastGCMC
    from poremc.cli import clean_json
    from fixtures import workload_fixture, make_potential, simulation_arguments, input_hash

    molecule, physics, initial = workload_fixture(args.case, args.host_atoms, args.kmax)
    setup_start=time.perf_counter()
    potential=make_potential(molecule, physics)
    options={}
    if args.mode != 'baseline05':
        options.update(sampling_mode='exact', ewald_half_spectrum=True)
    if args.mode.startswith(('delayed','biased')):
        biased=args.mode.startswith('biased')
        mesh=int(args.mode.removeprefix('biased' if biased else 'delayed'))
        cache=Path(args.cache_dir)/f'{args.case}-{args.host_atoms}-k{args.kmax}-m{mesh}.bin'
        cache.parent.mkdir(parents=True, exist_ok=True)
        existed=cache.exists()
        options.update(sampling_mode='delayed',host_grid_mesh=mesh,host_grid_cap=1e5,
                       host_grid_cache=str(cache),grid_root_bias=.95 if biased else 0.,
                       host_grid_steep=args.steep)
    else:
        cache=None;existed=False
    sim=FastGCMC(molecule,potential,physics['cell'],physics['temperature'],[physics['activity']],
                 seed=args.seed,**simulation_arguments(physics),**options)
    construction_s=time.perf_counter()-setup_start
    for positions in initial:sim.add_particle(0,positions)
    initial_energy=sim.total_energy()
    if args.burnin:
        sim.run(args.burnin,sample_every=max(1,args.burnin),record_energy=False)
    before=sim.counters
    result=sim.run(args.steps,sample_every=args.sample_every,record_energy=False)
    after=sim.counters
    counters={name:{key:after[name][key]-before[name][key] for key in ('attempted','accepted')}
              for name in sim.MOVE_NAMES}
    for value in counters.values():
        value['acceptance']=value['accepted']/value['attempted'] if value['attempted'] else 0.
    final=sim.total_energy()
    reference=endpoint_reference(potential,sim.entries())
    profile=sim.profile
    if hasattr(sim,'acceleration_profile'):
        profile['acceleration']=sim.acceleration_profile
    counts=[row[0] for row in result['counts']]
    row=dict(mode=args.mode,case=args.case,version=poremc.__version__,module_path=poremc.__file__,
             host_grid_steep=args.steep if args.mode.startswith(('delayed','biased')) else None,
             seed=args.seed,steps=args.steps,burnin=args.burnin,sample_every=args.sample_every,
             physics_sha256=input_hash(molecule,physics,initial),
             construction_seconds=construction_s,cache_existed_before=existed,
             cache_bytes=cache.stat().st_size if cache and cache.exists() else None,
             production_seconds=result['production_seconds'],
             attempted_moves_per_second=result['attempted_moves_per_second'],
             accepted_moves_per_second=sum(v['accepted'] for v in counters.values())/result['production_seconds'],
             acceptance=counters,initial_energy_K=initial_energy,final_energy_K=final,
             final_recompute_error_K=abs(final-reference),initial_molecules=len(initial),
             endpoint_reference='full NumPy finite-k sum, atom chunks of128; outside production timing',
             final_molecules=int(sim.counts[0]),count_mean=float(np.mean(counts)) if counts else None,
             count_range=[int(min(counts)),int(max(counts))] if counts else None,count_trace=counts,
             host_atoms=args.host_atoms,guest_sites=molecule.n_atoms,trials=physics['trials'],
             finite_full_modes=(2*args.kmax+1)**3-1,profile=profile,
             native_sha256=sha(Path(args.source)/'poremc/libporemc.so'),
             source_fast_mc_sha256=sha(Path(args.source)/'native/fast_mc.cpp'))
    sim.close()
    print(json.dumps(clean_json(row),allow_nan=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',action='store_true')
    p.add_argument('--source',default=str(ROOT))
    p.add_argument('--baseline',default=str(ROOT/'baseline_05'))
    p.add_argument('--case',choices=['aromatic12','flexible12'],default='aromatic12')
    p.add_argument('--host-atoms',type=int,default=4416)
    p.add_argument('--kmax',type=int,default=13)
    p.add_argument('--steps',type=int,default=1500)
    p.add_argument('--burnin',type=int,default=200)
    p.add_argument('--sample-every',type=int,default=10)
    p.add_argument('--seed',type=int,default=9306)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--mode',default='baseline05')
    p.add_argument('--modes',default='baseline05,native06,delayed32,delayed64')
    p.add_argument('--steep',choices=['linear','boltzmann'],default='boltzmann')
    p.add_argument('--cache-dir',default=str(ROOT/'evidence06/grid_cache'))
    p.add_argument('--output',default=str(ROOT/'evidence06/hostgrid_benchmark.json'))
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.worker:return worker(args)
    modes=args.modes.split(',')
    if 'baseline05' not in modes:raise ValueError('paired run must include baseline05')
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    data=dict(purpose='synthetic workload-matched PoreMC 0.5/0.6 CPU comparison; not direct RASPA timing',
              case=args.case,host_atoms=args.host_atoms,guest_sites=12,kmax=args.kmax,
              steps=args.steps,burnin=args.burnin,repeats=args.repeats,modes=modes,
              surrogate_steep_interpolation=args.steep,
              platform=platform.platform(),python=sys.version,rows=[],summary=[])
    try:data['cpu_model']=next(x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name'))
    except (OSError,StopIteration):pass
    if args.resume and output.exists():
        previous=json.loads(output.read_text())
        for key in ('case','host_atoms','kmax','steps','burnin','repeats','modes','surrogate_steep_interpolation'):
            if previous[key]!=data[key]:raise ValueError('resume mismatch: '+key)
        data=previous;data['summary']=[]
    def save():
        temp=output.with_suffix('.tmp')
        temp.write_text(json.dumps(data,indent=2,allow_nan=False));temp.replace(output)
    for repeat in range(args.repeats):
        order=modes if repeat%2==0 else list(reversed(modes))
        for mode in order:
            seed=args.seed+repeat
            completed=[r for r in data['rows'] if r['mode']==mode and r['seed']==seed]
            if completed:continue
            source=args.baseline if mode=='baseline05' else str(ROOT)
            cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--source',source,
                 '--mode',mode,'--case',args.case,'--host-atoms',str(args.host_atoms),
                 '--kmax',str(args.kmax),'--steps',str(args.steps),'--burnin',str(args.burnin),
                 '--sample-every',str(args.sample_every),'--seed',str(seed),'--cache-dir',args.cache_dir]
            cmd+=['--steep',args.steep]
            row=json.loads(subprocess.check_output(cmd,env=env,text=True))
            data['rows'].append(row);save()
            print(f"{args.case} repeat={repeat} {mode}: {row['attempted_moves_per_second']:.2f} moves/s, "
                  f"{row['accepted_moves_per_second']:.2f} accepted/s, meanN={row['count_mean']:.2f}",flush=True)
    baseline={r['seed']:r for r in data['rows'] if r['mode']=='baseline05'}
    for mode in modes:
        rows=[r for r in data['rows'] if r['mode']==mode]
        ratios=[]
        for row in rows:
            ref=baseline[row['seed']]
            assert row['physics_sha256']==ref['physics_sha256']
            assert abs(row['initial_energy_K']-ref['initial_energy_K'])<1e-5
            ratios.append(row['attempted_moves_per_second']/ref['attempted_moves_per_second'])
        data['summary'].append(dict(mode=mode,median_moves_s=statistics.median(r['attempted_moves_per_second'] for r in rows),
            median_accepted_s=statistics.median(r['accepted_moves_per_second'] for r in rows),
            paired_speedups=ratios,median_speedup=statistics.median(ratios),speedup_range=[min(ratios),max(ratios)],
            max_final_recompute_error_K=max(r['final_recompute_error_K'] for r in rows)))
    save();print(json.dumps(data['summary'],indent=2))


if __name__=='__main__':main()
