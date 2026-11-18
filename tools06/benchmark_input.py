#!/usr/bin/env python3
"""Compare the delivered 0.5 and 0.6 engines on one real PoreMC JSON input.

Uses independent worker processes, one CPU thread, identical fixed physical
inputs and initial particles. It does not infer RASPA speed from PoreMC data.
All variants use direct Ewald with the input's full rectangular kmax set.
Grid variants retain that exact target through delayed acceptance.
"""
from __future__ import annotations
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
import time

ROOT = Path(__file__).resolve().parents[1]
NEW_OPTIONS = ('sampling_mode', 'host_grid_mesh', 'host_grid_cap',
               'host_grid_cache', 'grid_root_bias', 'ewald_half_spectrum',
               'ewald_k_cutoff', 'host_grid_steep')


def worker(args):
    source = Path(args.source).resolve()
    sys.path.insert(0, str(source))
    import numpy as np
    import poremc
    from poremc.cli import construct_case, clean_json
    path = Path(args.input).resolve()
    original = json.loads(path.read_text())
    cfg = copy.deepcopy(original)
    if cfg.get('surrogate'):
        raise ValueError('This benchmark compares native ClassicalPotential inputs; remove an external surrogate.')
    cfg.setdefault('potential', {}).update(backend='native', algorithm='cached', threads=1)
    if cfg['potential'].get('coulomb') != 'ewald':
        raise ValueError('This comparison requires potential.coulomb=ewald.')
    sample = cfg.setdefault('sampling', {})
    for key in NEW_OPTIONS:
        sample.pop(key, None)
    sample.update(driver='cpp', reciprocal_method='direct', reciprocal_backend='cpu', seed=args.seed)
    sample.pop('cuda_library', None)
    if args.mode != 'baseline05':
        sample.update(sampling_mode='exact', ewald_half_spectrum=True, ewald_k_cutoff=0.)
    if args.mode.startswith(('delayed', 'biased')):
        biased = args.mode.startswith('biased')
        mesh = int(args.mode.removeprefix('biased' if biased else 'delayed'))
        if biased and cfg.get('root_bias'):
            raise ValueError('A biased variant requires no explicit root_bias in the input; use delayed32/64 instead.')
        cache = Path(args.output).resolve().parent/'host_grid_cache'/f'{path.stem}-{mesh}.bin'
        sample.update(sampling_mode='delayed', host_grid_mesh=mesh,
                      host_grid_cache=str(cache), host_grid_cap=args.cap,
                      grid_root_bias=.95 if biased else 0.)
    start = time.perf_counter()
    sim, _, _, _, _ = construct_case(cfg, path.parent)
    construction = time.perf_counter()-start
    initial_energy = sim.total_energy()
    initial_n = int(sim.counts[0])
    if args.burnin:
        sim.run(args.burnin, sample_every=max(1,args.burnin), record_energy=False)
    before = sim.counters
    stages_before = getattr(sim, 'acceleration_profile', {})
    result = sim.run(args.steps, sample_every=args.sample_every, record_energy=False)
    after = sim.counters
    changes = {name:{key:after[name][key]-before[name][key] for key in ('attempted','accepted')}
               for name in sim.MOVE_NAMES}
    final_energy = sim.total_energy()
    reference = sim.potential.total_energy(sim.entries())
    count = np.asarray(result['counts']).reshape(-1)
    row = dict(mode=args.mode, seed=args.seed, version=poremc.__version__,
               module_path=poremc.__file__, input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
               native_sha256=hashlib.sha256(Path(sim._lib._name).read_bytes()).hexdigest(),
               steps=args.steps, burnin=args.burnin, construction_seconds=construction,
               production_seconds=result['production_seconds'],
               attempted_moves_per_second=result['attempted_moves_per_second'],
               accepted_moves_per_second=sum(v['accepted'] for v in changes.values())/result['production_seconds'],
               production_acceptance=changes, initial_molecules=initial_n,
               initial_energy_K=initial_energy, final_energy_K=final_energy,
               final_recompute_error_K=abs(final_energy-reference),
               final_molecules=int(sim.counts[0]), count_mean=float(count.mean()) if count.size else None,
               count_trace=count.tolist(), long_range=sim.long_range_profile,
               acceleration_before=stages_before,
               acceleration_after=getattr(sim,'acceleration_profile',{}),
               physical_target='input ClassicalPotential; direct rectangular finite Ewald; no radial truncation')
    sim.close()
    print(json.dumps(clean_json(row), allow_nan=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', help='Complete real-system PoreMC JSON input, including initial_particles if desired')
    p.add_argument('--steps', type=int, default=10000, help='Production attempted moves, not RASPA cycles')
    p.add_argument('--burnin', type=int, default=1000)
    p.add_argument('--sample-every', type=int, default=10)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=6106)
    p.add_argument('--modes', default='baseline05,native06,delayed32,biased32,delayed64')
    p.add_argument('--cap', type=float, default=1e5, help='Reference field cap in K; physical target unchanged')
    p.add_argument('--baseline', default=str(ROOT/'baseline_05'))
    p.add_argument('--output', default='hpc_comparison_06.json')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--source', default=str(ROOT), help=argparse.SUPPRESS)
    p.add_argument('--mode', default='baseline05', help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker:
        return worker(args)
    if min(args.steps,args.repeats,args.sample_every)<1 or args.burnin<0:
        p.error('Positive steps/repeats/sample-every and nonnegative burnin are required.')
    modes = args.modes.split(',')
    supported={'baseline05','native06','delayed16','delayed32','delayed64','delayed128',
               'biased16','biased32','biased64','biased128'}
    if 'baseline05' not in modes or not set(modes)<=supported:
        p.error('Include baseline05 and choose supported exact/delayed/biased variants.')
    baseline = Path(args.baseline).resolve()
    library = baseline/'poremc'/('libporemc.dylib' if sys.platform=='darwin' else 'libporemc.so')
    if not library.exists():
        subprocess.run([sys.executable,str(baseline/'native/build.py')],check=True,cwd=baseline)
    out = Path(args.output).resolve();out.parent.mkdir(parents=True,exist_ok=True)
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    data=dict(scope='Same-input CPU PoreMC comparison; RASPA must be timed separately',
              platform=platform.platform(), input=str(Path(args.input).resolve()), rows=[], summary=[])
    def save():
        tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False));tmp.replace(out)
    for repeat in range(args.repeats):
        for mode in modes if repeat%2==0 else list(reversed(modes)):
            source=baseline if mode=='baseline05' else ROOT
            command=[sys.executable,str(Path(__file__).resolve()),str(Path(args.input).resolve()),
                     '--worker','--source',str(source),'--mode',mode,'--steps',str(args.steps),
                     '--burnin',str(args.burnin),'--sample-every',str(args.sample_every),
                     '--seed',str(args.seed+repeat),'--cap',str(args.cap),'--output',str(out)]
            row=json.loads(subprocess.check_output(command,text=True,env=env))
            data['rows'].append(row);save()
            print(f"{mode}, seed={row['seed']}: {row['attempted_moves_per_second']:.2f} attempts/s; "
                  f"{row['accepted_moves_per_second']:.2f} accepted/s; N={row['count_mean']:.2f}",flush=True)
    refs={r['seed']:r for r in data['rows'] if r['mode']=='baseline05'}
    for mode in modes:
        rows=[r for r in data['rows'] if r['mode']==mode]
        ratios=[]
        for r in rows:
            ref=refs[r['seed']]
            if r['input_sha256']!=ref['input_sha256'] or abs(r['initial_energy_K']-ref['initial_energy_K'])>1e-5:
                raise RuntimeError('Physical input or initial energy differs between paired variants.')
            ratios.append(r['attempted_moves_per_second']/ref['attempted_moves_per_second'])
        data['summary'].append(dict(mode=mode,median_attempts_s=statistics.median(r['attempted_moves_per_second'] for r in rows),
            median_accepted_s=statistics.median(r['accepted_moves_per_second'] for r in rows),
            paired_speedups=ratios,median_speedup=statistics.median(ratios),
            max_recompute_error_K=max(r['final_recompute_error_K'] for r in rows)))
    save();print(json.dumps(data['summary'],indent=2))


if __name__=='__main__':
    main()
