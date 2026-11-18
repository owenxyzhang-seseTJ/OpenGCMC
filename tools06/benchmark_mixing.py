#!/usr/bin/env python3
"""Longer independent chains for occupancy comparison, not throughput timing.

Workers may run concurrently to reduce elapsed validation time.  Their
production timings are retained only as execution records and explicitly
excluded from reported performance comparisons or ESS/s calculations.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,default=30000)
    p.add_argument('--burnin',type=int,default=5000)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--seed',type=int,default=15306)
    p.add_argument('--modes',default='baseline05,delayed32,delayed64,biased32')
    p.add_argument('--steep',choices=['linear','boltzmann'],default='boltzmann')
    p.add_argument('--output',default=str(ROOT/'evidence06/occupancy_mixing.json'))
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    modes=args.modes.split(',');path=Path(args.output)
    data=dict(purpose='longer occupancy/mixing comparison of the same finite Ewald target',
              case='aromatic12',host_atoms=4416,guest_sites=12,kmax=13,
              modes=modes,steps=args.steps,burnin=args.burnin,repeats=args.repeats,
              surrogate_steep_interpolation=args.steep,concurrent_workers=args.workers,
              performance_measurement_valid=False,rows=[])
    if args.resume and path.exists():
        saved=json.loads(path.read_text())
        for key in ('steps','burnin','repeats','modes','surrogate_steep_interpolation'):
            if saved[key]!=data[key]:raise ValueError('resume mismatch '+key)
        data=saved
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    tasks=[]
    for mode in modes:
        for repeat in range(args.repeats):
            seed=args.seed+repeat
            if any(r['mode']==mode and r['seed']==seed for r in data['rows']):continue
            source=ROOT/'baseline_05' if mode=='baseline05' else ROOT
            cmd=[sys.executable,str(ROOT/'tools06/benchmark_hostgrid.py'),'--worker','--source',str(source),
                 '--case','aromatic12','--mode',mode,'--steps',str(args.steps),'--burnin',str(args.burnin),
                 '--seed',str(seed),'--sample-every','10','--steep',args.steep,
                 '--cache-dir',str(ROOT/'evidence06/grid_cache/boltz_final')]
            tasks.append((mode,seed,cmd))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future={pool.submit(subprocess.check_output,cmd,env=env,text=True):(mode,seed) for mode,seed,cmd in tasks}
        for completed in as_completed(future):
            row=json.loads(completed.result());data['rows'].append(row)
            temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False));temp.replace(path)
            print(row['mode'],row['seed'],'meanN=',round(row['count_mean'],5),
                  'range=',row['count_range'],flush=True)
    references={r['seed']:r for r in data['rows'] if r['mode']=='baseline05'}
    for row in data['rows']:
        if row['seed'] in references:
            ref=references[row['seed']]
            assert row['physics_sha256']==ref['physics_sha256']
            assert abs(row['initial_energy_K']-ref['initial_energy_K'])<1e-5
    print(path)


if __name__=='__main__':main()
