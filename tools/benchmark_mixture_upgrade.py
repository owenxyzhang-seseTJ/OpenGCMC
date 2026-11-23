"""Compare statistically fixed-budget OpenGCMC runs on the same binary input."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np

from poremc.cli import construct_case
from poremc.scheduling import _point_summary, _publish


def run_benchmark(path, pressures, repeats, steps, burnin, cache):
    config = json.loads(path.read_text())
    events = []
    grouped = []
    modes = ('rebuild', 'incremental', 'delayed')
    for pressure in pressures:
        traces = {mode: [] for mode in modes}
        for replica in range(repeats):
            seed = int(np.random.SeedSequence([812519,replica,int(pressure)]).generate_state(1)[0])
            order = modes[replica%3:]+modes[:replica%3]
            for mode in order:
                case = copy.deepcopy(config)
                case['reservoir']['pressure_Pa'] = pressure
                sampling = case['sampling']
                sampling.update(steps=steps,burnin=burnin,seed=seed,record_energy=False)
                sampling['guest_update'] = 'rebuild' if mode=='rebuild' else 'incremental'
                if mode=='delayed':
                    sampling['host_field_cache'] = str(cache.resolve())
                else:
                    sampling.update(sampling_mode='exact',grid_root_bias=0.)
                start = time.perf_counter()
                sim,options,host,_,_ = construct_case(case,path.parent)
                setup = time.perf_counter()-start
                with sim:
                    result = sim.run(**options)
                    elapsed = time.perf_counter()-start
                    result.update(framework_mass_amu=host.mass_amu,species_order=[m.name for m in sim.species],
                                  setup_seconds=setup)
                    traces[mode].append([result])
                    prof = sim.profile
                events.append({
                    'pressure_Pa':pressure,'seed':seed,'mode':mode,'setup_s':setup,'wall_s':elapsed,
                    'moves_per_s':result['attempted_moves_per_second'],
                    'count_means':[v['mean'] for v in result['species_statistics'].values()],
                    'guest_environment_builds':prof['guest_environment_builds'],
                    'guest_local_updates':prof['guest_local_updates'],
                    'raw_field_builds':prof['acceleration'].get('host_grid',{}).get('raw_field_builds',0),
                    'raw_field_loads':prof['acceleration'].get('host_grid',{}).get('raw_field_loads',0)
                })
            print(f'{pressure:g} Pa replica {replica+1}/{repeats}',flush=True)
        reservoir = {**config['reservoir'],'pressure_Pa':pressure}
        rows = [r for r in events if r['pressure_Pa']==pressure]
        baseline = {r['seed']:r for r in rows if r['mode']=='rebuild'}
        test = {mode:_point_summary(reservoir,trace,dict(count_mcse=.20,log_selectivity_mcse=.10))
                for mode,trace in traces.items()}
        for mode in modes:
            measured = [r for r in rows if r['mode']==mode]
            test[mode]['median_moves_per_s'] = statistics.median(r['moves_per_s'] for r in measured)
            test[mode]['median_paired_wall_speedup'] = statistics.median(
                baseline[r['seed']]['wall_s']/r['wall_s'] for r in measured)
            test[mode]['median_paired_production_speedup'] = statistics.median(
                r['moves_per_s']/baseline[r['seed']]['moves_per_s'] for r in measured)
        grouped.append({'pressure_Pa':pressure,'modes':test})
    return {'platform':platform.platform(),'steps':steps,'burnin':burnin,'replicas':repeats,
            'observation_interval':config['sampling']['sample_every'],'timings':events,'points':grouped}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path)
    p.add_argument('--pressure-pa',type=float,nargs='+',default=[20000,100000,500000])
    p.add_argument('--replicas',type=int,default=8)
    p.add_argument('--steps',type=int,default=200000)
    p.add_argument('--burnin',type=int,default=30000)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    _publish(a.output,run_benchmark(a.input,a.pressure_pa,a.replicas,a.steps,a.burnin,a.cache))


if __name__=='__main__':
    main()
