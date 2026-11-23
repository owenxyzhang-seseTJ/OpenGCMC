"""Measure mixing across an activity ladder in a fixed binary adsorption model."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
from pathlib import Path
import time

import numpy as np

from poremc.cli import construct_case
from poremc.ensembles import GrandCanonicalReplicaExchange
from poremc.fast import FastGCMC
from poremc.scheduling import _point_summary, _publish


def run_ladder(path,pressures,seed,cache,warmup,rounds,steps):
    config=json.loads(path.read_text())
    config['reservoir']['pressure_Pa']=pressures[0]
    config['sampling']['host_field_cache']=str(cache)
    config['sampling']['seed']=seed
    start=time.perf_counter()
    first,_,host,_,_=construct_case(config,path.parent)
    records=[{'counts':[],'production_seconds':0.,'setup_seconds':0.,'framework_mass_amu':host.mass_amu,
              'species_order':[m.name for m in first.species]} for _ in pressures]
    with ExitStack() as stack:
        walkers=[stack.enter_context(first)]
        for i,pressure in enumerate(pressures[1:],1):
            walkers.append(stack.enter_context(FastGCMC(
                first.species,first.potential,first.cell,first.temperature,
                first.activities*(pressure/pressures[0]),seed=seed+i,trials=config['sampling']['trials'],
                **first._acceleration_options())))
        rex=GrandCanonicalReplicaExchange(walkers,seed=seed+101)
        for _ in range(warmup):
            rex.advance(steps,sample_every=config['sampling']['sample_every'])
        visited=[0]*len(walkers)
        round_trips=0
        for _ in range(rounds):
            batch=rex.advance(steps,sample_every=config['sampling']['sample_every'])
            for record,result in zip(records,batch):
                record['counts'].extend(result['counts'])
                record['production_seconds']+=result['production_seconds']
            low,high=rex.occupants[0],rex.occupants[-1]
            if visited[high]==1:
                visited[high]=2
            if visited[low]==2:
                round_trips+=1
            visited[low]=1
    return {'seed':seed,'records':records,'attempts':rex.attempts.tolist(),'accepted':rex.accepts.tolist(),
            'round_trips':round_trips,'wall_s':time.perf_counter()-start}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path)
    p.add_argument('--pressure-pa',type=float,nargs='+',
                   default=[2500,5000,10000,20000,40000,65000,100000,160000,250000,360000,500000])
    p.add_argument('--ladders',type=int,default=8)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--rounds',type=int,default=400)
    p.add_argument('--burnin-rounds',type=int,default=60)
    p.add_argument('--steps',type=int,default=1200)
    p.add_argument('--seed',type=int,default=191530)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    bulk=json.loads(a.input.read_text())['reservoir']
    start=time.perf_counter()

    def one(i):
        seed=int(np.random.SeedSequence([a.seed,i]).generate_state(1)[0])
        result=run_ladder(a.input,a.pressure_pa,seed,a.cache.resolve(),a.burnin_rounds,a.rounds,a.steps)
        print(f'Ladder {i+1}/{a.ladders}, round trips {result["round_trips"]}',flush=True)
        return result

    with ThreadPoolExecutor(a.workers) as pool:
        results=list(pool.map(one,range(a.ladders)))
    wall=time.perf_counter()-start
    points=[_point_summary({**bulk,'pressure_Pa':pressure},[[r['records'][j]] for r in results],
                           dict(count_mcse=.20,log_selectivity_mcse=.10))
            for j,pressure in enumerate(a.pressure_pa)]
    _publish(a.output,{'points':points,'workers':a.workers,'rounds':a.rounds,'burnin_rounds':a.burnin_rounds,
                       'steps_per_round':a.steps,'wall_s':wall,'root_seed':a.seed,
                       'ladders':[{k:v for k,v in r.items() if k!='records'} for r in results]})


if __name__=='__main__':
    main()
