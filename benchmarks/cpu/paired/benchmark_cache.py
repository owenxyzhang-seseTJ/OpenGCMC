"""Pair independent cold starts with one cold build reused over an isotherm."""

import argparse
from pathlib import Path
import statistics
import time

from benchmark_precision import read_model, setup, write_json


def point(model_path, directory, seed):
    start=time.perf_counter()
    model=read_model(model_path)
    with setup(model,model_path,seed,'delayed',directory,32,'incremental') as sim:
        setup_s=time.perf_counter()-start
        draws=sim.run(12000,burnin=4000,sample_every=200,record_energy=False)
        wall_s=time.perf_counter()-start
        grid=sim.acceleration_profile['host_grid']
    return {'wall_s':wall_s,'setup_s':setup_s,'field_builds':grid['raw_field_builds'],
            'field_loads':grid['raw_field_loads'],'counts':draws['counts']}


def run(input_dir,cache):
    models=[input_dir/f'xylene_{name}.json' for name in ('low','mid','high')]
    rows=[]
    for i,seed in enumerate((533701,533723,533753)):
        independent=lambda:[point(p,cache/str(seed)/p.stem,seed) for p in models]
        combined=lambda:[point(p,cache/str(seed)/'batch',seed) for p in models]
        if i%2:
            batch=combined()
            cold=independent()
        else:
            cold=independent()
            batch=combined()
        for a,b in zip(cold,batch):
            if a.pop('counts')!=b.pop('counts'):
                raise ValueError('cache state changed the paired trajectory')
        rows.append({'seed':seed,'independent':cold,'batch':batch,
                     'cold_total_s':sum(row['wall_s'] for row in cold),
                     'batch_total_s':sum(row['wall_s'] for row in batch)})
    return {'models':[p.stem for p in models],'production_attempts_per_point':12000,'warmup':4000,
            'repeats':rows,'batch_speedup':statistics.median(row['cold_total_s']/row['batch_total_s'] for row in rows)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs',type=Path)
    p.add_argument('fresh_cache_directory',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    a.fresh_cache_directory.mkdir(parents=True,exist_ok=False)
    write_json(a.output,run(a.inputs,a.fresh_cache_directory))
