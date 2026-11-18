#!/usr/bin/env python3
"""Execute independent JSON windows in separate CPU processes.

No dependent Markov moves are parallelized. Each output retains its config and
checkpoint. Read-only content-addressed electrode caches may be shared.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,subprocess,sys,os
ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('configs',nargs='+');p.add_argument('--workers',type=int,default=2);p.add_argument('-o','--output',default='windows')
    a=p.parse_args()
    if a.workers<1:raise ValueError('workers must be positive')
    out=Path(a.output).resolve();out.mkdir(exist_ok=True,parents=True)
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONPATH':str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH','')}
    def run(item):
        i,config=item;d=out/f'{i:03d}_{Path(config).stem}';d.mkdir(exist_ok=True)
        with (d/'process.log').open('w') as log:
            subprocess.run([sys.executable,'-m','poremc.thermo_cli','run',str(Path(config).resolve()),'-o',str(d)],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        return str(d)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for path in pool.map(run,enumerate(a.configs)):print(path,flush=True)
if __name__=='__main__':main()
