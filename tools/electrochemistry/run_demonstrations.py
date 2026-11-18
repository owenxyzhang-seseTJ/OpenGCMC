"""Execute released examples and preserve raw outputs. No external services needed."""
from pathlib import Path
import subprocess, sys, os
ROOT=Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT=ROOT/'benchmarks/runs/electrochemistry'
OUT.mkdir(parents=True, exist_ok=True)
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
for task, name in [('run','dummy_proton_titration'),('run','pcet_four_state'),
                   ('kinetics','two_contact_current'),('kinetics','voltammetry'),
                   ('transport','diffusion_voltammetry'),('transport','diffusion_chronoamperometry')]:
    out=OUT/name
    with open(OUT/f'{name}.log','w') as log:
        subprocess.run([sys.executable,'-m','poremc.reactive_cli',task,f'examples/electrochemistry/{name}.json','-o',str(out)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('completed',name,flush=True)
