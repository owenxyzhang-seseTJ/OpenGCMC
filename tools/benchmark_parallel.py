from pathlib import Path
import sys,json,time,os
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from poremc.topology import Molecule
from poremc.potentials import ClassicalPotential
from poremc.batching import run_native_replicas,TrialRequest,evaluate_requests
from poremc.cli import clean_json
rng=np.random.default_rng(71);cell=np.eye(3)*90;host=rng.random((8000,3))@cell
m=Molecule('chain',[2.9]*4,[65]*4,bond_lengths=[0,1.5,1.5,1.5])
pot=ClassicalPotential([m],cell,8,host,np.full(8000,3.),np.full(8000,30.),np.zeros(8000),algorithm='cached',threads=1)
common=dict(species=[m],potential=pot,cell=cell,temperature=300,activities=[3/np.linalg.det(cell)],seeds=[37,43,59,71],run_options={'steps':25000,'burnin':1000,'sample_every':50,'record_energy':False},simulation_options={'trials':8})
records=[]
for repeat in range(3):
    single=run_native_replicas(**common,max_workers=1);multi=run_native_replicas(**common,max_workers=4)
    same=all(a['counts']==b['counts'] for a,b in zip(single['replicas'],multi['replicas']))
    records.append({'sequential':single,'parallel4':multi,'same_traces':same,'speedup':single['wall_seconds']/multi['wall_seconds']})
    print('parallel',repeat,records[-1]['speedup'],same,flush=True)
    (root/'benchmarks/results/parallel_benchmark.json').write_text(json.dumps(clean_json(records),indent=2))
# Energy request coalescing. Same host, exact same coordinates in both paths.
requests=[TrialRequest(pot.host_evaluator,rng.random((1,4,3))@cell,m.sigma,m.epsilon,m.charge) for _ in range(64)]
seq=lambda:[r.environment.energy(r.coordinates,r.sigma,r.epsilon,r.charge) for r in requests]
batch=lambda:evaluate_requests(requests)
for x,y in zip(seq(),batch()):np.testing.assert_allclose(x,y,rtol=1e-12,atol=1e-7)
samples={}
for name,fn in [('separate',seq),('coalesced',batch)]:
    fn();values=[]
    for i in range(9):
        start=time.perf_counter()
        for j in range(20):fn()
        values.append((time.perf_counter()-start)*1000/20)
    samples[name]={'samples_ms':values,'median_ms':float(np.median(values))}
samples['speedup']=samples['separate']['median_ms']/samples['coalesced']['median_ms'];samples['requests']=64
(root/'benchmarks/results/batching_benchmark.json').write_text(json.dumps(samples,indent=2));print('batching',samples['speedup'])
