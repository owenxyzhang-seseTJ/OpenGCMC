"""Isolated-process benchmark of a selected algorithm in the current source tree."""
import argparse,json,sys,time,platform
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--kind',choices=['kernel','mc'],required=True)
p.add_argument('--algorithm',default='cell');p.add_argument('--fixture');p.add_argument('--case',default='flex_host');p.add_argument('--seed',type=int,default=42);p.add_argument('--steps',type=int,default=12000)
a=p.parse_args();root=Path(__file__).resolve().parents[1];sys.path.insert(0,a.source)
import numpy as np
from poremc.kernels import NativeEnvironment

def raw_fixture(path):
    import importlib.util
    spec=importlib.util.spec_from_file_location('fixture_reader',root/'tools/fixtures.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod.read_fixture(path),mod
meta={'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),'cpu':next((v.split(':',1)[1].strip() for v in Path('/proc/cpuinfo').read_text().splitlines() if v.startswith('model name')),''),
      'source':str(Path(a.source).name),'algorithm':a.algorithm,'kind':a.kind}
if a.kind=='kernel':
    d,mod=raw_fixture(a.fixture);t=time.perf_counter();env=NativeEnvironment(*mod.env_args(d),**mod.env_kwargs(d),algorithm=a.algorithm,threads=1);setup=time.perf_counter()-t
    def run():return env.energy(d['xyz'],d['ts'],d['te'],d['tq'])
    first=time.perf_counter();v=run();first=time.perf_counter()-first
    expected=np.fromfile(Path(a.fixture).with_suffix('.gpu'),dtype='<f8');np.testing.assert_allclose(v,expected,rtol=1e-9,atol=1e-6)
    for _ in range(3):run()
    samples=[]
    for _ in range(9):
        t=time.perf_counter()
        for _ in range(30):run()
        samples.append((time.perf_counter()-t)*1000/30)
    meta.update(fixture=Path(a.fixture).stem,setup_ms=setup*1000,first_call_ms=first*1000,samples_ms=samples,median_ms=float(np.median(samples)),
                B=len(d['xyz']),A=len(d['ts']),M=len(d['host']),mode=d['coulomb'])
    if hasattr(env,'profile'):meta['cache_profile']=env.profile
else:
    from poremc.topology import Molecule
    from poremc.potentials import ClassicalPotential
    from poremc.fast import FastGCMC
    rng=np.random.default_rng(20261117)
    if a.case=='dense_guest':
        m=Molecule('LJ',[3.4],[120]);cell=np.eye(3)*30;host=np.empty((0,3));hs=he=hq=np.empty(0);T=150;cutoff=8
        move={'translate':1.};z=1e-4;trial=1;displacement=4.
    else:
        rigid=a.case=='rigid_host';M=32000 if rigid else 8000;L=160 if rigid else 90
        m=Molecule('four-site',[2.9]*4,[65]*4,bond_lengths=[0,1.5,1.5,1.5],
             rigid_positions=np.array([[0,0,0],[1.5,0,0],[2,1.4,0],[3.4,1.6,.6]]) if rigid else None,
             torsions=[] if rigid else [{'indices':[0,1,2,3],'k':70,'periodicity':3}])
        cell=np.eye(3)*L;host=rng.random((M,3))@cell;hs=np.take([2.7,3.,3.3],np.arange(M)%3);he=np.take([20.,35.,50.],np.arange(M)%3);hq=np.zeros(M)
        T=300;cutoff=8;move={'insert':.25,'delete':.25,'translate':.2,'rotate':.1,'regrow':.2};z=3/np.linalg.det(cell);trial=8;displacement=1.
    t=time.perf_counter();pot=ClassicalPotential([m],cell,cutoff,host,hs,he,hq,algorithm=a.algorithm,threads=1)
    sim=FastGCMC([m],pot,cell,T,[z],seed=a.seed,trials=trial,move_weights=move,max_translation=displacement)
    if a.case=='dense_guest':
        for point in np.stack(np.meshgrid(*[np.arange(5)*6+3]*3,indexing='ij'),axis=-1).reshape(-1,3):sim.add_particle(0,point[None,:])
    setup=time.perf_counter()-t;t=time.perf_counter();res=sim.run(a.steps,burnin=1500,sample_every=20,record_energy=False)
    wall=time.perf_counter()-t
    meta.update(case=a.case,seed=a.seed,setup_seconds=setup,api_wall_seconds=wall,**res)
    if hasattr(sim,'profile'):meta['cache_profile']=sim.profile
    meta['final_positions']=np.asarray([p.positions for p in sim.particles],dtype='<f8').tolist()
    meta['count_trace']=res['counts']
    sim.close()
from poremc.cli import clean_json
print(json.dumps(clean_json(meta),allow_nan=False))
