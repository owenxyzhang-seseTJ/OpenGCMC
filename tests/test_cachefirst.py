import json,math
import numpy as np
import pytest
from tools.fixtures import read_fixture,env_args,env_kwargs
from poremc.kernels import NativeEnvironment,ReferenceEnvironment
from poremc.topology import Molecule
from poremc.potentials import ClassicalPotential
from poremc.engine import GCMC,ZeroPotential
from poremc.fast import FastGCMC
from poremc.cbmc import CBMC
from poremc.proposals import VoxelProposal,build_host_proposal
from poremc.cache import ArrayCache,content_key
from poremc.accelerators import GridHostSurrogate
from poremc.batching import TrialRequest,evaluate_requests

@pytest.mark.parametrize('index',range(16))
@pytest.mark.parametrize('algorithm',['direct','cell','cached'])
def test_saved_gpu_values(evidence,index,algorithm):
    f=evidence/'fixtures'/f'{index:02d}.bin';d=read_fixture(f);expected=np.fromfile(f.with_suffix('.gpu'),dtype='<f8')
    with NativeEnvironment(*env_args(d),**env_kwargs(d),algorithm=algorithm) as e:
        value=e.energy(d['xyz'],d['ts'],d['te'],d['tq'])
    np.testing.assert_allclose(value,expected,rtol=2e-11 if index<12 else 1e-9,atol=1e-7 if index<12 else 1e-6)

@pytest.mark.parametrize('algorithm',['cell','cached'])
@pytest.mark.parametrize('mode',['none','dsf','ewald-real'])
def test_excluding_original_indices(algorithm,mode):
    rng=np.random.default_rng(72);h=np.array([[48,0,0],[12,48,0],[-7,4,48.]])
    x=rng.random((200,3))@h;s=np.full(200,3.);ep=np.full(200,70.);q=rng.normal(0,.1,200)
    trial=rng.random((31,2,3))@h;params=([2.7,3.1],[40,90],[.1,-.2]);keep=np.r_[0:20,37:200]
    with NativeEnvironment(x,s,ep,q,h,8,coulomb=mode,algorithm=algorithm) as env,NativeEnvironment(x[keep],s[keep],ep[keep],q[keep],h,8,coulomb=mode,algorithm='direct') as ref:
        np.testing.assert_allclose(env.energy_excluding(trial,*params,20,37),ref.energy(trial,*params),rtol=2e-11,atol=1e-7)

@pytest.mark.parametrize('skew',[False,True])
def test_csr_plan_contains_every_possible_neighbour(skew):
    rng=np.random.default_rng(2);cell=np.eye(3)*60
    if skew:cell[1,0]=20;cell[2,0]=-12;cell[2,1]=9
    x=rng.random((500,3))@cell;inv=np.linalg.inv(cell)
    with NativeEnvironment(x,np.ones(500),np.ones(500),np.zeros(500),cell,8,algorithm='cached') as e:
        plan=e.export_spatial_plan();assert plan['indexed'];order=plan['order']
        for pos in rng.random((50,3))@cell:
            frac=np.mod(pos@inv,1);key=np.ravel_multi_index(np.minimum((frac*plan['bins']).astype(int),plan['bins']-1),tuple(plan['bins']))
            ids=order[plan['indices'][plan['offsets'][key]:plan['offsets'][key+1]]]
            delta=(pos-x)@inv;delta-=np.rint(delta);d2=np.sum((delta@cell)**2,axis=1)
            assert set(np.flatnonzero(d2<64)).issubset(set(ids.tolist()))

def test_pair_tables_reuse(evidence):
    d=read_fixture(evidence/'fixtures/13.bin')
    with NativeEnvironment(*env_args(d),**env_kwargs(d),algorithm='cached') as e:
        e.energy(d['xyz'],d['ts'],d['te'],d['tq']);before=e.profile
        e.energy(d['xyz'],d['ts'],d['te'],d['tq']);after=e.profile
        assert before['pair_cache_misses']==1 and after['pair_cache_misses']==1
        assert after['pair_cache_hits']>before['pair_cache_hits'] and after['spatial_plan_bytes']>0

def bias():
    p=np.ones((3,2,2));p[0]*=5
    return VoxelProposal(np.eye(3)*30,p)

def test_alias_histogram_and_periodicity():
    b=bias();x=b.sample(np.random.default_rng(42),150000)
    frac=x/30;idx=(frac*b.shape).astype(int);k=np.ravel_multi_index(idx.T,tuple(b.shape))
    empirical=np.bincount(k,minlength=b.n)/len(k)
    np.testing.assert_allclose(empirical,b.probabilities.ravel(),atol=.003)
    np.testing.assert_allclose(b.log_relative_density(x[:20]),b.log_relative_density(x[:20]+30*np.array([2,-4,1])))
    assert np.isclose(b.probabilities.sum(),1)

@pytest.mark.parametrize('fraction',[0,-1,1.1])
def test_proposal_rejects_zero_support(fraction):
    with pytest.raises(ValueError):VoxelProposal.from_energies(np.eye(3)*30,np.zeros((2,2,2)),300,fraction)

def test_proposal_checkpoint_roundtrip():
    b=bias();c=VoxelProposal.from_dict(json.loads(json.dumps(b.to_dict())))
    np.testing.assert_array_equal(b.probabilities,c.probabilities)
    np.testing.assert_array_equal(b.sample(np.random.default_rng(3),100),c.sample(np.random.default_rng(3),100))

@pytest.mark.parametrize('flex',[False,True])
def test_cbmc_k1_density_correction(flex):
    m=Molecule('test',[0,0],[0,0],bond_lengths=[0,1.5]) if flex else Molecule('test',[0],[0])
    b=bias();sampler=CBMC(300,b.cell,ZeroPotential(),trials=1,root_proposal=b)
    for seed in range(10):
        result=sampler.grow(m,[],np.random.default_rng(seed))
        expected=-float(b.log_relative_density(result.positions[0]))
        assert np.isclose(result.log_weight,expected)
        old=sampler.retrace(m,result.positions,[],np.random.default_rng(seed+20));assert np.isclose(old.log_weight,expected)

@pytest.mark.parametrize('trials',[1,4])
@pytest.mark.parametrize('algorithm',['cell','cached'])
def test_native_ideal_gas_biased_root(trials,algorithm):
    m=Molecule('ideal',[0],[0]);cell=np.eye(3)*30;pot=ClassicalPotential([m],cell,8,algorithm=algorithm)
    with FastGCMC([m],pot,cell,300,[3/27000],trials=trials,seed=242+trials,root_proposal=bias(),move_weights={'insert':.5,'delete':.5}) as sim:
        res=sim.run(80000,burnin=5000,sample_every=5)
        n=np.array(res['counts'])[:,0]
        assert abs(n.mean()-3)<.12
        assert abs(n.var()-3)<.35
        profile=sim.profile
        assert profile['guest_environment_reuses']>0 and profile['guest_environment_builds']<sim.step_count

@pytest.mark.parametrize('driver',['python','cpp'])
def test_exact_restart_with_bias(tmp_path,driver):
    m=Molecule('ideal',[0],[0]);cell=np.eye(3)*30;pot=ClassicalPotential([m],cell,8,algorithm='cached')
    klass=GCMC if driver=='python' else FastGCMC
    sim=klass([m],pot,cell,300,[3/27000],trials=3,seed=35,root_proposal=bias(),move_weights={'insert':.5,'delete':.5})
    sim.run(300,sample_every=5);f=tmp_path/'state.json';sim.save_checkpoint(f)
    expected=sim.run(400,sample_every=5)
    restored=klass.from_checkpoint(f,pot);actual=restored.run(400,sample_every=5)
    assert expected['counts']==actual['counts']
    for a,b in zip(sim.particles,restored.particles):np.testing.assert_array_equal(a.positions,b.positions)
    if driver=='cpp':sim.close();restored.close()

@pytest.mark.parametrize('mode',['none','dsf'])
def test_native_moving_state_matches_full_energy(mode):
    m=Molecule('chain',[2.5]*4,[30]*4,charge=[.1,-.1,.1,-.1],bond_lengths=[0,1.5,1.5,1.5],
       torsions=[{'indices':[0,1,2,3],'k':70.,'periodicity':3}])
    cell=np.eye(3)*32;rng=np.random.default_rng(16);host=rng.random((120,3))@cell
    pot=ClassicalPotential([m],cell,7,host,np.ones(120)*2.5,np.ones(120)*10,np.zeros(120),algorithm='cached',coulomb=mode)
    with FastGCMC([m],pot,cell,300,[2e-4],trials=4,seed=556) as sim:
        for i in range(8):
            sim.run(200,sample_every=50)
            entries=[(p.species,p.positions) for p in sim.particles]
            np.testing.assert_allclose(sim.total_energy(),pot.total_energy(entries),rtol=2e-10,atol=1e-6)

@pytest.mark.parametrize('ewald',[False,True])
def test_prepared_environment_cache_is_value_keyed(ewald):
    m=Molecule('q',[0],[0],charge=[0]);cell=np.eye(3)*30
    pot=ClassicalPotential([m],cell,8,coulomb='ewald' if ewald else 'none',kmax=2)
    pos=np.array([[2.,3.,4.]])
    first=pot.prepare_environment([(0,pos)]);second=pot.prepare_environment([(0,pos.copy())]);assert first is second
    pos[0,0]+=1;third=pot.prepare_environment([(0,pos)]);assert third is not first
    np.testing.assert_array_equal(first.positions,[[2.,3.,4.]])
    for i in range(12):pot.prepare_environment([(0,np.array([[i,1.,2.]]))])
    assert len(pot._prepared_cache)<=8

def test_persistent_grids_and_root_cache(tmp_path):
    m=Molecule('probe',[2],[50]);cell=np.eye(3)*30
    pot=ClassicalPotential([m],cell,8,[[15,15,15]],[3],[80],[0],algorithm='cached')
    a=GridHostSurrogate(pot,shape=(5,5,5),cache_dir=tmp_path);b=GridHostSurrogate(pot,shape=(5,5,5),cache_dir=tmp_path)
    assert not a.metadata['cache_hit'] and b.metadata['cache_hit']
    for k in a.fields:np.testing.assert_array_equal(a.fields[k].values,b.fields[k].values)
    x=build_host_proposal(pot,m,300,shape=(5,5,5),cache_dir=tmp_path)
    y=build_host_proposal(pot,m,250,shape=(5,5,5),cache_dir=tmp_path)
    assert not x.metadata['cache_hit'] and y.metadata['cache_hit']
    assert not np.allclose(x.probabilities,y.probabilities)

def test_cache_corruption_and_fingerprint(tmp_path):
    c=ArrayCache(tmp_path);a={'x':np.arange(4.)};k=content_key({'v':1},a);c.put(k,a)
    np.testing.assert_array_equal(c.get(k)['x'],a['x'])
    assert k!=content_key({'v':2},a)
    (tmp_path/(k+'.npz')).write_bytes(b'bad');assert c.get(k) is None

class CountingPotential:
    def __init__(self,value):self.value=value;self.single_calls=0
    def molecule_energies(self,x,m,env):return np.full(len(x),self.value)
    def molecule_energy(self,x,m,env):self.single_calls+=1;return float(self.value)

def test_delayed_acceptance_reuses_selected_surrogate_energy():
    m=Molecule('ideal',[0],[0]);sur=CountingPotential(0);target=CountingPotential(0)
    sim=GCMC([m],sur,np.eye(3)*30,300,[100/27000],target_potential=target,trials=4,seed=19,move_weights={'insert':.5,'delete':.5})
    assert sim.step(move='insert')
    assert sur.single_calls==0 and target.single_calls==1

def test_request_batching(evidence):
    d=read_fixture(evidence/'fixtures/13.bin')
    with NativeEnvironment(*env_args(d),**env_kwargs(d),algorithm='cached') as e:
        requests=[TrialRequest(e,x,d['ts'],d['te'],d['tq']) for x in np.array_split(d['xyz'],8)]
        expected=[e.energy(r.coordinates,r.sigma,r.epsilon,r.charge) for r in requests]
        before=e.profile['energy_calls'];actual=evaluate_requests(requests);assert e.profile['energy_calls']-before==1
        for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)


def test_parallel_native_replicas_match_sequential():
    from poremc.batching import run_native_replicas
    m=Molecule('ideal',[0],[0]);cell=np.eye(3)*30;pot=ClassicalPotential([m],cell,8,algorithm='cached')
    common=dict(species=[m],potential=pot,cell=cell,temperature=300,activities=[3/27000],seeds=[11,29,73],run_options={'steps':2000,'sample_every':10,'record_energy':False})
    a=run_native_replicas(**common,max_workers=1);b=run_native_replicas(**common,max_workers=3)
    for x,y in zip(a['replicas'],b['replicas']):assert x['counts']==y['counts']
    # Closing borrowed per-chain bindings must not destroy the shared host.
    assert not pot.host_evaluator._closed


def test_gpu_plan_export_roundtrip(evidence,tmp_path):
    import struct
    from tools.run_gpu_benchmark import prepare
    d=read_fixture(evidence/'fixtures/13.bin');p=tmp_path/'plan.bin';r=prepare(d,p)
    with open(p,'rb') as f:
        tag,*header=struct.unpack('<8s6i2Q',f.read(48))
        assert tag==b'PCACHE02' and header[3]==1 and header[5]==1
        fs=np.fromfile(f,dtype='<f8',count=3*len(d['host'])).reshape(3,-1)
        assert np.isfinite(fs).all() and (fs>=0).all() and (fs<1).all()
    assert r['profile']['expanded_plan']==1
