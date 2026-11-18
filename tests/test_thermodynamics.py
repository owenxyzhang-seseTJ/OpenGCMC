import numpy as np
import pytest
from poremc.response import *
from poremc.free_energy import *
from poremc.electrolyte import *
from poremc.electrode import *


def test_moments_merge():
    x=np.random.default_rng(2).normal(size=(1203,4))+1e7
    a=MomentAccumulator(4);b=MomentAccumulator(4)
    a.update(x[:600]);b.update(x[600:]);a.merge(b)
    np.testing.assert_allclose(a.mean,x.mean(axis=0),atol=1e-8)
    np.testing.assert_allclose(a.covariance,np.cov(x,rowvar=False,ddof=0),atol=1e-9)


def test_poisson_response():
    x=np.random.default_rng(3).poisson([3,7],size=(150000,2));r=weighted_response(x)
    np.testing.assert_allclose(r['response'],np.diag([3,7]),atol=.08)
    assert np.min(np.linalg.eigvalsh(r['response']))>0


def test_response_derivative():
    x=np.array([[i,j] for i in range(8) for j in range(9)])
    lw=-.3*x[:,0]**2-.2*x[:,1]**2+.1*x[:,0]*x[:,1]
    c=weighted_response(x,lw)['response'];h=1e-5
    for j in range(2):
        d=np.eye(2)[j]*h
        finite=(chemical_reweight(x,d,lw)['mean']-chemical_reweight(x,-d,lw)['mean'])/(2*h)
        np.testing.assert_allclose(finite,c[:,j],rtol=1e-8,atol=1e-9)


def test_salt_response_basis_and_gauge():
    S=np.array([[1,0],[0,1],[1,2]]);q=[1,2,-1]
    extent=np.random.default_rng(3).poisson([2,5],size=(5000,2));n=extent@S.T
    b=SaltCoordinates(S,q);np.testing.assert_allclose(b.extents(n),extent,atol=1e-12)
    r=b.response(n);np.testing.assert_allclose(r['salt_response'],np.cov(extent,rowvar=False,ddof=0),atol=1e-12)
    assert r['charge_null_residual']<1e-10
    np.testing.assert_allclose(chemical_reweight(n,np.array(q)*10)['mean'],n.mean(axis=0),atol=1e-11)
    with pytest.raises(ValueError):SaltCoordinates([[1,2],[1,2]],[1,-1])


def test_correlated_covariance_bootstrap():
    rng=np.random.default_rng(5);x=np.zeros((6000,2))
    for i in range(1,len(x)):x[i]=.9*x[i-1]+rng.normal(size=2)
    r=response_summary(x,300,block_length=200,bootstrap=20)
    assert r['response_ci95'].shape==(2,2,2);assert np.nanmax(r['tau_observables_and_products'])>5
    np.testing.assert_allclose(r['dmean_dmu_kelvin'],r['response']/300)


def test_landscape_unvisited_and_density():
    r=landscape([.2,.3,1.2],[[0,1,2,3]],temperature=300)
    assert np.isinf(r['free_energy_kelvin'][-1]) and not r['sampled'][-1]
    np.testing.assert_allclose(r['probability_mass'].sum(),1)
    r=landscape([.2,1.2],[[0,1,3]],density=True)
    np.testing.assert_allclose(r['free_energy_kelvin'],[0,np.log(2)])
    with pytest.raises(ValueError):landscape([5],[[0,1]])


def test_mbar_exact_weighted_enumeration():
    # Each state has 100 exact samples with integer histogram ratios.
    x=np.r_[np.zeros(80),np.ones(20),np.zeros(20),np.ones(80)]
    u=np.array([x*np.log(4),(1-x)*np.log(4)])
    fit=MBAR(u,[100,100]);r=fit.expectation(x,np.zeros(200))
    assert abs(r['mean']-.5)<1e-12
    np.testing.assert_allclose(fit.f,[0,0],atol=1e-12)
    assert fit.spectral_gap>.1
    np.testing.assert_allclose(fit.overlap.sum(axis=1),1,atol=1e-9)
    bootstrap=block_bootstrap_mbar(u,[100,100],np.zeros(200),x,[5,5],replicates=4)
    assert bootstrap['replicates'].shape==(4,)


def test_mbar_disconnected_rejected():
    with pytest.raises((ValueError,RuntimeError)):
        MBAR([[0,0,np.inf,np.inf],[np.inf,np.inf,0,0]],[2,2])

@pytest.mark.parametrize('nu',[(1,1),(1,2),(2,3)])
@pytest.mark.parametrize('p',[.2,.5,.8])
def test_exchange_detailed_balance(nu,p):
    n=np.array([5,8]);nu=np.array(nu);la=np.log([2e-4,1e-4]);lq=np.full(nu.sum(),-np.log(6000))
    a=log_exchange_ratio(n,nu,la,lq,157.3,300,True,p,2.1)
    b=log_exchange_ratio(n+nu,nu,la,lq,-157.3,300,False,p,-2.1)
    assert abs(a+b)<1e-12

@pytest.mark.parametrize('nu,q,mean_z',[( (1,1),(1,-1),(4.,4.) ),((1,2),(2,-1),(2.,4.))])
def test_ideal_salt_gcmc(nu,q,mean_z):
    species=[Ion('a',q[0],0,0),Ion('b',q[1],0,0)];cell=np.eye(3)*24;V=np.linalg.det(cell)
    h=IonicHamiltonian(species,cell,8,coulomb='none');sim=ElectrolyteGCMC(h,300,np.log(np.array(mean_z)/V),[ExchangeChannel('salt',nu,insert_probability=.7)],exchange_probability=1,seed=24)
    out=sim.run(32000,burnin=1000,sample_every=2);expected=ideal_salt_distribution(nu,np.log(mean_z))
    np.testing.assert_allclose(out['counts'].mean(axis=0),expected['mean'],atol=.12)
    assert np.all(out['counts']@q==0)
    np.testing.assert_allclose(np.cov(out['counts'],rowvar=False,ddof=0),expected['covariance'],atol=.25)
    h.close()


def make_electrode(slab=False,cache=None):
    cell=np.diag([24.,24.,48.]);xyz=np.array([[x,y,z] for z in [12.,36.] for x in [4,12,20] for y in [4,12,20]])
    return GaussianElectrodes(xyz,np.repeat([-.05,.05],9),cell,8,alpha=.3,kmax=3,width=1,dielectric=78.4,slab=slab,cache_dir=cache)

@pytest.mark.parametrize('slab',[False,True])
def test_electrode_constraints_delta_and_derivative(slab):
    e=make_electrode(slab);rng=np.random.default_rng(3);b=rng.normal(size=18)*100
    a=e.response.solve(b,e.potentials,.3);assert a['potential_residual_V']<1e-12
    assert abs(a['q'].sum()-.3)<1e-12
    db=rng.normal(size=18)*10;df,new=e.response.delta(a,db,-.2)
    exact=e.response.solve(b+db,e.potentials,.1)
    np.testing.assert_allclose(new['q'],exact['q'],atol=1e-12)
    assert abs(df-(exact['free_energy']-a['free_energy']))<1e-9
    pattern=np.r_[-np.ones(9)/2,np.ones(9)/2];d=1e-6
    plus=e.response.solve(b,e.potentials+d*pattern,.3)['free_energy'];minus=e.response.solve(b,e.potentials-d*pattern,.3)['free_energy']
    np.testing.assert_allclose((plus-minus)/(2*d),-VOLT_TO_K*pattern@a['q'],rtol=1e-9)
    bat=e.response.batch_deltas(a,np.stack([db,db]),[-.2,-.2]);np.testing.assert_allclose(bat,[df,df])


def test_electrode_cache(tmp_path):
    a=make_electrode(cache=tmp_path);b=make_electrode(cache=tmp_path)
    assert not a.response.cache_hit and b.response.cache_hit
    np.testing.assert_array_equal(a.response.P,b.response.P)


def test_empty_capacitor_fluctuations():
    e=make_electrode(slab=True);v=np.repeat([-.5,.5],9);r=e.response;s=r.solve(np.zeros(18),e.potentials)
    c=r.capacitance(np.tile(s['q'],(100,1)),v,300)
    delta=1e-5
    fd=v@(r.solve(np.zeros(18),e.potentials+delta*v)['q']-r.solve(np.zeros(18),e.potentials-delta*v)['q'])/(2*delta)
    np.testing.assert_allclose(fd,c['dcharge_dvoltage_e_per_V'],rtol=1e-10)
    samples=r.charge_samples(s,300,60000)
    assert np.max(np.abs(samples.sum(axis=1)))<1e-12
    np.testing.assert_allclose(np.var(samples@v),c['conditional_variance_e2'],rtol=.025)
    assert c['capacitance_F']>0 and c['BO_variance_e2']<1e-20

@pytest.mark.parametrize('slab',[False,True])
@pytest.mark.parametrize('electrode',[False,True])
@pytest.mark.parametrize('move',['insert','delete','translate','tiny_translate','single'])
def test_energy_delta_matches_full(slab,electrode,move):
    e=make_electrode(slab) if electrode else None
    h=IonicHamiltonian([Ion('cat',1),Ion('an',-1)],np.diag([24.,24.,48.]),8,kmax=3,electrodes=e,slab=slab)
    h.set_state([0,1,0,1],[[5,6,22],[11,12,26],[17,4,19],[20,16,30]])
    if move=='insert':rem=[];at=[0,1];ap=[[9,2,23],[15,19,27]]
    elif move=='delete':rem=[0,3];at=[];ap=[]
    elif move=='single':rem=[];at=[0];ap=[[9,3,26]]
    else:rem=[0];at=[0];ap=[[5+(1e-7 if move=='tiny_translate' else 1),6,22]]
    trial=h.propose(rem,at,ap);old=h.energy;h.commit(trial)
    np.testing.assert_allclose(old+trial['delta_energy'],h.full_energy(),atol=1e-8,rtol=1e-11)
    if electrode:assert abs(h.total_charge+h.electrode_state['q'].sum())<1e-12
    h.close()


def test_no_phantom_self_interaction():
    h=IonicHamiltonian([Ion('a',0)],np.eye(3)*24,8,coulomb='none');h.set_state([0],[[5,5,5]])
    t=h.propose([0],[0],[[5,5,5]]);assert t['delta_energy']==0;h.close()


def test_charged_channels_require_electrodes():
    h=IonicHamiltonian([Ion('a',1),Ion('b',-1)],np.eye(3)*24,8,coulomb='none')
    with pytest.raises(ValueError):ElectrolyteGCMC(h,300,[-10,-10],[ExchangeChannel('single',(1,0))])
    h.close()


def test_full_electrochemical_gauge():
    e=make_electrode();ions=[Ion('a',1),Ion('b',-1)]
    h=IonicHamiltonian(ions,e.cell,8,kmax=3,electrodes=e)
    before=h.energy;tr=h.propose([],[0],[[8,9,23]]);du=tr['delta_energy'];z=np.array([-10.,-10.]);ratio=-du/300+z[0]
    shift=.12;e2=make_electrode();e2.potentials+=shift
    h2=IonicHamiltonian(ions,e2.cell,8,kmax=3,electrodes=e2)
    du2=h2.propose([],[0],[[8,9,23]])['delta_energy'];z2=z+VOLT_TO_K*shift*np.array([1,-1])/300
    np.testing.assert_allclose(ratio,-du2/300+z2[0],atol=1e-10)
    h.close();h2.close()


def test_single_ion_interface_trajectory():
    e=make_electrode(True);h=IonicHamiltonian([Ion('a',1),Ion('b',-1)],e.cell,8,kmax=3,electrodes=e,slab=True)
    sim=ElectrolyteGCMC(h,300,np.log([1e-4,1e-4]),[ExchangeChannel('cat',(1,0)),ExchangeChannel('an',(0,1))],allow_single_ion=True,region=FractionalRegion(e.cell,(0,0,14/48),(1,1,34/48)),seed=17)
    out=sim.run(2000,sample_every=10)
    np.testing.assert_allclose(out['electrode_charges'].sum(axis=1)+out['counts']@h.valences,0,atol=1e-10)
    assert np.any(out['counts'][:,0]!=out['counts'][:,1])
    np.testing.assert_allclose(h.energy,h.full_energy(),atol=1e-7)
    h.close()


def test_bias_restart_and_fingerprint(tmp_path):
    h=IonicHamiltonian([Ion('a',1,0,0),Ion('b',-1,0,0)],np.eye(3)*24,8,coulomb='none')
    sim=ElectrolyteGCMC(h,300,np.log([2/24**3,2/24**3]),[ExchangeChannel('salt',(1,1))],bias=HarmonicCountBias((.5,.5),3,.2))
    sim.run(100);f=tmp_path/'state.json';sim.save_checkpoint(f);a=sim.run(500,sample_every=1);p=h.positions.copy()
    sim.restore_checkpoint(f);b=sim.run(500,sample_every=1)
    np.testing.assert_array_equal(a['counts'],b['counts']);np.testing.assert_array_equal(p,h.positions)
    sim.log_activities[0]+=.1
    with pytest.raises(ValueError):sim.restore_checkpoint(f)
    h.close()


def test_region_bounds_and_reservoir_units():
    r=FractionalRegion(np.diag([20.,20.,40.]),(0,0,.25),(1,1,.75))
    assert np.isclose(r.volume,8000)
    assert not r.contains([5,5,9]);assert r.contains(r.sample(np.random.default_rng(2),100))
    np.testing.assert_allclose(np.exp(log_activity_from_molarity(1.,0.,300)),6.02214076e-4)


def test_existing_molecular_gcmc_frozen_bias(tmp_path):
    from poremc.engine import GCMC,ZeroPotential
    from poremc.topology import Molecule
    m=Molecule('ideal',[0],[0]);b=HarmonicCountBias((1,),4,.5)
    sim=GCMC([m],ZeroPotential(),np.eye(3)*24,300,[2/24**3],trials=1,move_weights={'insert':.5,'delete':.5},count_bias=b,seed=29)
    out=sim.run(18000,burnin=500,sample_every=2,record_energy=False);x=np.array(out['counts'])[:,0]
    ns=np.arange(40);lw=ns*np.log(2)-gammaln(ns+1)-b.values(ns[:,None]);p=np.exp(lw-logsumexp(lw))
    assert abs(x.mean()-p@ns)<.1
    path=tmp_path/'molecular.json';sim.save_checkpoint(path);expected=sim.run(100,record_energy=False)
    restored=GCMC.from_checkpoint(path,ZeroPotential());actual=restored.run(100,record_energy=False)
    assert expected['counts']==actual['counts']


def test_joint_electrochemical_response():
    e=make_electrode();rng=np.random.default_rng(2);n=rng.poisson([2,4],size=(1000,2))
    q=np.array([e.response.solve(np.ones(18)*0,e.potentials,-(x[0]-x[1]))['q'] for x in n])
    r=electrochemical_response(n,q,np.repeat([-.5,.5],9),e.response,300)
    assert r['generalized_response'].shape==(3,3)
    assert r['eigenvalues'].min()>-1e-12


def test_voltage_reweight_empty_capacitor():
    e=make_electrode(True);v=np.repeat([-.5,.5],9);s=e.response.solve(np.zeros(18),e.potentials)
    q=np.tile(s['q'],(20,1));r=electrochemical_reweight(np.zeros((20,2)),q,[0,0],.07,v,e.response,300)
    direct=e.response.solve(np.zeros(18),e.potentials+.07*v)
    np.testing.assert_allclose(r['mean_electrode_charges'],direct['q'],atol=1e-12)
    np.testing.assert_allclose(r['log_partition_ratio'],-(direct['free_energy']-s['free_energy'])/300,atol=1e-12)


def test_full_charge_landscape_not_BO_delta_function():
    e=make_electrode(True);v=np.repeat([-.5,.5],9);s=e.response.solve(np.zeros(18),e.potentials)
    sigma=np.sqrt(300*(v@e.response.P@v));mu=s['q']@v;edges=np.linspace(mu-6*sigma,mu+6*sigma,101)
    r=interfacial_landscape([0],s['q'][None,:],v,e.response,300,[-.5,.5],edges)
    assert r['captured_mass']>.99999999
    mid=(edges[:-1]+edges[1:])/2
    mean=float(np.sum(r['probability_mass'][0]*mid));var=float(np.sum(r['probability_mass'][0]*(mid-mean)**2))
    np.testing.assert_allclose(var,sigma*sigma,rtol=.003)


def test_mbar_compression_is_exact():
    x=np.tile(np.array([0.,1.,2.,3.]),60)
    u=np.array([x*x*.2,(x-1.2)**2*.4,(x-2.2)**2*.3])
    a=MBAR(u,[80,80,80],compress=True);b=MBAR(u,[80,80,80],compress=False)
    assert a.unique_reduced_potentials==4 and b.unique_reduced_potentials==240
    np.testing.assert_allclose(a.f,b.f,rtol=1e-9,atol=1e-10)
    np.testing.assert_allclose(a.log_weights(x*.1),b.log_weights(x*.1),atol=1e-10)
