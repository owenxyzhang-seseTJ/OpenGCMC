"""Independent target-distribution and residual checks for native delayed CBMC.

The exact probes must keep evaluating the finite Ewald Hamiltonian.  The
surrogate may have substantial interpolation error; it only determines a
proposal and a first acceptance stage.  Tests deliberately exercise that
difference rather than checking that two surrogate code paths agree alone.
"""
from __future__ import annotations
import importlib.util
import math
from pathlib import Path
import tempfile
import numpy as np

from poremc.fast import FastGCMC
from poremc.potentials import ClassicalPotential
from poremc.proposals import VoxelProposal
from poremc.topology import Molecule

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("independent_longrange_physics",ROOT/'tests/test_native_longrange_physics.py')
ORACLE=importlib.util.module_from_spec(spec);spec.loader.exec_module(ORACLE)


def block_standard_error(values, blocks=40):
    x=np.asarray(values,float)
    used=len(x)//blocks*blocks
    return np.std(x[:used].reshape(blocks,-1,*x.shape[1:]).mean(axis=1),axis=0,ddof=1)/math.sqrt(blocks)


def check_prefix_and_residual():
    errors=[];residuals=[];telescoping=[]
    for kind in ('rigid','flexible'):
        molecule,potential,local=ORACLE.electrostatic_fixture(kind,skew=True,charge_scale=.7)
        with FastGCMC(molecule,potential,potential.cell,500.,[1e-5],trials=5,
                      sampling_mode='delayed',host_grid_mesh=16,host_grid_cap=2e4) as sim:
            for offset in ([7.,8.,10.],[18.,20.,23.]):sim.add_particle(0,local+offset)
            trial=local+np.array([11.,15.,9.])
            for exclude in (None,0):
                bath=[(s,x) for i,(s,x) in enumerate(sim.entries()) if i!=exclude]
                expected=ORACLE.independent_guest_energy(potential,bath+[(0,trial)])-ORACLE.independent_guest_energy(potential,bath)
                exact=float(sim.trial_molecule_energies(trial,exclude=exclude)[0])
                reference=float(sim.trial_surrogate_molecule_energies(trial,exclude=exclude)[0])
                errors.append(abs(exact-expected));residuals.append(exact-reference)
                stages=[]
                for site in range(molecule.n_atoms):
                    stages.append(float(sim.trial_surrogate_site_energies(trial[:site],trial[site:site+1],exclude=exclude)[0]))
                telescoping.append(abs(sum(stages)-reference))
    assert max(errors)<2e-8
    assert max(telescoping)<2e-8
    assert max(abs(x) for x in residuals)>.01, 'fixture must exercise a nonzero correction'
    return dict(max_exact_reference_error_K=max(errors),max_surrogate_prefix_error_K=max(telescoping),
                selected_residuals_K=residuals)


def check_nonuniform_root_poisson():
    molecule=Molecule('ideal_root_probe',[0.],[0.],charge=[0.])
    cell=np.diag([12.,14.,16.]);volume=np.linalg.det(cell)
    potential=ClassicalPotential([molecule],cell,5.,coulomb='ewald',alpha=.3,kmax=2,algorithm='cached')
    probabilities=np.full((2,3,4),.15/23);probabilities[0,0,0]=.85
    proposal=VoxelProposal(cell,probabilities)
    with FastGCMC(molecule,potential,cell,300.,[2.7/volume],seed=50811,trials=7,
                  move_weights={'insert':.37,'delete':.63},sampling_mode='delayed',
                  host_grid_mesh=8,root_proposal=proposal) as sim:
        result=sim.run(80000,burnin=3000,sample_every=5,record_energy=False)
        counts=np.asarray(result['counts'],float)[:,0]
        mean=float(counts.mean());variance=float(counts.var());se=float(block_standard_error(counts))
        assert abs(mean-2.7)<max(.08,6*se)
        assert abs(variance-2.7)<.25
        return dict(expected_mean=2.7,expected_variance=2.7,mean=mean,variance=variance,
                    mean_block_se=se,samples=len(counts),profile=sim.profile)


def canonical_fixture():
    molecule=Molecule('charged_one',[1.4],[35.],charge=[.12])
    cell=np.diag([18.,20.,22.])
    host=np.array([[.23,.28,.34],[.71,.64,.72]])@cell
    potential=ClassicalPotential([molecule],cell,7.,host_positions=host,
        host_sigma=[1.5,1.5],host_epsilon=[40.,40.],host_charge=[.25,-.25],
        coulomb='ewald',alpha=.3,kmax=3,algorithm='cached',allow_charge_background=True)
    return molecule,potential,1200.


def independent_one_body(potential, positions):
    """Independent host field and periodic self/background constants for N=1."""
    from scipy.special import erfc
    p=potential;x=np.asarray(positions);inv=np.linalg.inv(p.cell);volume=np.linalg.det(p.cell)
    molecule=p.species[0];charge=float(molecule.charge[0]);out=np.zeros(len(x))
    for rj,sj,ej,qj in zip(p.host_positions,p.host_sigma,p.host_epsilon,p.host_charge):
        f=(x-rj)@inv;d=(f-np.rint(f))@p.cell;r=np.linalg.norm(d,axis=1)
        active=r<p.cutoff;rv=r[active];sigma=(molecule.sigma[0]+sj)/2;epsilon=math.sqrt(molecule.epsilon[0]*ej)
        t6=(sigma/rv)**6;c6=(sigma/p.cutoff)**6
        out[active]+=4*epsilon*(t6*t6-t6-(c6*c6-c6) if p.shifted else t6*t6-t6)
        out[active]+=167100.94689828736*charge*qj*erfc(p.alpha*rv)/rv
    axis=range(-3,4)
    ids=np.array([(a,b,c) for a in axis for b in axis for c in axis if (a,b,c)!=(0,0,0)])
    vectors=2*np.pi*ids@inv.T;k2=np.sum(vectors*vectors,axis=1)
    weights=167100.94689828736*2*np.pi/volume*np.exp(-k2/(4*p.alpha**2))/k2
    host_s=np.exp(1j*(p.host_positions@vectors.T)).T@p.host_charge
    for start in range(0,len(x),2048):
        phase=x[start:start+2048]@vectors.T
        out[start:start+2048]+=2*charge*(np.cos(phase)@(weights*host_s.real)+np.sin(phase)@(weights*host_s.imag))
    out+=charge*charge*weights.sum()-167100.94689828736*p.alpha/math.sqrt(math.pi)*charge*charge
    out-=167100.94689828736*np.pi/(2*p.alpha**2*volume)*(2*p.host_charge.sum()*charge+charge*charge)
    return out


def observable(positions,cell):
    f=np.asarray(positions)@np.linalg.inv(cell)
    return np.column_stack((np.cos(2*np.pi*f[:,0]),np.sin(2*np.pi*f[:,0]),np.cos(2*np.pi*f[:,2])))


def quadrature(potential,temperature,n):
    f=np.stack(np.meshgrid(*[(np.arange(n)+.5)/n]*3,indexing='ij'),axis=-1).reshape(-1,3)
    x=f@potential.cell;u=independent_one_body(potential,x)
    w=np.exp(-(u-u.min())/temperature);w/=w.sum()
    return w@observable(x,potential.cell)


def check_canonical_quadrature(mode='delayed'):
    molecule,potential,temperature=canonical_fixture()
    reference=quadrature(potential,temperature,56)
    refined=quadrature(potential,temperature,80)
    quadrature_error=np.abs(reference-refined)
    assert quadrature_error.max()<.002
    samples=[]
    with FastGCMC(molecule,potential,potential.cell,temperature,[1e-5],seed=141069,trials=5,
                  move_weights={'regrow':1.},sampling_mode=mode,host_grid_mesh=8,
                  host_grid_cap=50.,grid_root_bias=.8) as sim:
        sim.add_particle(0,[[9.,10.,11.]])
        # Deliberately cap the proposal field at only 50 K.  The physical
        # Hamiltonian remains unchanged, making omission of correction visible.
        f=np.stack(np.meshgrid(*[(np.arange(32)+.5)/32]*3,indexing='ij'),axis=-1).reshape(-1,3)
        x=f@potential.cell
        u0=sim.trial_surrogate_molecule_energies(x[:,None,:],exclude=0)
        w=np.exp(-(u0-u0.min())/temperature);w/=w.sum()
        uncorrected=w@observable(x,potential.cell)
        sim.run(2000,sample_every=2000,record_energy=False)
        for _ in range(6000):
            sim.run(15,sample_every=15,record_energy=False)
            samples.append(sim.particles[0].positions[0])
        obs=observable(samples,potential.cell);mean=obs.mean(axis=0);se=block_standard_error(obs)
        error=np.abs(mean-refined)
        assert np.all(error<6*se+quadrature_error+.004),(mean,refined,se)
        assert np.max(np.abs(uncorrected-refined)/(se+.001))>6., 'uncorrected proposal must differ detectably'
        # The exact probe also agrees with an independent field, not its grid.
        probe=np.asarray(samples)[::400,None,:]
        expected=independent_one_body(potential,probe[:,0,:])
        actual=sim.trial_molecule_energies(probe,exclude=0)
        np.testing.assert_allclose(actual,expected,atol=2e-8,rtol=2e-12)
        surrogate=sim.trial_surrogate_molecule_energies(probe,exclude=0)
        residual=actual-surrogate
        assert np.ptp(residual)>1., 'coarse-grid fixture must have nonconstant residual'
        profile=sim.acceleration_profile
        if mode=='reference-mh':
            assert profile['reference_proposals']==profile['proposals_reaching_exact']
            assert profile['exact_evaluations']==2*profile['proposals_reaching_exact']
            assert profile['accepted']==sim.counters['regrow']['accepted']
        return dict(observables=['cos(2pi*x/Lx)','sin(2pi*x/Lx)','cos(2pi*z/Lz)'],
                    quadrature=refined.tolist(),quadrature_refinement_difference=quadrature_error.tolist(),
                    uncorrected_surrogate_quadrature=uncorrected.tolist(),
                    sampled_mean=mean.tolist(),block_se=se.tolist(),samples=len(samples),
                    residual_range_K=[float(residual.min()),float(residual.max())],profile=sim.profile)


def check_all_moves_and_restart():
    molecule,potential,local=ORACLE.electrostatic_fixture('flexible',skew=True,charge_scale=.15)
    errors=[];acceptance={}
    with tempfile.TemporaryDirectory() as directory:
        for move in ('exchange','translate','rotate','regrow'):
            weights={'insert':.5,'delete':.5} if move=='exchange' else {move:1.}
            with FastGCMC(molecule,potential,potential.cell,1500.,[2/np.linalg.det(potential.cell)],
                          seed=10606,trials=4,move_weights=weights,sampling_mode='delayed',
                          host_grid_mesh=8,host_grid_cache=str(Path(directory)/'field.bin')) as sim:
                sim.add_particle(0,local+[8.,9.,10.]);sim.add_particle(0,local+[21.,22.,23.])
                for _ in range(3):
                    sim.run(100,sample_every=100,record_energy=False)
                    expected=ORACLE.independent_guest_energy(potential,sim.entries())
                    errors.append(abs(sim.total_energy()-expected))
                requested=('insert','delete') if move=='exchange' else (move,)
                for name in requested:
                    assert sim.counters[name]['accepted']>0
                    acceptance[name]=sim.counters[name]
                if move=='regrow':
                    path=Path(directory)/'state.json';sim.save_checkpoint(path)
                    continued=sim.run(80,sample_every=10,record_energy=True)
                    positions=[p.positions.copy() for p in sim.particles]
                    with FastGCMC.from_checkpoint(path,potential) as resumed:
                        repeated=resumed.run(80,sample_every=10,record_energy=True)
                        assert continued['counts']==repeated['counts']
                        assert continued['energies_kelvin']==repeated['energies_kelvin']
                        for expected,actual in zip(positions,resumed.particles):np.testing.assert_array_equal(expected,actual.positions)
        assert max(errors)<2e-8
    return dict(max_full_energy_error_K=max(errors),moves=acceptance,exact_checkpoint_continuation=True)


def check_cache_identity():
    molecule,potential,local=ORACLE.electrostatic_fixture('rigid',skew=True,charge_scale=.6)
    trial=(local+np.array([9.,11.,13.]))[None,:,:]
    changed=ClassicalPotential([molecule],potential.cell,potential.cutoff,
        host_positions=potential.host_positions,host_sigma=potential.host_sigma,
        host_epsilon=potential.host_epsilon,host_charge=potential.host_charge*1.7,
        coulomb='ewald',alpha=potential.alpha,kmax=tuple(potential.ewald.kmax),algorithm='cached')
    def value(potential,path,temperature):
        with FastGCMC(molecule,potential,potential.cell,temperature,[1e-5],sampling_mode='delayed',
                      host_grid_mesh=16,host_grid_cache=path) as sim:
            return float(sim.trial_surrogate_molecule_energies(trial)[0]),sim.profile
    with tempfile.TemporaryDirectory() as directory:
        path=str(Path(directory)/'reused.bin')
        first,profile1=value(potential,path,300.)
        repeated,profile2=value(potential,path,700.)
        repeated_fresh,_=value(potential,None,700.)
        changed_cached,profile3=value(changed,path,300.)
        changed_fresh,_=value(changed,None,300.)
        # Raw fields are temperature-independent; Boltzmann interpolation may
        # reweight them at a new T.  A cache hit must equal a fresh build at T.
        assert repeated==repeated_fresh
        assert abs(changed_cached-changed_fresh)<1e-10
        assert abs(changed_cached-first)>1e-3
        return dict(temperature_reweighting_change_K=repeated-first,
                    unchanged_grid_reuse_error_K=abs(repeated-repeated_fresh),
                    changed_field_error_K=abs(changed_cached-changed_fresh),
                    physical_field_change_K=changed_cached-first,
                    construction_profiles=[profile1,profile2,profile3])


def test_prefix_and_residual():check_prefix_and_residual()
def test_nonuniform_root_poisson():check_nonuniform_root_poisson()
def test_canonical_quadrature():check_canonical_quadrature()
def test_reference_mh_canonical_quadrature():check_canonical_quadrature('reference-mh')
def test_all_moves_and_restart():check_all_moves_and_restart()
def test_cache_identity():check_cache_identity()


def test_linear_force_shift():
    molecule=Molecule('site',[3.2],[84.])
    cell=np.diag([27.,27.,27.])
    cutoff=10.
    rs=np.array([3.1, 6., 9.99, 9.999, 10., 10.01])
    xyz=np.column_stack((rs,np.zeros((len(rs),2))))
    def raw(r):
        return 4*84.*((3.2/r)**12-(3.2/r)**6)
    h=1e-4
    slope=(raw(cutoff+h)-raw(cutoff-h))/(2*h)
    expected=np.where(rs<cutoff,raw(rs)-raw(cutoff)-(rs-cutoff)*slope,0)
    for backend,algorithm in (('reference','direct'),('native','cached'),('native','direct')):
        pot=ClassicalPotential([molecule],cell,cutoff,host_positions=[[0,0,0]],
                               host_sigma=[3.2],host_epsilon=[84.],backend=backend,
                               algorithm=algorithm,shifted='force')
        with FastGCMC(molecule,pot,cell,300.,[1e-6]) as sim:
            actual=sim.trial_molecule_energies(xyz[:,None,:])
            np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-8)
            # Check the incremental guest-cell path as well as the host query.
            sim.add_particle(0,[[0.,0.,-0.2]])
            actual=sim.trial_molecule_energies(xyz[:,None,:])
            rg=np.hypot(rs,.2)
            guest=np.where(rg<cutoff,raw(rg)-raw(cutoff)-(rg-cutoff)*slope,0)
            np.testing.assert_allclose(actual,expected+guest,rtol=1e-10,atol=1e-8)
        with pot._new_evaluator([[0,0,0]],[3.2],[84.],[0.]) as env:
            energy=env.energy(xyz[:,None,:],molecule.sigma,molecule.epsilon,molecule.charge)
            np.testing.assert_allclose(energy,expected,rtol=1e-10,atol=1e-8)
