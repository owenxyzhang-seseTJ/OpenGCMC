import numpy as np
import pytest

from poremc.ensembles import GrandCanonicalReplicaExchange, activity_exchange_log_ratio
from poremc.fast import FastGCMC
from poremc.potentials import ClassicalPotential
from poremc.topology import Molecule


def _mixture(kind='none'):
    a = Molecule.from_dict({'name':'single','sigma':[2.1],'epsilon':[50.]})
    b = Molecule.from_dict({'name':'dimer','sigma':[2.3,2.4],'epsilon':[80.,70.],
                            'charge':[.25,-.25],'parents':[-1,0],'bond_lengths':[0.,1.25]})
    cell = np.array([[34.,0.,0.],[6.,35.,0.],[8.,4.,36.]])
    potential = ClassicalPotential([a,b],cell,8.,coulomb=kind,kmax=4,
                   host_positions=[[5.,6.,7.],[11.,9.,8.]],host_sigma=[2.9,2.9],
                   host_epsilon=[80.,80.],host_charge=[.4,-.4])
    return a,b,potential


@pytest.mark.parametrize('mode',['exact','delayed'])
def test_incremental_mixture_and_suffix_match_rebuild(mode,tmp_path):
    a,b,pot = _mixture('ewald')
    options = dict(seed=321,trials=[2,[2,3]],sampling_mode=mode,host_grid_mesh=8,
                   partial_regrowth=True,species_probabilities=[1,3])
    act = np.array([8.,13.])/np.linalg.det(pot.cell)
    with FastGCMC([a,b],pot,pot.cell,298,act,guest_update='incremental',**options) as local, \
         FastGCMC([a,b],pot,pot.cell,298,act,guest_update='rebuild',**options) as ref:
        for s,p in [(0,[[33.8,2.,3.]]),(1,[[.3,7.,6.],[1.55,7.,6.]])]:
            for sim in (local,ref):
                sim.add_particle(s,p)
        trial=np.array([[[33.95,7.,6.],[35.2,7.,6.]],[[31.,2.,7.],[32.25,2.,7.]]])
        np.testing.assert_allclose(local.trial_molecule_energies(trial,exclude=0,species=1),
                                   ref.trial_molecule_energies(trial,exclude=0,species=1),atol=1e-10)
        ra=local.run(1400,sample_every=20)
        rb=ref.run(1400,sample_every=20)
        assert ra['counts']==rb['counts']
        np.testing.assert_allclose(ra['energies_kelvin'],rb['energies_kelvin'],atol=1e-8)
        assert local.profile['guest_local_updates']>50 and local.profile['guest_environment_builds']==0
        np.testing.assert_allclose(local.total_energy(),pot.total_energy(local.entries()),atol=1e-7)
        path=tmp_path/'mixture.json'
        local.save_checkpoint(path)
        before=local.run(600)
        with FastGCMC.from_checkpoint(path,pot) as resumed:
            assert resumed.run(600)['counts']==before['counts']
            for (_,x),(_,y) in zip(local.entries(),resumed.entries()):
                np.testing.assert_array_equal(x,y)


def test_mixture_matches_independent_poisson_and_zero_activity():
    species=[Molecule.from_dict({'name':name,'sigma':[0.],'epsilon':[0.]}) for name in ('A','B')]
    cell=np.eye(3)*20
    pot=ClassicalPotential(species,cell,8.)
    weights=dict(insert=1,delete=2)
    with FastGCMC(species,pot,cell,298,np.array([2.,5.])/8000,seed=513,
                  species_probabilities=[2,1],move_weights=weights,trials=1) as sim:
        counts=np.array(sim.run(150000,burnin=1000,sample_every=5,record_energy=False)['counts'])
        np.testing.assert_allclose(counts.mean(0),[2.,5.],atol=.15)
        np.testing.assert_allclose(counts.var(0),[2.,5.],atol=.3)
        assert abs(np.cov(counts.T)[0,1]) < .14
    with FastGCMC(species,pot,cell,298,[0.,5/8000],trials=1,move_weights=weights) as sim:
        assert all(row[0]==0 for row in sim.run(1000,record_energy=False)['counts'])


def test_deferred_pme_tracks_fully_refreshed_state(tmp_path):
    a,b,pot=_mixture('ewald')
    options=dict(seed=924,trials=2,reciprocal_method='pme',pme_mesh=16,guest_update='incremental')
    activities=np.array([2.,4.])/np.linalg.det(pot.cell)
    with FastGCMC([a,b],pot,pot.cell,298,activities,pme_refresh=1,**options) as full, \
         FastGCMC([a,b],pot,pot.cell,298,activities,pme_refresh=8,**options) as sparse:
        trial=np.array([[13.,4.,8.],[14.25,4.,8.]])
        for sim in (full,sparse):
            sim.add_particle(1,trial)
        assert sparse.long_range_profile['pme_pending_nodes']>0
        np.testing.assert_allclose(sparse.total_energy(),full.total_energy(),atol=1e-8)
        state=tmp_path/'pending.json'
        sparse.save_checkpoint(state)
        a1=sparse.run(500,burnin=10,sample_every=5)
        a2=full.run(500,burnin=10,sample_every=5)
        with FastGCMC.from_checkpoint(state,pot) as resume:
            b1=resume.run(500,burnin=10,sample_every=5)
        assert a1['counts']==a2['counts']==b1['counts']
        np.testing.assert_allclose(a1['energies_kelvin'],a2['energies_kelvin'],atol=3e-8,rtol=1e-11)
        np.testing.assert_array_equal(a1['energies_kelvin'],b1['energies_kelvin'])
        assert sparse.long_range_profile['fft_transforms'] < full.long_range_profile['fft_transforms']/2


def test_raw_cache_reuses_only_compatible_fields(tmp_path):
    a,b,pot=_mixture('ewald')
    options=dict(sampling_mode='delayed',host_grid_mesh=8,host_field_cache=tmp_path)
    with FastGCMC([a,b],pot,pot.cell,298,[.001,.002],**options) as cold:
        fields=cold.acceleration_profile['host_grid']
        assert fields['raw_field_builds']==4
        trial=[[14.1,6.3,5.4],[15.35,6.3,5.4]]
        expected=cold.trial_surrogate_molecule_energies(trial,species=1)
    with FastGCMC([a,b],pot,pot.cell,298,[.009,.004],**options) as hot:
        assert hot.acceleration_profile['host_grid']['raw_field_loads']==4
        np.testing.assert_array_equal(hot.trial_surrogate_molecule_energies(trial,species=1),expected)
    changed=ClassicalPotential([a,b],pot.cell,8.,coulomb='ewald',kmax=4,
                  host_positions=pot.host_positions,host_sigma=pot.host_sigma,
                  host_epsilon=pot.host_epsilon,host_charge=2*pot.host_charge)
    with FastGCMC([a,b],changed,pot.cell,315,[.001,.002],**options) as modified:
        fields=modified.acceleration_profile['host_grid']
        assert fields['raw_field_loads']==3 and fields['raw_field_builds']==1
    next(tmp_path.glob('lj-*.bin')).write_bytes(b'short write')
    with FastGCMC([a,b],pot,pot.cell,298,[.001,.002],**options) as repaired:
        assert repaired.acceleration_profile['host_grid']['raw_field_builds']==1


def test_activity_exchange_preserves_target_labels():
    assert activity_exchange_log_ratio([0],[2],[0.],[.1]) == -np.inf
    assert activity_exchange_log_ratio([2],[5],[.1],[.2])==pytest.approx(-3*np.log(2))
    m=Molecule.from_dict({'name':'ideal','sigma':[0.],'epsilon':[0.]})
    h=np.eye(3)*20
    pot=ClassicalPotential([m],h,8.)
    with FastGCMC([m],pot,h,298,[2./8000],seed=4,trials=1) as a, \
         FastGCMC([m],pot,h,298,[4./8000],seed=8,trials=1) as b:
        rex=GrandCanonicalReplicaExchange([a,b],seed=932)
        rex.advance(300)
        records=[rex.advance(180,sample_every=5) for _ in range(250)]
        means=[np.mean([r[i]['counts'] for r in records]) for i in range(2)]
        np.testing.assert_allclose(means,[2.,4.],atol=.2)
        assert rex.accepts[0]>10


def test_projected_electrode_bounds_and_osmotic_reference():
    from poremc.electrode import ElectrodeResponse, solve_projected_charge
    from poremc.ensembles import osmotic_phase_probabilities

    rng=np.random.default_rng(153)
    x=rng.normal(size=(14,10))
    a=x@x.T+np.eye(14)*2.1
    b=rng.normal(size=14)*170.
    volts=rng.uniform(-.1,.15,14)
    reference=ElectrodeResponse(a).solve(b,volts,.75)
    coarse=solve_projected_charge(lambda p:a@p,b,volts,.75,eigenvalue_lower_bound=2.,max_iterations=2)
    assert not coarse['converged']
    assert coarse['free_energy_lower_bound']<=reference['free_energy']<=coarse['free_energy']
    fine=solve_projected_charge(lambda p:a@p,b,volts,.75,eigenvalue_lower_bound=2.)
    assert fine['converged']
    np.testing.assert_allclose(fine['q'],reference['q'],rtol=1e-8,atol=1e-8)
    np.testing.assert_allclose(fine['q'].sum(),.75,atol=1e-12)
    p=osmotic_phase_probabilities(300.,0.,[0.,300.*np.log(4)],[100.,140.],[0.,np.log(2)])
    np.testing.assert_allclose(p,[2/3,1/3],atol=1e-12)
