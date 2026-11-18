import math
import numpy as np
import pytest
from poremc.hydration import *
from poremc.reactive import *
from poremc.electrode import GaussianElectrodes, VOLT_TO_K
from poremc.electrolyte import FractionalRegion
from poremc.calibration import *


def electrode(shift=0., slab=False):
    cell = np.diag([24., 24., 48.])
    xyz = np.array([[x, y, z] for z in (12., 36.) for x in (4., 12., 20.) for y in (4., 12., 20.)])
    return GaussianElectrodes(xyz, np.repeat([-.05+shift, .05+shift], 9), cell, 8.,
                             alpha=.3, kmax=3, dielectric=1., width=1., slab=slab)


def tiny_water(electrodes=None, slab=False, backend='native'):
    t = [spce_water(), point_species('X+', 1.), point_species('Y-', -1.)]
    cell = electrodes.cell if electrodes is not None else np.diag([24., 24., 48.])
    h = MolecularHamiltonian(t, cell, cutoff=8, kmax=3, alpha=.3,
                              electrodes=electrodes, slab=slab, backend=backend)
    h.set_molecules([Molecule(0, 0, t[0].geometry+[5, 6, 18]),
                     Molecule(0, 0, t[0].geometry+[10, 8, 22]),
                     Molecule(1, 0, [[5, 10, 24]]), Molecule(2, 0, [[15, 16, 28]])])
    return h


def ideal_reaction(template, edge, *, pH=7., potential=0., state=0, seed=14):
    templates, edges = prepare_reactions([template], [edge])
    h = MolecularHamiltonian(templates, np.eye(3)*24, cutoff=8, coulomb='none')
    h.set_molecules([Molecule(0, state, [[12, 12, 12]], mobile=False, exchangeable=False)])
    return ReactiveGCMC(h, Reservoirs(300., pH, potential), reactions=edges,
                        move_weights=(.2, 0, .8), seed=seed)


def test_spce_geometry_and_units():
    t = spce_water()
    np.testing.assert_allclose(np.linalg.norm(t.geometry[1:], axis=1), 1.)
    assert abs(t.geometry[1] @ t.geometry[2] + 1/3) < 1e-14
    assert abs(t.states[0].charge) < 1e-15
    assert abs(t.states[0].epsilon[0] - .1553*4184/8.31446261815324) < 1e-10
    assert t.states[0].sigma == (3.166, 0., 0.)


def test_rotation_haar_moments():
    rng = np.random.default_rng(22)
    x = np.array([orientation(rng)[:, 0] for _ in range(15000)])
    np.testing.assert_allclose(x.mean(axis=0), 0, atol=.02)
    np.testing.assert_allclose(x.T @ x/len(x), np.eye(3)/3, atol=.015)


def test_reject_double_dielectric():
    with pytest.raises(ValueError, match='double-screen'):
        MolecularHamiltonian([spce_water()], np.eye(3)*24, dielectric=78.4)


def test_rigid_geometry_guard():
    h = MolecularHamiltonian([spce_water()], np.eye(3)*24)
    xyz = spce_water().geometry.copy()
    xyz[1, 0] += .01
    with pytest.raises(ValueError, match='rigid geometry'):
        h.set_molecules([Molecule(0, 0, xyz)])
    h.close()


@pytest.mark.parametrize('with_electrode', [False, True])
@pytest.mark.parametrize('slab', [False, True])
def test_multisite_increment_full(with_electrode, slab):
    e = electrode(slab=slab) if with_electrode else None
    h = tiny_water(e, slab=slab)
    region = FractionalRegion(h.cell, lower=(0, 0, .25), upper=(1, 1, .75)) if slab else None
    sim = ReactiveGCMC(h, region=region, move_weights=(1, 0, 0), seed=77)
    for _ in range(35):
        sim.displace()
        np.testing.assert_allclose(h.energy, h.full_energy(), atol=2e-7, rtol=0)
    if e:
        assert abs(sim.check_neutrality()) < 1e-10
        solve = e.solve(np.r_[h.atomic.host_positions, h.atomic.positions], np.r_[h.atomic.host_charge, h.atomic.valences[h.atomic.types]])
        np.testing.assert_allclose(solve['q'], h.atomic.electrode_state['q'], atol=1e-12)
    h.close()


def test_native_reference_same_water_energy():
    a, b = tiny_water(), tiny_water(backend='reference')
    np.testing.assert_allclose(a.energy, b.energy, atol=1e-8)
    m = a.molecules[0].clone(); m.positions += .02
    np.testing.assert_allclose(a.propose([0], [m])['delta_energy'], b.propose([0], [m])['delta_energy'], atol=1e-8)
    a.close(); b.close()


def test_full_vs_delta_water_insertion_deletion():
    h = tiny_water(electrode())
    original = h.energy
    m = Molecule(0, 0, spce_water().geometry+[17, 8, 30])
    tr = h.propose([], [m]); d = tr['delta_energy']; h.commit(tr)
    np.testing.assert_allclose(h.energy-original, d, atol=1e-8)
    np.testing.assert_allclose(h.energy, h.full_energy(), atol=1e-7)
    back = h.propose([len(h.molecules)-1], [])
    np.testing.assert_allclose(back['delta_energy'], -d, atol=1e-7)
    h.commit(back)
    np.testing.assert_allclose(h.energy, original, atol=1e-7)
    h.close()


@pytest.mark.parametrize('trials', [1, 4])
@pytest.mark.parametrize('pins', [.25, .7])
def test_water_grand_ideal_poisson(trials, pins):
    geom = spce_water().geometry
    t = MolecularTemplate('ideal rigid', geom, [MolecularState('water', (0, 0, 0), (0, 0, 0), (0, 0, 0))])
    h = MolecularHamiltonian([t], np.eye(3)*20, cutoff=8, coulomb='none')
    c = MolecularExchange((1,), trials=trials, insert_probability=pins)
    sim = ReactiveGCMC(h, log_activities=[math.log(4/8000)], exchanges=[c],
                       move_weights=(0, 1, 0), seed=42)
    out = sim.run(9000, burnin=500, sample_every=1)
    assert abs(out['counts'].mean()-4) < .25
    assert abs(out['counts'].var()-4) < .65
    assert out['physical_time'] is None
    h.close()


def test_multivalent_neutral_molecular_channel():
    t = [point_species('Z2+', 2, 0, 0), point_species('X-', -1, 0, 0)]
    h = MolecularHamiltonian(t, np.eye(3)*20, cutoff=8, coulomb='none')
    sim = ReactiveGCMC(h, log_activities=np.log(np.array([2, 4])/8000),
                       exchanges=[MolecularExchange((1, 2))], move_weights=(0, 1, 0))
    out = sim.run(2000, sample_every=2)
    assert np.all(out['counts'] @ [2, -1] == 0)
    h.close()


@pytest.mark.parametrize('pH', [6., 7., 8.])
def test_henderson_hasselbalch_sampling(pH):
    sim = ideal_reaction(acid_site(pKa=7., sigma=0., epsilon=0.), ReactionEdge(0, 0, 1), pH=pH)
    out = sim.run(9000, sample_every=1, burnin=300)
    expected = 1/(1+10**(pH-7))
    assert abs(out['state_counts'][:, 1].mean()-expected) < .025
    sim.h.close()


@pytest.mark.parametrize('E', [-.05, 0., .05])
def test_nernst_sampling(E):
    sim = ideal_reaction(redox_site(E0_V=0., sigma=0., epsilon=0.), ReactionEdge(0, 0, 1), potential=E)
    out = sim.run(8000, sample_every=1, burnin=300)
    expected = 1/(1+math.exp(VOLT_TO_K*E/300))
    assert abs(out['state_counts'][:, 1].mean()-expected) < .03
    sim.h.close()


def test_calibration_offsets_sign():
    r = Reservoirs(300., 7., .1)
    t = acid_site(pKa=7., reference_deprotonation_kelvin=900.)
    # At pH=pKa, a physical deprotonation cost 900 K is canceled.
    assert abs(900/300+r.state_bias(t.states[0])-r.state_bias(t.states[1])) < 1e-13
    t = redox_site(E0_V=.1, reference_reduction_kelvin=600.)
    assert abs(600/300+r.state_bias(t.states[1])-r.state_bias(t.states[0])) < 1e-13


def test_charge_stoichiometry_guard():
    a = MolecularState('a', (0,), (0,), (0,))
    b = MolecularState('b', (0,), (0,), (0,), bound_electrons=1)
    t = MolecularTemplate('bad', [[0, 0, 0]], [a, b])
    with pytest.raises(ValueError, match='charge change'):
        prepare_reactions([t], [ReactionEdge(0, 0, 1)])


def test_no_charged_titration_without_compensation():
    t, e = prepare_reactions([acid_site()], [ReactionEdge(0, 0, 1)])
    h = MolecularHamiltonian(t, np.eye(3)*24)
    with pytest.raises(ValueError, match='electrodes'):
        ReactiveGCMC(h, reactions=e)
    h.close()


@pytest.mark.parametrize('kind', ['acid', 'redox'])
def test_global_potential_gauge_invariance(kind):
    t = acid_site() if kind=='acid' else redox_site()
    deltas = []
    for shift in [0., .37]:
        h = MolecularHamiltonian([t], electrode(shift).cell, cutoff=8, alpha=.3, kmax=3,
                                  electrodes=electrode(shift))
        h.set_molecules([Molecule(0, 0, [[11, 12, 22]], mobile=False, exchangeable=False)])
        r = Reservoirs(300., 7., .05+shift, shift)
        m = h.molecules[0].clone(); m.state = 1
        tr = h.propose([0], [m])
        deltas.append(tr['delta_energy']/300+r.state_bias(t.states[1])-r.state_bias(t.states[0]))
        h.close()
    np.testing.assert_allclose(deltas[0], deltas[1], atol=1e-10)


def test_ncmc_zero_force_distribution():
    sim = ideal_reaction(acid_site(pKa=7., sigma=0., epsilon=0.),
                          ReactionEdge(0, 0, 1, stages=4, relaxations=1), pH=7.3)
    out = sim.run(4500, burnin=100, sample_every=1)
    assert abs(out['state_counts'][:, 1].mean()-1/(1+10**.3)) < .035
    assert len(sim.ncmc_work) > 100
    # Intermediate states must not appear as saved physical configurations.
    assert np.all(out['state_counts'][:, 2:] == 0)
    sim.h.close()


def test_ncmc_rejection_restores_coordinates_and_states():
    templates, edges = prepare_reactions([spce_water(), acid_site(pKa=7)],
                                        [ReactionEdge(1, 0, 1, stages=4, relaxations=4)])
    h = MolecularHamiltonian(templates, np.eye(3)*24, cutoff=8, coulomb='none')
    h.set_molecules([Molecule(0, 0, templates[0].geometry+[5, 5, 5]),
                     Molecule(1, 0, [[12, 12, 12]], mobile=False, exchangeable=False)])
    sim = ReactiveGCMC(h, Reservoirs(300., 120.), reactions=edges)
    before = h.snapshot()
    assert not sim.reaction(edges[0], uid=1)
    for a, b in zip(before['molecules'], h.molecules):
        assert a.uid == b.uid and a.state == b.state
        np.testing.assert_array_equal(a.positions, b.positions)
    assert h.next_uid == before['next_uid']
    assert sim.counters['bridge_relaxation'][0] == 12
    h.close()


def test_checkpoint_exact_rng_and_positions(tmp_path):
    sim = ideal_reaction(acid_site(pKa=7., sigma=0., epsilon=0.), ReactionEdge(0, 0, 1), pH=7.3)
    sim.run(120, sample_every=1)
    path = tmp_path/'checkpoint.json'; sim.save_checkpoint(path)
    a = sim.run(300, sample_every=1)
    final_rng = sim.rng.bit_generator.state
    sim.restore_checkpoint(path)
    b = sim.run(300, sample_every=1)
    np.testing.assert_array_equal(a['state_counts'], b['state_counts'])
    assert sim.rng.bit_generator.state == final_rng
    sim.reservoirs = Reservoirs(300, 6.)
    with pytest.raises(ValueError, match='mismatch'):
        sim.restore_checkpoint(path)
    sim.h.close()


def test_water_hydration_shell_orientation():
    t = spce_water()
    m = Molecule(0, 0, t.geometry+[3, 0, 0])
    out = hydration_shells([[m]], [[0, 0, 0]], cutoff=4)
    assert out['coordination'][0] == 1
    assert len(out['orientation_cosine']) == 1
    assert abs(out['distances_A'][0]-3) < 1e-14
    assert 'raw' in out['normalization']


def test_vertical_gap_and_windows_constant_environment():
    template = redox_site(sigma=0, epsilon=0)
    sim = ideal_reaction(template, ReactionEdge(0, 0, 1, stages=3), state=0)
    assert abs(vertical_gap(sim.h, 0, 0, 1)) < 1e-12
    out = sample_reaction_windows(sim, 0, sim.reactions[0].bridge,
                                  samples_per_window=3, decorrelation_moves=1, burnin_moves=1)
    np.testing.assert_allclose(out['free_energy_kelvin'], 0., atol=1e-10)
    assert sim.h.molecules[0].state == 0
    sim.h.close()


def test_endpoint_mbar_gaussian_and_bootstrap():
    # Harmonic displaced wells: gaps mean_A=ΔF+lambda, mean_B=ΔF-lambda.
    rng = np.random.default_rng(79); T=300.; lam=180.; df=70.
    a = rng.normal(df+lam, np.sqrt(2*lam*T), 10000)
    b = rng.normal(df-lam, np.sqrt(2*lam*T), 10000)
    out = endpoint_free_energy(a, b, T, bootstrap=5, block_length=10)
    assert abs(out['delta_free_energy_kelvin']-df) < 12
    assert abs(out['reorganization_linear_response_kelvin']-lam) < 12
    assert out['overlap_spectral_gap'] > .1
    assert out['kinetic_prefactor'] is None


def test_kinetic_export_no_reservoir_double_count():
    t = redox_site(E0_V=.2, reference_reduction_kelvin=500)
    out = export_kinetic_states(t, [0, 500], reference_label='same-water calibration at zero V')
    assert abs(out['states'][1]['free_energy_kelvin']+VOLT_TO_K*.2) < 1e-10
    assert not out['rate_parameters_supplied']

@pytest.mark.parametrize('slab',[False, True])
@pytest.mark.parametrize('with_electrode',[False, True])
def test_local_charge_hessian_matches_full_energy(slab, with_electrode):
    from poremc.reaction_cache import LocalChargeResponse
    a=MolecularState('a',(.4,-.6,.2),(3,3,0),(50,50,0))
    b=MolecularState('b',(.3,-.9,-.4),(3,3,0),(50,50,0),bound_electrons=1)
    t=MolecularTemplate('site',[[0,0,0],[1.3,0,0],[1.8,.6,0]],[a,b])
    templates,edges=prepare_reactions([spce_water(),t],[ReactionEdge(1,0,1,stages=5)])
    e=electrode(slab=slab) if with_electrode else None
    h=MolecularHamiltonian(templates,np.diag([24,24,48]),cutoff=8,alpha=.3,kmax=3,electrodes=e,slab=slab)
    h.set_molecules([Molecule(0,0,templates[0].geometry+[7,9,20]),Molecule(1,0,templates[1].geometry+[12,12,25])])
    cache=LocalChargeResponse(h,1)
    states=edges[0].bridge
    fast=cache.deltas(states)
    slow=[]
    for s in states:
        m=h.molecules[1].clone();m.state=s
        slow.append(h.propose([1],[m])['delta_energy'])
    np.testing.assert_allclose(fast,slow,atol=2e-8,rtol=1e-11)
    m=h.molecules[0].clone();m.positions+=.01
    h.commit(h.propose([0],[m]))
    with pytest.raises(ValueError,match='stale'):cache.deltas(states)
    h.close()


def test_periodic_molecule_crossing_is_allowed():
    t=spce_water();h=MolecularHamiltonian([t],np.eye(3)*24,cutoff=8)
    h.set_molecules([Molecule(0,0,t.geometry+[23.7,4,4])])
    sim=ReactiveGCMC(h,move_weights=(1,0,0))
    assert sim._inside(h.molecules[0].positions)
    h.close()


def test_reaction_windows_to_current_bridge_no_double_count():
    from poremc.kinetics import KineticEdge
    t = redox_site('cal', E0_V=.1, reference_reduction_kelvin=1000.)
    cal={'states':[0,1], 'free_energy_kelvin':[0.,1000.],
         'mean_electrode_charge_e':[[0.,0.],[.4,.6]], 'overlap_spectral_gap':.2}
    net,meta=network_from_reaction_windows(t,cal,[0,1],[KineticEdge(0,1,k0_per_s=10)],
        temperature=300,reference_label='exact model',electrode_groups=[0,1],
        reference_potentials_V=[.1,.1])
    r=net.rates([.1,.1])
    np.testing.assert_allclose(r[0,0],r[0,1],rtol=1e-13)
    np.testing.assert_allclose(net.q[1],[.4,.6])
    assert meta['supplied_rates_not_inferred_from_MC']


def test_reaction_windows_reject_missing_endpoint():
    from poremc.kinetics import KineticEdge
    with pytest.raises(ValueError,match='occur'):
        network_from_reaction_windows(redox_site('x'),{'states':[0],'free_energy_kelvin':[0.]},[0,1],
            [KineticEdge(0,1)],temperature=300,reference_label='bad')
