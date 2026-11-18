import math
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import expit
from poremc.kinetics import *
from poremc.electrode import ELEMENTARY_CHARGE, VOLT_TO_K


def two_contacts(model='butler_volmer', q=False):
    states = [KineticState('O'), KineticState('R', bound_electrons=1)]
    edges = [KineticEdge(0, 1, electrode=j, model=model, k0_per_s=20.) for j in range(2)]
    return ReactionNetwork(states, edges, electrodes=2, temperature=300.,
             state_electrode_charge_e=[[0, 0], [.5, .5]] if q else None)


@pytest.mark.parametrize('model', ['butler_volmer', 'marcus', 'mhc', 'metropolis'])
@pytest.mark.parametrize('dg', [-20., -2., 0., 3., 20.])
def test_local_detailed_balance(model, dg):
    edge = KineticEdge(0, 1, model=model, k0_per_s=12., transfer_coefficient=.3)
    a, b = paired_log_rates(dg, 300., edge)
    assert abs((a-b)+dg) < 1e-12


@pytest.mark.parametrize('dg', [-5., 0., 5.])
def test_mhc_vs_independent_quadrature(dg):
    lam = 10.
    e = KineticEdge(0, 1, model='mhc', k0_per_s=1., reorganization_kelvin=lam*300)
    def integral(d):
        return quad(lambda x: expit(-x)*math.exp(-(lam+d-x)**2/(4*lam)),
                    -100, 100, epsabs=1e-12)[0]
    expected = integral(dg)/integral(0)
    got = paired_rates(dg, 300, e)[0]
    np.testing.assert_allclose(got, expected, rtol=2e-4)


@pytest.mark.parametrize('model', ['butler_volmer', 'marcus', 'mhc'])
def test_equal_reservoirs_have_zero_current(model):
    n = two_contacts(model)
    r = n.steady_state([.03, .03])
    np.testing.assert_allclose(r['faradaic_current_A'], 0., atol=1e-28)
    expected = 1/(1+math.exp(VOLT_TO_K*.03/300))
    assert abs(r['probability'][1]-expected) < 1e-13


def test_two_reservoir_analytic_current_and_conservation():
    n = two_contacts()
    v = [-.05, .05]
    rates = n.rates(v)
    a, b = rates[0]; c, d = rates[1]
    flux = (a*d-b*c)/(a+b+c+d)
    out = n.steady_state(v)
    np.testing.assert_allclose(out['edge_flux_per_s_per_site'], [flux, -flux], rtol=1e-12)
    np.testing.assert_allclose(out['faradaic_current_A'], [-ELEMENTARY_CHARGE*flux, ELEMENTARY_CHARGE*flux], rtol=1e-12)
    assert abs(out['faradaic_current_A'].sum()) < 1e-29


def test_ssa_time_weighting_and_exact_horizon():
    n = two_contacts()
    r = n.ssa([{'duration_s':200., 'potentials_V':[-.05,.05]}], seed=30)
    target = n.steady_state([-.05,.05])
    assert r['duration_s'] == 200.
    assert abs(r['residence_time_s'].sum()-200) < 1e-8
    np.testing.assert_allclose(r['time_weighted_probability'], target['probability'], atol=.02)
    np.testing.assert_allclose(r['faradaic_current_A'], target['faradaic_current_A'], rtol=.06)
    assert all(e['time_s'] < 200 for e in r['events'])
    # Electron conservation telescopes exactly regardless of random waiting times.
    assert r['net_electrons_to_species'].sum() == r['final_state']


def test_single_electrode_no_unbounded_net_electrons():
    states = [KineticState('O'), KineticState('R', bound_electrons=1)]
    n = ReactionNetwork(states, [KineticEdge(0,1,k0_per_s=5)], temperature=300.)
    r = n.ssa([{'duration_s':100., 'potentials_V':[0.]}], seed=24)
    assert r['net_electrons_to_species'][0] == r['final_state']
    assert abs(r['faradaic_current_A'][0]) <= ELEMENTARY_CHARGE/100


def test_capacitive_faradaic_bookkeeping_and_bins():
    n = two_contacts(q=True)
    r = n.ssa([{'duration_s':5., 'potentials_V':[-.05,.05]}], seed=6)
    b = bin_current(r['events'], np.linspace(0, 5, 51), 2, require_displacement=True)
    np.testing.assert_allclose(np.sum(b['faradaic_current_A']*.1, axis=0), -ELEMENTARY_CHARGE*r['net_electrons_to_species'], atol=1e-30)
    # Sum of total external-lead charge vanishes for closed redox with compensating electrodes.
    assert abs(np.sum(b['total_lead_current_A']*.1)) < 1e-30


def test_missing_displacement_not_silently_zero():
    n = two_contacts(q=False)
    r = n.ssa([{'duration_s':1., 'potentials_V':[-.05,.05]}])
    with pytest.raises(ValueError, match='missing'):
        bin_current(r['events'], [0,1], 2, require_displacement=True)
    assert 'total_lead_current_A' not in r


def test_ssa_voltage_step_capacitance_impulse():
    states=[KineticState('a'), KineticState('b')]
    n=ReactionNetwork(states,[KineticEdge(0,1,electrons=0,electrode=None)],
                      state_electrode_charge_e=[[0],[0]], capacitance_F=2e-18)
    r=n.ssa([{'duration_s':.1,'potentials_V':[0.]},{'duration_s':.1,'potentials_V':[.2]}])
    steps=[e for e in r['events'] if e['kind']=='voltage_step']
    assert len(steps)==1
    assert abs(steps[0]['displacement_charge_C'][0]-4e-19)<1e-30


def test_continuous_ramp_master_equation_and_capacitance():
    states=[KineticState('a'), KineticState('b')]
    n=ReactionNetwork(states,[KineticEdge(0,1,electrons=0,electrode=None)],
                      state_electrode_charge_e=[[0],[0]], capacitance_F=2e-18)
    r=n.ramp([0,1,2],[[0],[.2],[0]],samples=51)
    assert r['mass_error']<1e-10
    assert r['minimum_probability']>=0
    np.testing.assert_allclose(r['faradaic_current_A'],0,atol=1e-30)
    np.testing.assert_allclose(np.abs(r['displacement_current_A']),4e-19,rtol=1e-12)


def test_cv_probability_conservation():
    n=two_contacts(q=True)
    r=n.ramp([0,.2,.4],[[-.1,0],[.1,0],[-.1,0]],samples=71)
    assert r['mass_error']<1e-8
    assert r['minimum_probability']>-1e-9
    assert r['total_lead_current_A'].shape==(len(r['time_s']),2)


def test_pH_equilibrium_kinetic_ensemble():
    states=[KineticState('A'), KineticState('HA',free_energy_dimensionless=-math.log(10)*7,bound_protons=1)]
    n=ReactionNetwork(states,[KineticEdge(0,1,electrode=None,electrons=0,protons=1)],temperature=300.)
    r=n.steady_state([0],pH=8.)
    assert abs(r['probability'][1]-1/11)<1e-14


def test_pcet_nernst_pH_slope():
    states=[KineticState('O'), KineticState('RH',bound_electrons=1,bound_protons=1)]
    n=ReactionNetwork(states,[KineticEdge(0,1,electrons=1,protons=1)],temperature=300.)
    E=-math.log(10)*300/VOLT_TO_K*7
    r=n.steady_state([E],pH=7.)
    assert abs(r['probability'][1]-.5)<1e-12


def test_charge_table_gauge_compensation():
    # +electron on species is balanced by +1 total electrode charge. Under a
    # global voltage shift, -q.dV and electron bath work cancel exactly.
    states=[KineticState('O'),KineticState('R',bound_electrons=1)]
    n=ReactionNetwork(states,[KineticEdge(0,1,electrode=0)],electrodes=2,
        state_electrode_charge_e=[[0,0],[.4,.6]],state_voltage_reference_V=[0,0])
    a=n.rates([0,0]); b=n.rates([.2,.2])
    np.testing.assert_allclose(a,b,rtol=1e-12)


def test_rate_cache_and_invalid_stoichiometry():
    n=two_contacts(); a=n.rates([0,0]); b=n.rates([0,0])
    assert a is b and n.profile['rate_cache_hits']==1
    with pytest.raises(ValueError,match='balance'):
        ReactionNetwork([KineticState('a'),KineticState('b')],[KineticEdge(0,1)])


def test_invalid_rate_overflow_not_clipped():
    e=KineticEdge(0,1)
    with pytest.raises(OverflowError):paired_rates(-1e5,300,e)


def test_nonunique_steady_state_rejected():
    states=[KineticState('a'),KineticState('b'),KineticState('isolated')]
    n=ReactionNetwork(states,[KineticEdge(0,1,electrode=None,electrons=0)])
    with pytest.raises(ValueError,match='unique'):
        n.steady_state([0])
