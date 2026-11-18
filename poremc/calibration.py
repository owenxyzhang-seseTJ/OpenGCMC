"""Bridge atomistic endpoint sampling to calibrated reaction-state free energies.

Vertical gap = U_B(X)-U_A(X) with electrode charges minimized for each endpoint.
Use two endpoint ensembles or overlapping free-energy windows. Kinetic parameters
are NOT inferred from equilibrium Monte Carlo step counts.
"""
from __future__ import annotations
import numpy as np
from scipy.special import logsumexp
from .free_energy import MBAR
from .reaction_cache import state_energy_differences


def vertical_gap(hamiltonian, uid, state_a, state_b):
    i = hamiltonian.molecule_index(uid)
    m = hamiltonian.molecules[i]
    a, b = m.clone(), m.clone()
    a.state, b.state = state_a, state_b
    # Differences referenced to the same current configuration avoid drift.
    da = 0. if m.state == state_a else hamiltonian.propose([i], [a])['delta_energy']
    db = 0. if m.state == state_b else hamiltonian.propose([i], [b])['delta_energy']
    return float(db - da)


def endpoint_free_energy(gaps_a_kelvin, gaps_b_kelvin, temperature, *, bootstrap=0,
                         block_length=1, seed=124):
    a, b = np.asarray(gaps_a_kelvin, float), np.asarray(gaps_b_kelvin, float)
    if a.ndim != 1 or b.ndim != 1 or min(len(a), len(b)) < 2 or temperature <= 0 or not np.isfinite(np.r_[a, b]).all():
        raise ValueError('two finite endpoint ensembles and positive T required')
    u = np.vstack([np.zeros(len(a) + len(b)), np.r_[a, b] / temperature])
    fit = MBAR(u, [len(a), len(b)])
    delta = float((fit.f[1] - fit.f[0]) * temperature)
    lam = float((a.mean() - b.mean()) / 2)
    result = {'delta_free_energy_kelvin': delta, 'reorganization_linear_response_kelvin': lam,
              'delta_free_energy_linear_response_kelvin': float((a.mean() + b.mean()) / 2),
              'variance_a_kelvin2': float(a.var()), 'variance_b_kelvin2': float(b.var()),
              'variance_predicted_linear_response_kelvin2': 2 * temperature * lam,
              'overlap': fit.overlap, 'overlap_spectral_gap': fit.spectral_gap,
              'forward_FEP_kelvin': float(-temperature * (logsumexp(-a / temperature) - np.log(len(a)))),
              'reverse_FEP_kelvin': float(temperature * (logsumexp(b / temperature) - np.log(len(b)))),
              'kinetic_prefactor': None,
              'interpretation': 'lambda is a linear-response diagnostic; rate prefactor requires independent calibration'}
    if bootstrap:
        if bootstrap < 2 or block_length < 1 or block_length > min(len(a), len(b)):
            raise ValueError('invalid block bootstrap')
        rng = np.random.default_rng(seed)
        vals = []
        for _ in range(bootstrap):
            resampled = []
            for x in (a, b):
                starts = rng.integers(len(x), size=int(np.ceil(len(x) / block_length)))
                ids = ((starts[:, None] + np.arange(block_length)) % len(x)).ravel()[:len(x)]
                resampled.append(x[ids])
            uu = np.vstack([np.zeros(len(a) + len(b)), np.concatenate(resampled) / temperature])
            f = MBAR(uu, [len(a), len(b)])
            vals.append((f.f[1] - f.f[0]) * temperature)
        result['bootstrap_free_energy_kelvin'] = np.asarray(vals)
        result['delta_free_energy_ci95_kelvin'] = np.percentile(vals, [2.5, 97.5])
    return result


def export_kinetic_states(template, relative_environment_free_energies_kelvin, *, reference_label,
                          relative_error_kelvin=None):
    """Build state records without adding a reservoir potential twice.

    The input free energies are physical conditional-state free energies with the
    same fixed reservoir convention, NOT already pH/E-biased state populations.
    """
    from .kinetics import KineticState
    g = np.asarray(relative_environment_free_energies_kelvin, float)
    if g.shape != (len(template.states),) or not np.isfinite(g).all() or not reference_label:
        raise ValueError('one calibrated environment free energy per endpoint required')
    records = []
    for state, dg in zip(template.states, g):
        if state.name.startswith('_bridge_'):
            raise ValueError('export physical endpoints only, not intermediate states')
        records.append(KineticState(state.name, state.intrinsic_kelvin + float(dg),
                      state.intrinsic_dimensionless - state.log_degeneracy,
                      int(state.bound_electrons), int(state.bound_protons)))
    return {'states': [s.__dict__ for s in records], 'reference': reference_label,
            'environment_free_energy_kelvin': g.tolist(),
            'relative_error_kelvin': relative_error_kelvin,
            'rate_parameters_supplied': False}


def sample_reaction_windows(simulation, uid, bridge_states, *, samples_per_window=100,
                            decorrelation_moves=10, burnin_moves=200):
    """Fixed-composition explicit-solvent reaction-window sampling and MBAR.

    Random-scan rigid displacements sample each frozen Hamiltonian. This routine
    does not perform grand exchange or reaction moves and has no time units.
    All endpoint physical energies are evaluated on every saved configuration.
    """
    if samples_per_window < 2 or decorrelation_moves < 1 or burnin_moves < 0:
        raise ValueError('invalid window sampling lengths')
    h = simulation.h
    states = tuple(int(s) for s in bridge_states)
    if len(states) < 2 or len(set(states)) != len(states):
        raise ValueError('at least two distinct intermediate states required')
    snapshot = h.snapshot()
    all_columns, window_states, electrode_mean = [], [], []
    try:
        for state in states:
            i = h.molecule_index(uid)
            m = h.molecules[i].clone()
            m.state = state
            h.commit(h.propose([i], [m]))
            for _ in range(burnin_moves):
                simulation.displace()
            qtrace = []
            for _ in range(samples_per_window):
                for _ in range(decorrelation_moves):
                    simulation.displace()
                i = h.molecule_index(uid)
                m = h.molecules[i]
                # One local charge Hessian reuses water/Ewald/electrode response
                # for all frozen intermediate states at this configuration.
                col = state_energy_differences(h, uid, states) / simulation.temperature
                all_columns.append(col)
                window_states.append(state)
                if h.atomic.electrodes:
                    qtrace.append(h.atomic.electrode_state['q'].copy())
            if qtrace:
                electrode_mean.append(np.mean(qtrace, axis=0))
        u = np.asarray(all_columns).T
        n = np.full(len(states), samples_per_window, int)
        fit = MBAR(u, n)
        return {'states': np.asarray(states), 'u_kn': u, 'n_k': n,
                'free_energy_kelvin': fit.f * simulation.temperature,
                'overlap': fit.overlap, 'overlap_spectral_gap': fit.spectral_gap,
                'mean_electrode_charge_e': np.asarray(electrode_mean),
                'samples_per_window': samples_per_window,
                'equilibration_moves_per_window': burnin_moves,
                'moves_between_samples': decorrelation_moves,
                'ensemble': 'fixed solvent/ion composition; intermediate state fixed per window'}
    finally:
        h.restore(snapshot)


def network_from_reaction_windows(template, calibration, physical_states, edges, *, temperature,
                                  reference_label, electrode_groups=None,
                                  reference_potentials_V=None, capacitance_F=None):
    """Construct an explicitly calibrated coarse-grained kinetic network.

    Calibration contains ONLY conditional physical free energies from fixed-state
    windows. Intrinsic pKa/E0 offsets are added once; proton/electron reservoir
    terms are subsequently supplied to ReactionNetwork. User-supplied edges must
    include calibrated rate constants/barriers; an MC acceptance rate is not one.
    `physical_states` identifies template indices and sets the network ordering.
    Electrode groups assign each Gaussian site to an external lead. Conditional
    window mean charges are retained; a voltage change uses a labelled local
    linearization when reference_potentials_V is provided.
    """
    from .hydration import MolecularTemplate
    from .kinetics import KineticState, KineticEdge, ReactionNetwork
    physical = tuple(int(i) for i in physical_states)
    indices = tuple(int(i) for i in calibration['states'])
    if len(physical) < 2 or len(set(physical)) != len(physical):
        raise ValueError('distinct physical endpoint states required')
    if any(i not in indices or not 0 <= i < len(template.states) for i in physical):
        raise ValueError('physical endpoints must occur in calibration and template')
    endpoint_template = MolecularTemplate(template.name, template.geometry,
                                         tuple(template.states[i] for i in physical))
    rows = np.asarray([indices.index(i) for i in physical], int)
    g = np.asarray(calibration['free_energy_kelvin'], float)[rows]
    g -= g[0]
    exported = export_kinetic_states(endpoint_template, g, reference_label=reference_label)
    kinetic_states = [KineticState(**s) for s in exported['states']]
    kinetic_edges = [KineticEdge(**e) if isinstance(e, dict) else e for e in edges]
    if not kinetic_edges:
        raise ValueError('explicit kinetic edges with physical rates required')
    q = None
    nleads = max([0] + [e.electrode for e in kinetic_edges if e.electrode is not None])+1
    if electrode_groups is not None:
        group = np.asarray(electrode_groups)
        raw = np.asarray(calibration['mean_electrode_charge_e'], float)
        if group.ndim != 1 or raw.ndim != 2 or raw.shape != (len(indices), len(group)):
            raise ValueError('electrode grouping must match the calibration charge table')
        if not np.issubdtype(group.dtype, np.integer) or group.min() < 0:
            raise ValueError('electrode groups must be nonnegative integers')
        nleads = max(nleads, int(group.max())+1)
        q = np.column_stack([raw[rows][:, group == k].sum(axis=1) for k in range(nleads)])
    elif reference_potentials_V is not None:
        raise ValueError('voltage interpolation requires electrode charge grouping')
    network = ReactionNetwork(kinetic_states, kinetic_edges, temperature=temperature,
                              electrodes=nleads, state_electrode_charge_e=q,
                              capacitance_F=capacitance_F,
                              state_voltage_reference_V=reference_potentials_V)
    metadata = dict(exported, physical_template_states=list(physical),
                    overlap_spectral_gap=float(calibration.get('overlap_spectral_gap', float('nan'))),
                    kinetic_edges=[e.__dict__ for e in kinetic_edges],
                    supplied_rates_not_inferred_from_MC=True,
                    voltage_extrapolation=('local linear response' if reference_potentials_V is not None else 'frozen environmental free energies'))
    return network, metadata
