"""Finite-state electrochemical reaction networks with a physical-time clock.

A network is a coarse-grained Markov model, not an MC-clock conversion.
Hydration and electrode-polarization free energies enter state tables; rate
prefactors/barriers must be calibrated separately. Currents are anodic-positive.
SSA is exact for the declared constant / piecewise-constant-rate model. Ramped
potential protocols use the master equation, not a frozen-rate approximation.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import math
import numpy as np
from scipy.special import log_expit, logsumexp, roots_hermite
from scipy.integrate import solve_ivp
from .electrode import VOLT_TO_K, ELEMENTARY_CHARGE


@dataclass(frozen=True)
class KineticState:
    name: str
    free_energy_kelvin: float = 0.
    free_energy_dimensionless: float = 0.
    bound_electrons: int = 0
    bound_protons: int = 0

    def __post_init__(self):
        if not self.name or not np.isfinite([self.free_energy_kelvin, self.free_energy_dimensionless]).all():
            raise ValueError('invalid kinetic state')
        if int(self.bound_electrons) != self.bound_electrons or int(self.bound_protons) != self.bound_protons:
            raise ValueError('bound particle counts must be integers')


@dataclass(frozen=True)
class KineticEdge:
    source: int
    target: int
    name: str = 'electron_transfer'
    electrode: int | None = 0
    electrons: int = 1       # positive: consumed from electrode in forward event
    protons: int = 0         # positive: consumed from proton bath in forward event
    model: str = 'butler_volmer'
    k0_per_s: float = 1.
    transfer_coefficient: float = .5
    reorganization_kelvin: float = 3000.
    chemical_work_kelvin: float = 0.

    def __post_init__(self):
        if min(self.source, self.target) < 0 or self.source == self.target:
            raise ValueError('invalid reaction endpoints')
        if self.model not in ('butler_volmer', 'marcus', 'mhc', 'metropolis'):
            raise ValueError('unknown rate law')
        if not np.isfinite([self.k0_per_s, self.transfer_coefficient, self.reorganization_kelvin,
                            self.chemical_work_kelvin]).all() or self.k0_per_s <= 0:
            raise ValueError('finite positive rate prefactor required')
        if not 0 < self.transfer_coefficient < 1 or self.reorganization_kelvin <= 0:
            raise ValueError('invalid transfer coefficient / reorganization energy')
        if int(self.electrons) != self.electrons or int(self.protons) != self.protons:
            raise ValueError('reaction stoichiometry must be integral')
        if self.electrons and self.electrode is None:
            raise ValueError('electron exchange requires an electrode reservoir')


@lru_cache(maxsize=64)
def _mhc_quadrature(nodes):
    x, w = roots_hermite(nodes)
    return x, np.log(w) - .5 * math.log(math.pi)


def paired_log_rates(delta_g_kBT, temperature, edge: KineticEdge):
    """Forward/reverse logs satisfy log(kf/kr)=-Δg to roundoff.

    MHC uses a flat electrode DOS and Fermi occupation, normalized to k0 at
    Δg=0. Marcus k0 is an attempt frequency, NOT the rate at zero driving.
    """
    d = float(delta_g_kBT)
    if not np.isfinite(d) or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('finite driving free energy and positive T required')
    base = math.log(edge.k0_per_s)
    if edge.model == 'butler_volmer':
        return base - edge.transfer_coefficient * d, base + (1 - edge.transfer_coefficient) * d
    if edge.model == 'metropolis':
        return base - max(d, 0), base - max(-d, 0)
    lam = edge.reorganization_kelvin / temperature
    if edge.model == 'marcus':
        return base - (lam + d) ** 2 / (4 * lam), base - (lam - d) ** 2 / (4 * lam)
    # Evaluate the favorable branch. Particle-hole symmetry plus exact local
    # detailed balance avoids exp(+d) overflow in the unfavorable branch.
    x, lw = _mhc_quadrature(160)
    def log_integral(driving):
        return float(logsumexp(lw + log_expit(-(lam + driving + 2 * math.sqrt(lam) * x))))
    favorable = base + log_integral(-abs(d)) - log_integral(0.)
    return (favorable - d, favorable) if d >= 0 else (favorable, favorable + d)


def paired_rates(delta_g_kBT, temperature, edge):
    logs = paired_log_rates(delta_g_kBT, temperature, edge)
    if max(logs) > math.log(np.finfo(float).max) - 2:
        raise OverflowError('rate overflow; revise physical input, do not clip one direction')
    return np.exp(logs)


class ReactionNetwork:
    def __init__(self, states, edges, *, temperature=298.15, electrodes=1,
                 state_electrode_charge_e=None, capacitance_F=None,
                 state_voltage_reference_V=None, area_A2=None):
        self.states = tuple(states)
        self.edges = tuple(edges)
        self.temperature = float(temperature)
        self.n = len(self.states)
        self.m = int(electrodes)
        if self.n < 2 or self.m < 1 or not self.edges or temperature <= 0 or not np.isfinite(temperature):
            raise ValueError('invalid network dimensions or temperature')
        if len({s.name for s in self.states}) != self.n:
            raise ValueError('state names must be unique')
        for e in self.edges:
            if max(e.source, e.target) >= self.n or (e.electrode is not None and not 0 <= e.electrode < self.m):
                raise ValueError('edge index out of range')
            a, b = self.states[e.source], self.states[e.target]
            if b.bound_electrons - a.bound_electrons != e.electrons or b.bound_protons - a.bound_protons != e.protons:
                raise ValueError('edge must balance bound electrons and protons')
        self.q = None if state_electrode_charge_e is None else np.asarray(state_electrode_charge_e, float)
        if self.q is not None and (self.q.shape != (self.n, self.m) or not np.isfinite(self.q).all()):
            raise ValueError('charge table must be (states,electrodes)')
        self.C = np.zeros((self.m, self.m)) if capacitance_F is None else np.asarray(capacitance_F, float)
        if self.C.ndim == 0:
            self.C = np.eye(self.m) * self.C
        elif self.C.shape == (self.m,):
            self.C = np.diag(self.C)
        if self.C.shape != (self.m, self.m) or not np.isfinite(self.C).all() or not np.allclose(self.C, self.C.T, atol=1e-30):
            raise ValueError('capacitance must be finite symmetric matrix')
        if np.linalg.eigvalsh(self.C).min() < -max(1e-30, np.max(np.abs(self.C)) * 1e-10):
            raise ValueError('capacitance must be positive semidefinite')
        self.reference_V = None if state_voltage_reference_V is None else np.broadcast_to(np.asarray(state_voltage_reference_V, float), (self.m,)).copy()
        if self.reference_V is not None and (self.q is None or not np.isfinite(self.reference_V).all()):
            raise ValueError('voltage linearization requires charge table and finite reference')
        self.area_A2 = area_A2
        if area_A2 is not None and (area_A2 <= 0 or not np.isfinite(area_A2)):
            raise ValueError('positive area required')
        self._last_key, self._last_rates = None, None
        self.profile = {'rate_evaluations': 0, 'rate_cache_hits': 0}

    def rates(self, potentials_V, pH=7., proton_reference_V=0.):
        v = np.broadcast_to(np.asarray(potentials_V, float), (self.m,))
        if not np.isfinite(v).all() or not np.isfinite([pH, proton_reference_V]).all():
            raise ValueError('invalid reservoir settings')
        key = (tuple(v), float(pH), float(proton_reference_V))
        if key == self._last_key:
            self.profile['rate_cache_hits'] += 1
            return self._last_rates
        g = np.array([s.free_energy_kelvin / self.temperature + s.free_energy_dimensionless for s in self.states])
        if self.reference_V is not None:
            # dOmega/dPsi=-e*q at fixed composition. This is a LOCAL linear
            # model, not an exact extrapolation of hydration over arbitrary bias.
            g -= VOLT_TO_K / self.temperature * (self.q @ (v - self.reference_V))
        out = []
        for e in self.edges:
            d = (g[e.target] - g[e.source] + e.chemical_work_kelvin / self.temperature
                 + e.protons * math.log(10) * pH
                 - e.protons * VOLT_TO_K * proton_reference_V / self.temperature)
            if e.electrons:
                d += e.electrons * VOLT_TO_K * v[e.electrode] / self.temperature
            out.append(paired_rates(d, self.temperature, e))
        rates = np.asarray(out)
        rates.flags.writeable = False
        self._last_key, self._last_rates = key, rates
        self.profile['rate_evaluations'] += 1
        return rates

    def generator(self, potentials_V, pH=7., proton_reference_V=0.):
        rates = self.rates(potentials_V, pH, proton_reference_V)
        Q = np.zeros((self.n, self.n))
        for e, (kf, kr) in zip(self.edges, rates):
            Q[e.source, e.target] += kf
            Q[e.target, e.source] += kr
        np.fill_diagonal(Q, -Q.sum(axis=1))
        return Q

    def fluxes(self, probabilities, potentials_V, pH=7., proton_reference_V=0.):
        p = np.asarray(probabilities, float)
        if p.shape != (self.n,):
            raise ValueError('probability dimension mismatch')
        rates = self.rates(potentials_V, pH, proton_reference_V)
        return np.array([p[e.source] * r[0] - p[e.target] * r[1] for e, r in zip(self.edges, rates)])

    def faradaic_current(self, probabilities, potentials_V, pH=7., proton_reference_V=0.):
        current = np.zeros(self.m)
        for e, flux in zip(self.edges, self.fluxes(probabilities, potentials_V, pH, proton_reference_V)):
            if e.electrode is not None:
                current[e.electrode] -= ELEMENTARY_CHARGE * e.electrons * flux
        return current

    def steady_state(self, potentials_V, pH=7., proton_reference_V=0., site_count=1.):
        if site_count <= 0 or not np.isfinite(site_count):
            raise ValueError('positive site count required')
        Q = self.generator(potentials_V, pH, proton_reference_V)
        scale = np.max(np.abs(Q))
        if scale == 0:
            raise ValueError('no active rates')
        a = Q.T.copy() / scale
        a[-1] = 1.
        b = np.zeros(self.n)
        b[-1] = 1.
        try:
            p = np.linalg.solve(a, b)
        except np.linalg.LinAlgError as exc:
            raise ValueError('network lacks a unique steady state') from exc
        if p.min() < -1e-10 or np.max(np.abs(Q.T @ p)) > 1e-8 * scale:
            raise ValueError('stationary solution failed')
        p = np.maximum(p, 0)
        p /= p.sum()
        current = self.faradaic_current(p, potentials_V, pH, proton_reference_V) * site_count
        out = {'probability': p, 'edge_flux_per_s_per_site': self.fluxes(p, potentials_V, pH, proton_reference_V),
               'faradaic_current_A': current, 'site_count': site_count,
               'stationarity_residual_per_s': Q.T @ p}
        if self.area_A2:
            out['faradaic_current_density_A_m2'] = current / (self.area_A2 * 1e-20)
        return out

    def ssa(self, segments, *, initial_state=0, seed=123, max_events=1000000):
        """Exact direct SSA on one site, piecewise constant reservoirs.

        Segment keys: duration_s, potentials_V, pH (optional), proton_reference_V.
        Waiting times extending beyond a segment are censored, not counted.
        Results include residence-time averages, not event-count averages.
        """
        if not 0 <= initial_state < self.n or max_events < 1:
            raise ValueError('invalid initial state/event limit')
        rng = np.random.default_rng(seed)
        state, clock = int(initial_state), 0.
        dwell = np.zeros(self.n)
        events = []
        previous_v = None
        for seg in segments:
            duration = float(seg['duration_s'])
            v = np.broadcast_to(np.asarray(seg['potentials_V'], float), (self.m,)).copy()
            if duration <= 0 or not np.isfinite(duration):
                raise ValueError('positive segment duration required')
            pH = float(seg.get('pH', 7.))
            phi = float(seg.get('proton_reference_V', 0.))
            rates = self.rates(v, pH, phi)
            if previous_v is not None and np.any(previous_v != v):
                events.append({'time_s': clock, 'kind': 'voltage_step', 'state_before': state,
                     'state_after': state, 'edge': -1, 'direction': 0,
                     'electrons_to_species': np.zeros(self.m).tolist(),
                     'displacement_charge_C': (self.C @ (v - previous_v)).tolist()})
            end = clock + duration
            while clock < end:
                outgoing = []
                for j, e in enumerate(self.edges):
                    if state == e.source and rates[j, 0] > 0:
                        outgoing.append((j, 1, e.target, rates[j, 0]))
                    elif state == e.target and rates[j, 1] > 0:
                        outgoing.append((j, -1, e.source, rates[j, 1]))
                total = float(sum(x[3] for x in outgoing))
                if total == 0:
                    dwell[state] += end - clock
                    clock = end
                    break
                wait = -math.log(max(rng.random(), np.finfo(float).tiny)) / total
                if clock + wait >= end:
                    dwell[state] += end - clock
                    clock = end
                    break
                if len(events) >= max_events:
                    raise RuntimeError('event limit reached before requested physical time')
                dwell[state] += wait
                clock += wait
                cumulative = np.cumsum([x[3] for x in outgoing])
                k = min(int(np.searchsorted(cumulative, rng.random() * total)), len(outgoing) - 1)
                ei, sign, new, _ = outgoing[k]
                edge = self.edges[ei]
                electrons = np.zeros(self.m)
                if edge.electrode is not None:
                    electrons[edge.electrode] = sign * edge.electrons
                displacement = None if self.q is None else (ELEMENTARY_CHARGE * (self.q[new] - self.q[state])).tolist()
                events.append({'time_s': clock, 'kind': 'reaction', 'state_before': state,
                    'state_after': new, 'edge': ei, 'direction': sign,
                    'electrons_to_species': electrons.tolist(),
                    'protons_to_species': sign * edge.protons,
                    'displacement_charge_C': displacement})
                state = new
            previous_v = v
        if clock <= 0:
            raise ValueError('nonempty protocol required')
        electron_count = sum((np.asarray(x['electrons_to_species']) for x in events), np.zeros(self.m))
        faradaic = -ELEMENTARY_CHARGE * electron_count / clock
        out = {'duration_s': clock, 'final_state': state, 'events': events, 'residence_time_s': dwell,
               'time_weighted_probability': dwell / clock, 'net_electrons_to_species': electron_count,
               'faradaic_current_A': faradaic, 'scope': 'one independent reaction site'}
        if self.q is not None:
            charge = sum((np.asarray(x['displacement_charge_C']) for x in events), np.zeros(self.m))
            out.update({'displacement_current_A': charge / clock, 'total_lead_current_A': faradaic + charge / clock})
        return out

    def ramp(self, times_s, potentials_V, *, samples=501, initial_probability=None,
             pH=7., proton_reference_V=0., site_count=1., rtol=1e-8, atol=1e-11):
        """Master equation for a continuous piecewise-linear potential protocol.

        Capacitive current = e*q_state^T dp/dt + C*dV/dt. The q_state table is
        conditional mean electrode charge. Total lead current = I_F + dQ/dt;
        the Faradaic electron is NOT counted a second time as an extra charge.
        """
        times = np.asarray(times_s, float)
        v = np.asarray(potentials_V, float)
        if v.ndim == 1 and self.m == 1:
            v = v[:, None]
        if times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0) or v.shape != (len(times), self.m) or not np.isfinite(v).all() or not np.isfinite(times).all():
            raise ValueError('invalid potential waveform')
        if samples < 2 or site_count <= 0 or not np.isfinite(site_count):
            raise ValueError('invalid sampling/site count')
        p0 = self.steady_state(v[0], pH, proton_reference_V)['probability'] if initial_probability is None else np.asarray(initial_probability, float)
        if p0.shape != (self.n,) or not np.isfinite(p0).all() or p0.min() < 0 or not np.isclose(p0.sum(), 1):
            raise ValueError('invalid initial probability')
        out_times = np.unique(np.r_[np.linspace(times[0], times[-1], samples), times])
        probs = np.empty((len(out_times), self.n))
        for s in range(len(times) - 1):
            a, b = times[s:s + 2]
            slope = (v[s + 1] - v[s]) / (b - a)
            def rhs(t, p):
                potential = v[s] + slope * (t - a)
                return self.generator(potential, pH, proton_reference_V).T @ p
            sol = solve_ivp(rhs, (a, b), p0, method='BDF', dense_output=True,
                            rtol=rtol, atol=atol, max_step=(b - a) / 20)
            if not sol.success:
                raise RuntimeError(sol.message)
            ids = np.flatnonzero((out_times >= a) & (out_times <= b))
            probs[ids] = sol.sol(out_times[ids]).T
            p0 = sol.y[:, -1]
        potentials = np.column_stack([np.interp(out_times, times, v[:, j]) for j in range(self.m)])
        currents, charging = [], []
        for t, p, E in zip(out_times, probs, potentials):
            s = min(np.searchsorted(times, t, side='right') - 1, len(times) - 2)
            slope = (v[s + 1] - v[s]) / (times[s + 1] - times[s])
            dp = self.generator(E, pH, proton_reference_V).T @ p
            currents.append(self.faradaic_current(p, E, pH, proton_reference_V))
            charging.append((ELEMENTARY_CHARGE * (dp @ self.q) if self.q is not None else np.zeros(self.m)) + self.C @ slope)
        current = np.asarray(currents) * site_count
        charge = np.asarray(charging) * site_count
        result = {'time_s': out_times, 'potentials_V': potentials, 'probability': probs,
                  'faradaic_current_A': current, 'displacement_current_A': charge,
                  'site_count': site_count, 'mass_error': float(np.max(np.abs(probs.sum(axis=1) - 1))),
                  'minimum_probability': float(probs.min()),
                  'charge_model': 'state-dependent charge plus linear capacitance' if self.q is not None else 'linear capacitance only; state-dependent displacement unavailable'}
        if self.q is not None:
            result['total_lead_current_A'] = current + charge
        if self.area_A2:
            result['faradaic_current_density_A_m2'] = current / (self.area_A2 * 1e-20)
        return result


def bin_current(events, edges_s, electrode_count, *, require_displacement=False):
    edges = np.asarray(edges_s, float)
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0) or not np.isfinite(edges).all():
        raise ValueError('strictly increasing time bins required')
    far = np.zeros((len(edges) - 1, electrode_count))
    disp = np.zeros_like(far)
    complete = True
    for event in events:
        t = event['time_s']
        if t < edges[0] or t > edges[-1]:
            continue
        j = min(np.searchsorted(edges, t, side='right') - 1, len(edges) - 2)
        far[j] -= ELEMENTARY_CHARGE * np.asarray(event['electrons_to_species'])
        if event.get('displacement_charge_C') is None:
            complete = False
        else:
            disp[j] += np.asarray(event['displacement_charge_C'])
    if require_displacement and not complete:
        raise ValueError('electrode charge information missing')
    dt = np.diff(edges)[:, None]
    result = {'time_edges_s': edges, 'faradaic_current_A': far / dt,
              'displacement_available': complete}
    if complete:
        result.update({'displacement_current_A': disp / dt, 'total_lead_current_A': (far + disp) / dt})
    return result
