"""Explicit-solvent semigrand / grand-canonical Monte Carlo at constant pH/E.

The MC chain has NO physical clock. Faradaic kinetics is in kinetics.py.
Reaction endpoints have fixed dummy-atom topology. Standard-state free-energy
calibration must use the same water, electrostatics and compensation convention.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
import json
import math
import time
import numpy as np
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation
from .hydration import MolecularState, MolecularTemplate, Molecule, rotated_molecule
from .electrolyte import FractionalRegion, log_exchange_ratio
from .electrode import VOLT_TO_K
from .cache import content_key


@dataclass(frozen=True)
class Reservoirs:
    temperature: float = 298.15
    pH: float = 7.
    electron_potential_V: float = 0.
    proton_reference_V: float = 0.

    def __post_init__(self):
        if not np.isfinite(list(self.__dict__.values())).all() or self.temperature <= 0:
            raise ValueError('finite reservoirs and positive temperature required')

    def state_bias(self, state: MolecularState) -> float:
        return ((state.intrinsic_kelvin + VOLT_TO_K * (
                 state.bound_electrons * self.electron_potential_V -
                 state.bound_protons * self.proton_reference_V)) / self.temperature
                + state.intrinsic_dimensionless - state.log_degeneracy
                + state.bound_protons * math.log(10) * self.pH)


def acid_site(name='acid', pKa=7., *, sigma=3.5, epsilon=50.,
              reference_deprotonation_kelvin=0.) -> MolecularTemplate:
    """A^- <-> HA reference thermodynamic cycle, one coarse titration center.

    Δg_deprot=(ΔU-ΔG_ref)/kBT + ln(10)(pKa-pH) + e*phi_H/kBT.
    For atomistic sites supply a multisite template including dummy proton sites.
    """
    a = MolecularState('A-', (-1.,), (sigma,), (epsilon,))
    ha = MolecularState('HA', (0.,), (sigma,), (epsilon,), bound_protons=1,
                        intrinsic_kelvin=reference_deprotonation_kelvin,
                        intrinsic_dimensionless=-math.log(10) * pKa)
    return MolecularTemplate(name, np.zeros((1, 3)), [a, ha])


def redox_site(name='redox', E0_V=0., *, oxidized_charge=1., sigma=3.5,
               epsilon=50., reference_reduction_kelvin=0.) -> MolecularTemplate:
    ox = MolecularState('Ox', (oxidized_charge,), (sigma,), (epsilon,))
    red = MolecularState('Red', (oxidized_charge - 1.,), (sigma,), (epsilon,),
                         bound_electrons=1,
                         intrinsic_kelvin=-VOLT_TO_K * E0_V - reference_reduction_kelvin)
    return MolecularTemplate(name, np.zeros((1, 3)), [ox, red])


@dataclass(frozen=True)
class ReactionEdge:
    template: int
    state_a: int
    state_b: int
    weight: float = 1.
    stages: int = 1
    relaxations: int = 0
    bridge: tuple[int, ...] = ()
    name: str = 'reaction'

    def __post_init__(self):
        if self.template < 0 or min(self.state_a, self.state_b) < 0 or self.state_a == self.state_b:
            raise ValueError('invalid reaction endpoints')
        if self.stages < 1 or self.relaxations < 0 or not np.isfinite(self.weight) or self.weight <= 0:
            raise ValueError('invalid reaction protocol')


def prepare_reactions(templates, reactions):
    """Pre-register alchemical states; no mutable force-field tables in hot loops.

    Each edge is selected with a configuration-independent probability. At a
    fixed intermediate lambda, random-scan rigid displacement is reversible.
    There is no relaxation at lambda=0 or 1 inside the switching protocol.
    """
    templates = [MolecularTemplate(t.name, t.geometry, t.states, t.explicit_water, t.citation)
                 for t in templates]
    edges = []
    for ei, edge in enumerate(reactions):
        if edge.template >= len(templates):
            raise ValueError('reaction template out of range')
        t = templates[edge.template]
        if max(edge.state_a, edge.state_b) >= len(t.states):
            raise ValueError('reaction endpoint out of range')
        a, b = t.states[edge.state_a], t.states[edge.state_b]
        dq = b.charge - a.charge
        dh, de = b.bound_protons - a.bound_protons, b.bound_electrons - a.bound_electrons
        if not np.isclose(dq, dh - de, atol=1e-10, rtol=0):
            raise ValueError('reaction charge change must equal Δbound_protons-Δbound_electrons')
        if not np.isclose(dh, round(dh)) or not np.isclose(de, round(de)):
            raise ValueError('physical endpoint stoichiometry must be integral')
        path = [edge.state_a]
        states = list(t.states)
        for k in range(1, edge.stages):
            lam = k / edge.stages
            mix = lambda x, y: np.asarray(x) + lam * (np.asarray(y) - np.asarray(x))
            s = MolecularState(f'_bridge_{ei}_{k}', tuple(mix(a.charges, b.charges)),
                   tuple(mix(a.sigma, b.sigma)), tuple(mix(a.epsilon, b.epsilon)),
                   float(mix(a.bound_protons, b.bound_protons)),
                   float(mix(a.bound_electrons, b.bound_electrons)),
                   float(mix(a.intrinsic_kelvin, b.intrinsic_kelvin)),
                   float(mix(a.intrinsic_dimensionless, b.intrinsic_dimensionless)),
                   float(mix(a.log_degeneracy, b.log_degeneracy)))
            path.append(len(states))
            states.append(s)
        path.append(edge.state_b)
        templates[edge.template] = MolecularTemplate(t.name, t.geometry, states, t.explicit_water, t.citation)
        edges.append(replace(edge, bridge=tuple(path)))
    return templates, edges


@dataclass(frozen=True)
class MolecularExchange:
    stoichiometry: tuple[int, ...]
    weight: float = 1.
    insert_probability: float = .5
    trials: int = 1
    name: str = 'exchange'

    def __post_init__(self):
        nu = np.asarray(self.stoichiometry)
        if nu.ndim != 1 or np.any(nu < 0) or not np.array_equal(nu, nu.astype(int)) or not nu.sum():
            raise ValueError('nonnegative integral stoichiometry required')
        if self.trials < 1 or not 0 < self.insert_probability < 1 or self.weight <= 0 or not np.isfinite(self.weight):
            raise ValueError('invalid exchange probabilities')
        if self.trials > 1 and nu.sum() != 1:
            raise ValueError('multiple-try exchange currently supports a single molecule')


class ReactiveGCMC:
    def __init__(self, hamiltonian, reservoirs=None, *, region=None, log_activities=None,
                 exchanges=(), reactions=(), seed=123, move_weights=(.5, .25, .25),
                 translation_A=.3, rotation_rad=.35):
        self.h = hamiltonian
        self.reservoirs = reservoirs or Reservoirs()
        self.region = region or FractionalRegion(self.h.cell)
        if not np.allclose(self.region.cell, self.h.cell):
            raise ValueError('region/cell mismatch')
        if self.h.atomic.slab and self.region.periodic[2]:
            raise ValueError('slab electrostatics require a nonperiodic z region; set region.lower and region.upper')
        self.log_activities = np.zeros(len(self.h.templates)) if log_activities is None else np.asarray(log_activities, float)
        if self.log_activities.shape != (len(self.h.templates),) or not np.isfinite(self.log_activities).all():
            raise ValueError('one finite log activity per molecular template required')
        self.exchanges, self.reactions = tuple(exchanges), tuple(reactions)
        self.weights = np.asarray(move_weights, float)
        if self.weights.shape != (3,) or np.any(self.weights < 0) or not np.isfinite(self.weights).all():
            raise ValueError('three nonnegative move weights required')
        if not self.exchanges:
            self.weights[1] = 0.
        if not self.reactions:
            self.weights[2] = 0.
        if self.weights.sum() <= 0:
            raise ValueError('at least one active move required')
        self.weights /= self.weights.sum()
        self.translation_A, self.rotation_rad = float(translation_A), float(rotation_rad)
        if min(translation_A, rotation_rad) <= 0 or not np.isfinite([translation_A, rotation_rad]).all():
            raise ValueError('positive displacement sizes required')
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.counters = {key: [0, 0] for key in ('displacement', 'exchange', 'reaction', 'bridge_relaxation')}
        self.last_work = None
        self.ncmc_work = []
        for c in self.exchanges:
            if len(c.stoichiometry) != len(self.h.templates):
                raise ValueError('channel dimension mismatch')
            for i in np.flatnonzero(c.stoichiometry):
                if len(self.h.templates[i].states) != 1:
                    raise ValueError('exchange only single-state species; reactive sites have fixed count')
            dq = sum(n * t.states[0].charge for n, t in zip(c.stoichiometry, self.h.templates))
            if abs(dq) > 1e-9 and self.h.atomic.ewald and self.h.atomic.electrodes is None:
                raise ValueError('charged exchange requires explicit compensating electrodes')
        for e in self.reactions:
            if not e.bridge:
                raise ValueError('call prepare_reactions before building the Hamiltonian')
            t = self.h.templates[e.template]
            dq = t.states[e.state_b].charge - t.states[e.state_a].charge
            if abs(dq) > 1e-9 and self.h.atomic.ewald and self.h.atomic.electrodes is None:
                raise ValueError('charge-changing titration/redox needs electrodes or a neutral coupled model')
        for m in self.h.molecules:
            if not self._inside(m.positions):
                raise ValueError('initial molecule outside accessible region')
        self.check_neutrality()

    @property
    def temperature(self):
        return self.reservoirs.temperature

    def check_neutrality(self):
        h = self.h.atomic
        q = h.total_charge + (h.electrode_state['q'].sum() if h.electrodes else 0.)
        if h.ewald is not None and abs(q) > 1e-8:
            raise ValueError(f'non-neutral full electrostatic system: {q}')
        return float(q)

    def _inside(self, positions):
        # Whole molecules stay unwrapped: only genuinely restricted axes have
        # walls. Rejecting atoms crossing a periodic face biases water activity.
        f = np.asarray(positions).reshape(-1, 3) @ self.region.inverse
        axes = ~self.region.periodic
        return bool(np.all(f[:, axes] >= self.region.lower[axes] - 1e-12) and
                    np.all(f[:, axes] < self.region.upper[axes] + 1e-12))

    def _state_bias(self, mol):
        return self.reservoirs.state_bias(self.h.templates[mol.template].states[mol.state])

    def _accept(self, logratio):
        if np.isnan(logratio):
            raise FloatingPointError('NaN acceptance ratio')
        return math.log(max(self.rng.random(), np.finfo(float).tiny)) < min(0., logratio)

    def displace(self, *, bridge=False):
        key = 'bridge_relaxation' if bridge else 'displacement'
        self.counters[key][0] += 1
        movable = [i for i, m in enumerate(self.h.molecules) if m.mobile]
        if not movable:
            return False
        i = int(self.rng.choice(movable))
        old = self.h.molecules[i]
        new = old.clone()
        if len(new.positions) > 1 and self.rng.random() < .5:
            axis = self.rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            rot = Rotation.from_rotvec(axis * self.rng.uniform(-self.rotation_rad, self.rotation_rad)).as_matrix()
            new.positions = (new.positions - new.positions[0]) @ rot.T + new.positions[0]
        else:
            new.positions += self.rng.uniform(-self.translation_A, self.translation_A, 3)
        new.positions += self.region.wrap(new.positions[0]) - new.positions[0]
        if not self._inside(new.positions):
            return False
        trial = self.h.propose([i], [new])
        ok = self._accept(-trial['delta_energy'] / self.temperature)
        if ok:
            self.h.commit(trial)
            self.counters[key][1] += 1
        return ok

    def _random_molecule(self, template):
        xyz = rotated_molecule(self.h.templates[template], self.region.sample(self.rng)[0], self.rng)
        return Molecule(template, 0, xyz)

    def _weighted_choice(self, items):
        w = np.array([x.weight for x in items], float)
        return items[int(self.rng.choice(len(items), p=w / w.sum()))]

    def exchange(self, channel=None):
        self.counters['exchange'][0] += 1
        if not self.exchanges and channel is None:
            return False
        c = channel or self._weighted_choice(self.exchanges)
        if c.trials > 1:
            return self._multiple_try_exchange(c)
        nu = np.asarray(c.stoichiometry, int)
        groups = [[i for i, m in enumerate(self.h.molecules) if m.template == j and m.exchangeable]
                  for j in range(len(nu))]
        n = np.array([len(g) for g in groups])
        insert = self.rng.random() < c.insert_probability
        remove, add = [], []
        if insert:
            for j in range(len(nu)):
                add.extend(self._random_molecule(j) for _ in range(nu[j]))
            if any(not self._inside(m.positions) for m in add):
                return False
        else:
            if np.any(n < nu):
                return False
            for g, count in zip(groups, nu):
                if count:
                    remove.extend(self.rng.choice(g, size=count, replace=False).tolist())
        trial = self.h.propose(remove, add)
        dg = sum(self._state_bias(m) for m in add) - sum(self._state_bias(self.h.molecules[i]) for i in remove)
        lr = log_exchange_ratio(n, nu, self.log_activities,
                                [-math.log(self.region.volume)] * nu.sum(), trial['delta_energy'],
                                self.temperature, insert, c.insert_probability, dg)
        ok = self._accept(lr)
        if ok:
            self.h.commit(trial)
            self.counters['exchange'][1] += 1
        return ok

    def _multiple_try_exchange(self, c):
        tid = int(np.flatnonzero(c.stoichiometry)[0])
        group = [i for i, m in enumerate(self.h.molecules) if m.template == tid and m.exchangeable]
        n, K = len(group), c.trials
        insert = self.rng.random() < c.insert_probability
        if not insert and not n:
            return False
        remove = [] if insert else [int(self.rng.choice(group))]
        deletion = None if insert else self.h.propose(remove, [])
        base_delta = 0. if insert else deletion['delta_energy']
        weights, trials = [], []
        if not insert:
            old = self.h.molecules[remove[0]]
            weights.append(base_delta / self.temperature - self._state_bias(old))
        for _ in range(K if insert else K - 1):
            mol = self._random_molecule(tid)
            if not self._inside(mol.positions):
                weights.append(-np.inf)
                trials.append(None)
                continue
            tr = self.h.propose(remove, [mol])
            weights.append(-(tr['delta_energy'] - base_delta) / self.temperature - self._state_bias(mol))
            trials.append(tr)
        logW = float(logsumexp(weights) - math.log(K))
        if insert:
            if not np.isfinite(logW):
                return False
            choice = int(self.rng.choice(K, p=np.exp(np.asarray(weights) - logsumexp(weights))))
            selected = trials[choice]
            lr = (self.log_activities[tid] + math.log(self.region.volume / (n + 1)) + logW
                  + math.log((1 - c.insert_probability) / c.insert_probability))
        else:
            selected = deletion
            lr = (-self.log_activities[tid] + math.log(n / self.region.volume) - logW
                  + math.log(c.insert_probability / (1 - c.insert_probability)))
        ok = self._accept(lr)
        if ok:
            self.h.commit(selected)
            self.counters['exchange'][1] += 1
        return ok

    def reaction(self, edge=None, uid=None):
        self.counters['reaction'][0] += 1
        if not self.reactions and edge is None:
            return False
        e = edge or self._weighted_choice(self.reactions)
        group = [m for m in self.h.molecules if m.template == e.template]
        if not group:
            return False
        old = group[int(self.rng.integers(len(group)))] if uid is None else self.h.molecules[self.h.molecule_index(uid)]
        if old.template != e.template or old.state not in (e.state_a, e.state_b):
            return False
        uid = old.uid
        path = e.bridge if old.state == e.state_a else tuple(reversed(e.bridge))
        snapshot = self.h.snapshot() if len(path) > 2 else None
        work = 0.
        for k, state in enumerate(path[1:]):
            i = self.h.molecule_index(uid)
            old = self.h.molecules[i]
            new = old.clone()
            new.state = state
            tr = self.h.propose([i], [new])
            work += tr['delta_energy'] / self.temperature + self._state_bias(new) - self._state_bias(old)
            if not np.isfinite(work):
                if snapshot is not None:
                    self.h.restore(snapshot)
                self.last_work = np.inf
                return False
            if snapshot is None:
                ok = self._accept(-work)
                if ok:
                    self.h.commit(tr)
                    self.counters['reaction'][1] += 1
                self.last_work = float(work)
                return ok
            self.h.commit(tr)
            if k < len(path) - 2:
                for _ in range(e.relaxations):
                    self.displace(bridge=True)
        ok = self._accept(-work)
        if not ok:
            self.h.restore(snapshot)
        else:
            self.counters['reaction'][1] += 1
        self.last_work = float(work)
        self.ncmc_work.append([float(work), bool(ok)])
        return ok

    def step(self):
        k = int(self.rng.choice(3, p=self.weights))
        result = (self.displace, self.exchange, self.reaction)[k]()
        self.steps += 1
        return result

    def run(self, steps=10000, burnin=0, sample_every=10, record_positions=False):
        if steps < 1 or burnin < 0 or sample_every < 1:
            raise ValueError('invalid run length')
        for _ in range(int(burnin)):
            self.step()
        counts, energies, charges, states, frames = [], [], [], [], []
        t0 = time.perf_counter()
        for i in range(int(steps)):
            self.step()
            if (i + 1) % sample_every == 0:
                counts.append([sum(m.template == j for m in self.h.molecules) for j in range(len(self.h.templates))])
                energies.append(self.h.energy)
                states.append(np.concatenate(self.h.state_counts()))
                if self.h.atomic.electrodes:
                    charges.append(self.h.atomic.electrode_state['q'].copy())
                if record_positions:
                    frames.append([m.clone() for m in self.h.molecules])
        elapsed = time.perf_counter() - t0
        return {'counts': np.asarray(counts), 'energy_kelvin': np.asarray(energies),
                'state_counts': np.asarray(states), 'electrode_charges': np.asarray(charges),
                'frames': frames, 'wall_seconds': elapsed, 'attempts': int(steps),
                'moves_per_second': steps / elapsed, 'counters': {k: v.copy() for k, v in self.counters.items()},
                'neutrality_error_e': self.check_neutrality(),
                'energy_recompute_error_K': self.h.energy - self.h.full_energy(),
                'physical_time': None,
                'time_meaning': 'equilibrium MC attempts, not seconds'}

    def fingerprint(self):
        return content_key({'hamiltonian': self.h.fingerprint(), 'reservoirs': self.reservoirs.__dict__,
                     'exchanges': [e.__dict__ for e in self.exchanges], 'reactions': [e.__dict__ for e in self.reactions],
                     'translation': self.translation_A, 'rotation': self.rotation_rad},
                     {'weights': self.weights, 'log_activities': self.log_activities,
                      'region_lower': self.region.lower, 'region_upper': self.region.upper})

    def save_checkpoint(self, path):
        def encode(m):
            return {**m.__dict__, 'positions': m.positions.tolist()}
        payload = {'schema': 'poremc-reactive-04', 'fingerprint': self.fingerprint(),
                   'molecules': [encode(m) for m in self.h.molecules], 'next_uid': self.h.next_uid,
                   'rng': self.rng.bit_generator.state, 'steps': self.steps,
                   'counters': self.counters, 'ncmc_work': self.ncmc_work}
        Path(path).write_text(json.dumps(payload, indent=2), encoding='utf8')

    def restore_checkpoint(self, path):
        data = json.loads(Path(path).read_text(encoding='utf8'))
        if data.get('schema') != 'poremc-reactive-04' or data['fingerprint'] != self.fingerprint():
            raise ValueError('checkpoint model/reservoir/proposal mismatch')
        self.h.set_molecules([Molecule(**m) for m in data['molecules']])
        self.h.next_uid = data['next_uid']
        self.rng.bit_generator.state = data['rng']
        self.steps, self.counters = data['steps'], data['counters']
        self.ncmc_work = data.get('ncmc_work', [])
        self.check_neutrality()
