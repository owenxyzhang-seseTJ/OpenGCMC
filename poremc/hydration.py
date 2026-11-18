"""Rigid explicit solvent and reactive multisite molecules on the 0.3 kernels.

Lengths: Angstrom; energies: U/k_B in kelvin; charges: elementary charge.
Rigid primary-cell intramolecular LJ and bare Coulomb pairs are excluded.
Periodic-image interactions remain in Ewald. Coordinates stay molecule-unwrapped.
SPC/E is an effective fixed-charge water model, not a bond-breaking water model.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence
import copy
import math
import numpy as np
from scipy.spatial.transform import Rotation
from .electrolyte import Ion, IonicHamiltonian, FractionalRegion
from .units import COULOMB_KA, KJ_MOL_TO_K
from .cache import content_key


@dataclass(frozen=True)
class MolecularState:
    name: str
    charges: tuple[float, ...]
    sigma: tuple[float, ...]
    epsilon: tuple[float, ...]
    bound_protons: float = 0.0
    bound_electrons: float = 0.0
    intrinsic_kelvin: float = 0.0
    intrinsic_dimensionless: float = 0.0
    log_degeneracy: float = 0.0

    def __post_init__(self):
        for name in ('charges', 'sigma', 'epsilon'):
            object.__setattr__(self, name, tuple(float(x) for x in getattr(self, name)))
        n = len(self.charges)
        if not self.name or not n or len(self.sigma) != n or len(self.epsilon) != n:
            raise ValueError('state parameters must have the same nonzero length')
        vals = [*self.charges, *self.sigma, *self.epsilon, self.bound_protons,
                self.bound_electrons, self.intrinsic_kelvin,
                self.intrinsic_dimensionless, self.log_degeneracy]
        if not np.isfinite(vals).all() or min(self.sigma) < 0 or min(self.epsilon) < 0:
            raise ValueError('nonfinite or negative force-field parameter')

    @property
    def charge(self) -> float:
        return float(sum(self.charges))


@dataclass
class MolecularTemplate:
    name: str
    geometry: np.ndarray
    states: Sequence[MolecularState]
    explicit_water: bool = False
    citation: str = ''

    def __post_init__(self):
        self.geometry = np.asarray(self.geometry, float).copy()
        self.states = tuple(self.states)
        if not self.name or not self.states or self.geometry.shape != (len(self.states[0].charges), 3):
            raise ValueError('invalid molecular template')
        if not np.isfinite(self.geometry).all() or len({s.name for s in self.states}) != len(self.states):
            raise ValueError('invalid geometry or duplicate states')
        if any(len(s.charges) != len(self.geometry) for s in self.states):
            raise ValueError('all states need the same dummy-atom topology')
        for i in range(len(self.geometry)):
            if np.any(np.linalg.norm(self.geometry[i] - self.geometry[:i], axis=1) < 1e-10):
                raise ValueError('coincident sites are unsupported')
        self.geometry -= self.geometry[0]
        self.geometry.flags.writeable = False

    def signature(self):
        return {'name': self.name, 'geometry': self.geometry.tolist(),
                'states': [s.__dict__ for s in self.states],
                'explicit_water': self.explicit_water, 'citation': self.citation}


@dataclass
class Molecule:
    template: int
    state: int
    positions: np.ndarray
    uid: int = -1
    mobile: bool = True
    exchangeable: bool = True

    def __post_init__(self):
        self.positions = np.asarray(self.positions, float).reshape(-1, 3).copy()

    def clone(self):
        return Molecule(self.template, self.state, self.positions.copy(), self.uid,
                        self.mobile, self.exchangeable)


def spce_water() -> MolecularTemplate:
    """SPC/E, parameters documented by LAMMPS; normalized SO(3) measure.

    epsilon_O=0.1553 kcal/mol; sigma_O=3.166 A; qH=0.4238 e;
    rOH=1 A; tetrahedral angle acos(-1/3). No vaporization self-energy
    correction is silently added: water chemical potential must use this zero.
    """
    theta = math.acos(-1 / 3)
    state = MolecularState('water', (-.8476, .4238, .4238), (3.166, 0., 0.),
                           (.1553 * 4.184 * KJ_MOL_TO_K, 0., 0.))
    return MolecularTemplate('SPC/E', np.array([[0, 0, 0], [1, 0, 0],
                              [math.cos(theta), math.sin(theta), 0]]), [state],
                              explicit_water=True,
                              citation='10.1021/j100308a038; https://docs.lammps.org/Howto_spc.html')


def point_species(name, charge, sigma=3., epsilon=50.) -> MolecularTemplate:
    return MolecularTemplate(name, np.zeros((1, 3)),
                             [MolecularState(name, (charge,), (sigma,), (epsilon,))])


def orientation(rng: np.random.Generator) -> np.ndarray:
    """Haar-uniform rotation from a normalized four-dimensional Gaussian."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    return Rotation.from_quat(q).as_matrix()


def rotated_molecule(template: MolecularTemplate, anchor, rng) -> np.ndarray:
    return template.geometry @ orientation(rng).T + np.asarray(anchor)


class MolecularHamiltonian:
    """Molecule-aware transaction layer over cached 0.3 atomic Hamiltonian.

    Atoms of every molecule remain contiguous. Whole-molecule replacement uses
    the native range-exclusion path and never subtracts a singular new--old pair.
    Exclusion constants are precomputed once per rigid template and state.
    """
    def __init__(self, templates, cell, *, cutoff=8., alpha=.3, kmax=5,
                 coulomb='ewald', dielectric=1., shifted=False, backend='native',
                 algorithm='cached', electrodes=None, slab=False, **host):
        self.templates = list(templates)
        if not self.templates or len({t.name for t in self.templates}) != len(self.templates):
            raise ValueError('unique molecular templates required')
        if any(t.explicit_water for t in self.templates) and dielectric != 1.:
            raise ValueError('explicit SPC/E requires dielectric=1; do not double-screen water')
        self.atom_types = []
        species = []
        for ti, template in enumerate(self.templates):
            types = []
            for si, state in enumerate(template.states):
                row = []
                for j, (q, sig, eps) in enumerate(zip(state.charges, state.sigma, state.epsilon)):
                    row.append(len(species))
                    species.append(Ion(f'{ti}:{si}:{j}', q, sig, eps))
                types.append(np.asarray(row, int))
            self.atom_types.append(types)
        self.atomic = IonicHamiltonian(species, cell, cutoff, alpha, kmax, dielectric,
                                        coulomb, shifted, backend, algorithm, electrodes,
                                        slab=slab, **host)
        self.cell = self.atomic.cell
        self.exclusions = [[self._exclusion(t.geometry, s) for s in t.states]
                           for t in self.templates]
        self.molecules: list[Molecule] = []
        self.correction = 0.
        self.next_uid = 0

    def _exclusion(self, xyz, state):
        out = 0.
        h = self.atomic
        for i in range(len(xyz)):
            for j in range(i):
                r = float(np.linalg.norm(xyz[i] - xyz[j]))
                if r >= h.cutoff:
                    raise ValueError('all rigid intramolecular distances must be below cutoff')
                sig = .5 * (state.sigma[i] + state.sigma[j])
                eps = math.sqrt(state.epsilon[i] * state.epsilon[j])
                s6 = (sig / r) ** 6
                lj = 4 * eps * (s6 * s6 - s6)
                if h.shifted:
                    c6 = (sig / h.cutoff) ** 6
                    lj -= 4 * eps * (c6 * c6 - c6)
                out -= lj
                if h.ewald is not None:
                    out -= h.K * state.charges[i] * state.charges[j] / r
        return float(out)

    def _validate(self, mol):
        if not 0 <= mol.template < len(self.templates):
            raise ValueError('template out of range')
        t = self.templates[mol.template]
        if not 0 <= mol.state < len(t.states) or mol.positions.shape != t.geometry.shape or not np.isfinite(mol.positions).all():
            raise ValueError('invalid molecule')
        ref = np.linalg.norm(t.geometry[:, None] - t.geometry[None, :], axis=2)
        obs = np.linalg.norm(mol.positions[:, None] - mol.positions[None, :], axis=2)
        if not np.allclose(ref, obs, rtol=1e-10, atol=1e-8):
            raise ValueError('rigid geometry changed or molecule was atom-wise wrapped')

    def _flatten(self, molecules):
        types = [self.atom_types[m.template][m.state] for m in molecules]
        return (np.concatenate(types) if types else np.empty(0, int),
                np.concatenate([m.positions for m in molecules]) if types else np.empty((0, 3)))

    def set_molecules(self, molecules):
        mols = [m.clone() for m in molecules]
        used = {m.uid for m in mols if m.uid >= 0}
        if len(used) != sum(m.uid >= 0 for m in mols):
            raise ValueError('duplicate molecule UID')
        for m in mols:
            self._validate(m)
            if m.uid < 0:
                while self.next_uid in used:
                    self.next_uid += 1
                m.uid = self.next_uid
                used.add(m.uid)
                self.next_uid += 1
        self.next_uid = max(self.next_uid, max(used, default=-1) + 1)
        types, xyz = self._flatten(mols)
        self.atomic.set_state(types, xyz)
        self.molecules = mols
        self.correction = sum(self.exclusions[m.template][m.state] for m in mols)

    @property
    def energy(self):
        return self.atomic.energy + self.correction

    def full_energy(self):
        return self.atomic.full_energy() + sum(self._exclusion(m.positions, self.templates[m.template].states[m.state]) for m in self.molecules)

    def molecule_index(self, uid):
        return next(i for i, m in enumerate(self.molecules) if m.uid == uid)

    def propose(self, remove=(), add=()):
        remove = tuple(int(i) for i in remove)
        if len(set(remove)) != len(remove) or any(i < 0 or i >= len(self.molecules) for i in remove):
            raise ValueError('invalid molecule removal')
        adds = [m.clone() for m in add]
        for m in adds:
            self._validate(m)
        offsets = np.r_[0, np.cumsum([len(m.positions) for m in self.molecules])]
        atoms = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in remove]) if remove else np.empty(0, int)
        types, xyz = self._flatten(adds)
        trial = self.atomic.propose(atoms, types, xyz)
        dc = (sum(self.exclusions[m.template][m.state] for m in adds) -
              sum(self.exclusions[self.molecules[i].template][self.molecules[i].state] for i in remove))
        return {'atomic': trial, 'remove': remove, 'add': adds,
                'delta_correction': dc, 'delta_energy': trial['delta_energy'] + dc}

    def commit(self, trial):
        if not np.isfinite(trial['delta_energy']):
            raise ValueError('cannot commit nonfinite molecule energy')
        # UID uniqueness checked before modifying the atomic state.
        keep = [m for i, m in enumerate(self.molecules) if i not in trial['remove']]
        used = {m.uid for m in keep}
        for m in trial['add']:
            if m.uid < 0:
                m.uid = self.next_uid
                self.next_uid += 1
            if m.uid in used:
                raise ValueError('duplicate UID in proposal')
            used.add(m.uid)
        self.atomic.commit(trial['atomic'])
        self.molecules = keep + trial['add']
        self.correction += trial['delta_correction']

    def snapshot(self):
        return {'molecules': [m.clone() for m in self.molecules], 'next_uid': self.next_uid}

    def restore(self, snapshot):
        self.set_molecules(snapshot['molecules'])
        self.next_uid = snapshot['next_uid']

    def state_counts(self):
        return [np.bincount([m.state for m in self.molecules if m.template == i], minlength=len(t.states))
                for i, t in enumerate(self.templates)]

    def fingerprint(self):
        return content_key({'kind': 'molecular-hamiltonian-04',
                            'atomic': self.atomic.fingerprint(),
                            'templates': [t.signature() for t in self.templates]}, {})

    def close(self):
        self.atomic.close()


def hydration_shells(frames, ion_positions, *, water_template=0, cutoff=3.5, cell=None,
                     bins=None):
    """Raw ion--O distance and coordination statistics, including water alignment.

    For interfaces returns counts, not a bulk-RDF mislabeled with full-shell
    normalization. `frames` is a sequence of molecule lists; ions may be one
    fixed position per frame or multiple fixed probes. cos(theta) uses O->H2
    bisector dot ion->O, so negative indicates hydrogens pointing toward ion.
    """
    from .cell import closest_image
    if cutoff <= 0 or not np.isfinite(cutoff):
        raise ValueError('positive shell cutoff required')
    frames = list(frames)
    probes = np.asarray(ion_positions, float).reshape(-1, 3)
    if not len(frames) or not len(probes):
        raise ValueError('frames and probe positions required')
    distances, cosines, coordination = [], [], []
    for frame in frames:
        waters = [m for m in frame if m.template == water_template]
        for ion in probes:
            n = 0
            for w in waters:
                d = w.positions[0] - ion
                if cell is not None:
                    d = closest_image(d, cell)
                r = np.linalg.norm(d)
                distances.append(r)
                if r < cutoff:
                    n += 1
                    dip = w.positions[1] + w.positions[2] - 2 * w.positions[0]
                    norm = np.linalg.norm(dip) * r
                    cosines.append(float(dip @ d / norm) if norm else np.nan)
            coordination.append(n)
    edges = np.linspace(0, 10, 51) if bins is None else np.asarray(bins, float)
    hist, _ = np.histogram(distances, edges)
    return {'coordination': np.asarray(coordination), 'distances_A': np.asarray(distances),
            'orientation_cosine': np.asarray(cosines), 'distance_counts': hist, 'edges_A': edges,
            'normalization': 'raw histogram; no bulk density assumption'}
