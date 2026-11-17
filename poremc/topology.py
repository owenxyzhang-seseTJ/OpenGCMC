"""Molecular reference measures and constrained tree topology.

Positions use Angstrom; U/k_B and temperature use kelvin. A flexible topology
has fixed or explicitly bounded fluctuating bond lengths and bond angles,
with freely sampled torsions. The radial/angular reference measure is normalized,
including SO(3) for rigid molecules.
No finite internal Markov chain is used as an allegedly exact bond proposal.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Any
import numpy as np

KB_SI = 1.380649e-23


def _vector(value, n, dtype=float, default=0):
    if value is None:
        return np.full(n, default, dtype=dtype)
    a = np.asarray(value, dtype=dtype)
    if a.ndim == 0:
        a = np.full(n, a.item(), dtype=dtype)
    if a.shape != (n,):
        raise ValueError(f"expected vector of length {n}, got {a.shape}")
    return a.copy()


@dataclass
class Molecule:
    name: str
    sigma: np.ndarray
    epsilon: np.ndarray
    charge: np.ndarray | None = None
    parents: np.ndarray | None = None
    bond_lengths: np.ndarray | None = None
    angle_refs: np.ndarray | None = None
    angles: np.ndarray | None = None
    dihedral_refs: np.ndarray | None = None
    torsions: list[dict] = field(default_factory=list)
    exclusions: list[tuple[int, int]] | None = None
    log_q_internal: float | None = None
    rigid_positions: np.ndarray | None = None
    molar_mass: float | None = None
    q_internal_temperature: float | None = None
    bond_ranges: np.ndarray | None = None
    bond_force_constants: np.ndarray | None = None
    angle_ranges: np.ndarray | None = None
    angle_force_constants: np.ndarray | None = None

    def __post_init__(self):
        self.sigma = np.asarray(self.sigma, float).copy()
        if self.sigma.ndim != 1 or len(self.sigma) == 0:
            raise ValueError("sigma must contain one value per atom")
        n = self.n_atoms
        self.epsilon = _vector(self.epsilon, n)
        self.charge = _vector(self.charge, n)
        default_parents = np.arange(n, dtype=int) - 1
        self.parents = default_parents if self.parents is None else _vector(self.parents, n, int)
        self.bond_lengths = _vector(self.bond_lengths, n)
        if self.angle_refs is None:
            self.angle_refs = np.array([-1 if p < 0 else int(self.parents[p]) for p in self.parents], int)
        else:
            self.angle_refs = _vector(self.angle_refs, n, int)
        self.angles = _vector(self.angles, n, default=math.radians(109.5))
        self.bond_force_constants = _vector(self.bond_force_constants, n)
        self.angle_force_constants = _vector(self.angle_force_constants, n)
        self.bond_ranges = None if self.bond_ranges is None else np.asarray(self.bond_ranges, float).copy()
        self.angle_ranges = None if self.angle_ranges is None else np.asarray(self.angle_ranges, float).copy()
        for ranges, name in ((self.bond_ranges, "bond_ranges"), (self.angle_ranges, "angle_ranges")):
            if ranges is not None and (ranges.shape != (n, 2) or not np.isfinite(ranges).all()):
                raise ValueError(f"{name} must be a finite array of shape (n_atoms, 2)")
        for constants in (self.bond_force_constants, self.angle_force_constants):
            if not np.isfinite(constants).all() or np.any(constants < 0):
                raise ValueError("harmonic force constants must be finite and nonnegative")
        if self.dihedral_refs is None:
            self.dihedral_refs = np.array([-1 if a < 0 else int(self.parents[a]) for a in self.angle_refs], int)
        else:
            self.dihedral_refs = _vector(self.dihedral_refs, n, int)
        for a in (self.sigma, self.epsilon, self.charge, self.bond_lengths, self.angles):
            if not np.all(np.isfinite(a)):
                raise ValueError("molecular parameters must be finite")
        if np.any(self.sigma < 0) or np.any(self.epsilon < 0):
            raise ValueError("sigma and epsilon must be nonnegative")
        if self.parents[0] != -1:
            raise ValueError("atom 0 is the unique growth root")
        if self.rigid_positions is not None:
            self.rigid_positions = np.asarray(self.rigid_positions, float).copy()
            if self.rigid_positions.shape != (n, 3) or not np.all(np.isfinite(self.rigid_positions)):
                raise ValueError("rigid_positions must have shape (n_atoms, 3)")
            self.rigid_positions -= self.rigid_positions[0]
        else:
            for i in range(1, n):
                p, a, d = int(self.parents[i]), int(self.angle_refs[i]), int(self.dihedral_refs[i])
                if p < 0 or p >= i:
                    raise ValueError("tree parents must precede each atom in growth order")
                if self.bond_lengths[i] <= 0:
                    raise ValueError("nonroot bond lengths must be positive")
                if a >= i or a == p or a < -1:
                    raise ValueError("angle reference must be an earlier atom distinct from parent")
                if d >= i or d in (p, a) and d >= 0 or d < -1:
                    raise ValueError("dihedral reference must be an earlier distinct atom")
                if a >= 0 and not 0 < self.angles[i] < math.pi:
                    raise ValueError("constrained bond angles must lie strictly between 0 and pi")
        if self.rigid_positions is not None:
            for ranges in (self.bond_ranges, self.angle_ranges):
                if ranges is not None and np.any(ranges[:, 1] != ranges[:, 0]):
                    raise ValueError("rigid positions cannot have fluctuating bond or angle ranges")
        else:
            for i in range(1, n):
                if self.bond_ranges is not None:
                    lo, hi = self.bond_ranges[i]
                    if not 0 < lo <= hi:
                        raise ValueError("nonroot bond bounds must satisfy 0 < lower <= upper")
                    if lo == hi and not math.isclose(lo, self.bond_lengths[i], abs_tol=1e-12):
                        raise ValueError("fixed bond range must equal its reference bond length")
                if self.angle_ranges is not None and self.angle_refs[i] >= 0:
                    lo, hi = self.angle_ranges[i]
                    if not 0 < lo <= hi < math.pi:
                        raise ValueError("bond angle bounds must satisfy 0 < lower <= upper < pi")
                    if lo == hi and not math.isclose(lo, self.angles[i], abs_tol=1e-12):
                        raise ValueError("fixed angle range must equal its reference angle")
        if self.bond_ranges is not None and not np.array_equal(self.bond_ranges[0], [0., 0.]):
            raise ValueError("root bond range must be [0, 0]")
        if self.exclusions is None:
            bonds = {(min(i, int(p)), max(i, int(p))) for i, p in enumerate(self.parents) if p >= 0}
            adj = [set() for _ in range(n)]
            for i, j in bonds:
                adj[i].add(j); adj[j].add(i)
            ex = set(bonds)
            for i in range(n):
                for j in adj[i]:
                    for k in adj[j]:
                        if k != i:
                            ex.add(tuple(sorted((i, k))))
            self.exclusions = sorted(ex)
        else:
            self.exclusions = sorted({tuple(sorted(map(int, x))) for x in self.exclusions})
        for pair in self.exclusions:
            if len(pair) != 2 or pair[0] < 0 or pair[1] >= n or pair[0] == pair[1]:
                raise ValueError("invalid intramolecular exclusion")
        self.torsions = [dict(t) for t in self.torsions]
        for t in self.torsions:
            ids = list(map(int, t["indices"]))
            if len(ids) != 4 or len(set(ids)) != 4 or min(ids) < 0 or max(ids) >= n:
                raise ValueError("torsion indices must be four distinct valid atoms")
            t["indices"] = ids
            t["k"] = float(t.get("k", 0.0))
            t["periodicity"] = int(t.get("periodicity", 1))
            t["phase"] = float(t.get("phase", 0.0))
            if t["periodicity"] < 1 or not np.isfinite([t["k"], t["phase"]]).all():
                raise ValueError("invalid torsion parameters")
        if self.log_q_internal is not None and not np.isfinite(self.log_q_internal):
            raise ValueError("log_q_internal must be finite")
        if self.q_internal_temperature is not None and self.q_internal_temperature <= 0:
            raise ValueError("q_internal_temperature must be positive")

    @property
    def n_atoms(self):
        return len(self.sigma)

    @property
    def is_rigid(self):
        return self.rigid_positions is not None or self.n_atoms == 1

    def validate_positions(self, coordinates, atol=1e-7):
        """Verify that a stored molecule belongs to its constrained state space.

        Coordinates must be unwrapped within each molecule. For a rigid body,
        proper-rotation Kabsch alignment also rejects a mirror image of a
        noncoplanar chiral reference, which pair distances alone would accept.
        """
        x = np.asarray(coordinates, float)
        if x.shape != (self.n_atoms, 3) or not np.isfinite(x).all():
            raise ValueError("molecular positions must be finite with shape (n_atoms, 3)")
        if self.rigid_positions is not None:
            a = self.rigid_positions - self.rigid_positions.mean(axis=0)
            b = x - x.mean(axis=0)
            u, _, vt = np.linalg.svd(a.T @ b)
            correction = np.eye(3)
            correction[-1, -1] = 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0
            r = u @ correction @ vt
            if np.max(np.linalg.norm(a @ r - b, axis=1)) > atol:
                raise ValueError("coordinates do not match the rigid reference by a proper rotation")
        elif not self.is_rigid:
            for i in range(1, self.n_atoms):
                p = int(self.parents[i])
                v = x[i] - x[p]
                length = np.linalg.norm(v)
                bounds = None if self.bond_ranges is None else self.bond_ranges[i]
                if bounds is None or bounds[0] == bounds[1]:
                    if not math.isclose(length, self.bond_lengths[i], rel_tol=atol, abs_tol=atol):
                        raise ValueError(f"atom {i} violates its constrained bond length; use unwrapped coordinates")
                elif not bounds[0]-atol <= length <= bounds[1]+atol:
                    raise ValueError(f"atom {i} lies outside its finite bond-length interval")
                a = int(self.angle_refs[i])
                if a >= 0:
                    ref = x[a] - x[p]
                    refnorm = np.linalg.norm(ref)
                    if refnorm < atol:
                        raise ValueError("coincident angle reference atoms")
                    cosine = np.dot(v, ref)/(length*refnorm)
                    bounds = None if self.angle_ranges is None else self.angle_ranges[i]
                    if bounds is None or bounds[0] == bounds[1]:
                        if not math.isclose(cosine, math.cos(self.angles[i]), abs_tol=atol, rel_tol=atol):
                            raise ValueError(f"atom {i} violates its constrained bond angle")
                    else:
                        angle = math.acos(float(np.clip(cosine, -1., 1.)))
                        if not bounds[0]-atol <= angle <= bounds[1]+atol:
                            raise ValueError(f"atom {i} lies outside its finite bond-angle interval")
        return x

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        data = dict(data)
        if "angles_degrees" in data:
            if "angles" in data:
                raise ValueError("specify angles or angles_degrees, not both")
            data["angles"] = np.deg2rad(data.pop("angles_degrees"))
        if "angle_ranges_degrees" in data:
            if "angle_ranges" in data:
                raise ValueError("specify angle_ranges or angle_ranges_degrees, not both")
            data["angle_ranges"] = np.deg2rad(data.pop("angle_ranges_degrees"))
        return cls(**data)

    def to_dict(self):
        return {"name": self.name, "sigma": self.sigma.tolist(), "epsilon": self.epsilon.tolist(),
                "charge": self.charge.tolist(), "parents": self.parents.tolist(),
                "bond_lengths": self.bond_lengths.tolist(), "angle_refs": self.angle_refs.tolist(),
                "angles": self.angles.tolist(), "dihedral_refs": self.dihedral_refs.tolist(),
                "torsions": self.torsions, "exclusions": [list(x) for x in self.exclusions],
                "log_q_internal": self.log_q_internal,
                "rigid_positions": None if self.rigid_positions is None else self.rigid_positions.tolist(),
                "molar_mass": self.molar_mass, "q_internal_temperature": self.q_internal_temperature,
                "bond_ranges": None if self.bond_ranges is None else self.bond_ranges.tolist(),
                "bond_force_constants": self.bond_force_constants.tolist(),
                "angle_ranges": None if self.angle_ranges is None else self.angle_ranges.tolist(),
                "angle_force_constants": self.angle_force_constants.tolist()}


def activity_from_fugacity_pa(fugacity_pa: float, temperature: float, molecule: Molecule) -> float:
    """Return activity [A^-3] for this normalized internal reference measure.

    z = f/(k_B T q_int); q_int is the isolated molecular configurational
    partition integral over exactly the same constrained reference measure.
    For flexible molecules the caller must provide it; zero is NOT assumed.
    """
    if temperature <= 0 or fugacity_pa < 0 or not np.isfinite([temperature, fugacity_pa]).all():
        raise ValueError("temperature must be positive and fugacity nonnegative")
    logq = molecule.log_q_internal
    if logq is None:
        if not molecule.is_rigid:
            raise ValueError("flexible fugacity conversion requires explicit log_q_internal")
        logq = 0.0
    if molecule.q_internal_temperature is not None and not math.isclose(
            temperature, molecule.q_internal_temperature, rel_tol=1e-12, abs_tol=1e-10):
        raise ValueError("log_q_internal was specified at a different temperature")
    if fugacity_pa == 0:
        return 0.0
    return math.exp(math.log(fugacity_pa) - math.log(KB_SI * temperature) - 30 * math.log(10) - logq)


def independent_torsion_log_partition(k: float, temperature: float) -> float:
    """log mean exp[-k(1+cos(n phi-phase))/T] for one independent torsion.

    No nonbonded or other torsional coupling may be present in this analytic
    specialization. A scaled Bessel implementation avoids overflow.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    x = float(k) / temperature
    try:
        from scipy.special import i0e
        return -x + abs(x) + math.log(float(i0e(x)))
    except ImportError:
        if abs(x) > 600:
            raise ValueError("large torsion stiffness requires scipy.special.i0e")
        return -x + math.log(float(np.i0(x)))


def independent_bond_angle_log_partition(molecule: Molecule, temperature: float) -> float:
    """Normalized isolated partition integral for independent bonded terms.

    For each variable bond, integrate 3*r^2/(hi^3-lo^3) times its harmonic
    Boltzmann factor. For each variable bond angle, use sin(theta) divided
    by cos(lo)-cos(hi). Fixed constraints contribute one. This function
    excludes torsions and all nonbonded couplings, which must be included in
    the full q_internal used for physical-fugacity conversion.
    """
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be positive and finite")
    if molecule.is_rigid:
        return 0.0
    from scipy.integrate import quad

    def log_integral(lo, hi, center, stiffness, kind):
        if lo == hi or stiffness == 0:
            return 0.0
        minimum = float(np.clip(center, lo, hi))
        emin = .5*stiffness*(minimum-center)**2
        if kind == "bond":
            norm = 3./(hi**3-lo**3)
            jacobian = lambda x: x*x
        else:
            norm = 1./(math.cos(lo)-math.cos(hi))
            jacobian = math.sin
        def integrand(x):
            du = .5*stiffness*(x-center)**2-emin
            return jacobian(x)*math.exp(-du/temperature)
        # Explicitly split at the harmonic minimum so narrow wells are resolved
        # from both sides; any residual quadrature failure is reported.
        points = [center] if lo < center < hi else None
        integral, error = quad(integrand, lo, hi, points=points,
                               epsabs=1e-12, epsrel=1e-11, limit=200)
        if not integral > 0 or not math.isfinite(integral):
            raise FloatingPointError("bonded partition quadrature underflowed")
        if error > max(1e-10*integral, 1e-12):
            raise FloatingPointError("bonded partition quadrature did not converge")
        return math.log(norm)+math.log(integral)-emin/temperature

    total = 0.0
    for i in range(1, molecule.n_atoms):
        if molecule.bond_ranges is not None:
            lo, hi = molecule.bond_ranges[i]
            total += log_integral(lo, hi, molecule.bond_lengths[i],
                                  molecule.bond_force_constants[i], "bond")
        if molecule.angle_refs[i] >= 0 and molecule.angle_ranges is not None:
            lo, hi = molecule.angle_ranges[i]
            total += log_integral(lo, hi, molecule.angles[i],
                                  molecule.angle_force_constants[i], "angle")
    return total


def random_unit_vectors(rng, count):
    x = rng.normal(size=(count, 3))
    lengths = np.linalg.norm(x, axis=1)
    while np.any(lengths == 0):
        bad = lengths == 0
        x[bad] = rng.normal(size=(int(bad.sum()), 3))
        lengths = np.linalg.norm(x, axis=1)
    return x / lengths[:, None]


def random_rotation(rng):
    """Uniform SO(3), obtained from a normalized isotropic Gaussian quaternion."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def local_rotation(rng, max_angle):
    axis = random_unit_vectors(rng, 1)[0]
    a = rng.uniform(-max_angle, max_angle)
    x, y, z = axis
    c, s = math.cos(a), math.sin(a)
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return c * np.eye(3) + (1-c) * np.outer(axis, axis) + s * cross


def propose_atom(molecule: Molecule, prefix, atom_index, n_trials, cell, rng):
    """Sample the normalized constrained reference conditional distribution."""
    if atom_index == 0:
        return rng.random((n_trials, 3)) @ cell
    p = int(molecule.parents[atom_index])
    bounds = None if molecule.bond_ranges is None else molecule.bond_ranges[atom_index]
    if bounds is not None and bounds[1] > bounds[0]:
        lo, hi = bounds
        r = np.cbrt(lo**3 + rng.random(n_trials)*(hi**3-lo**3))
    else:
        r = np.full(n_trials, molecule.bond_lengths[atom_index])
    a = int(molecule.angle_refs[atom_index])
    if a < 0:
        return prefix[p] + r[:, None] * random_unit_vectors(rng, n_trials)
    # e1 points from the parent towards the angle reference.
    e1 = np.asarray(prefix[a] - prefix[p], float)
    norm = np.linalg.norm(e1)
    if norm < 1e-12:
        raise ValueError("coincident angle reference atoms")
    e1 /= norm
    d = int(molecule.dihedral_refs[atom_index])
    v = prefix[d] - prefix[a] if d >= 0 else np.eye(3)[np.argmin(np.abs(e1))]
    e2 = v - np.dot(v, e1) * e1
    if np.linalg.norm(e2) < 1e-12:
        v = np.eye(3)[np.argmin(np.abs(e1))]
        e2 = v - np.dot(v, e1) * e1
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    phi = rng.uniform(-math.pi, math.pi, n_trials)
    bounds = None if molecule.angle_ranges is None else molecule.angle_ranges[atom_index]
    if bounds is not None and bounds[1] > bounds[0]:
        cos_lo, cos_hi = math.cos(bounds[0]), math.cos(bounds[1])
        cosine = cos_lo - rng.random(n_trials)*(cos_lo-cos_hi)
    else:
        cosine = np.full(n_trials, math.cos(molecule.angles[atom_index]))
    sine = np.sqrt(np.maximum(0., 1-cosine*cosine))
    directions = cosine[:, None] * e1 + sine[:, None] * (
        np.cos(phi)[:, None] * e2 + np.sin(phi)[:, None] * e3)
    return prefix[p] + r[:, None] * directions
