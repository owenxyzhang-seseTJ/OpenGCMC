"""Molecular NPT, two-box Gibbs NVT, replica exchange and TMMC.

These are executable ensemble kernels, independent of the GCMC driver. Energies
are U/k_B in kelvin; cells and volumes are Angstrom and Angstrom**3; external
pressure is in Pa. Molecules are (species_index, unwrapped_coordinates) pairs.

Volume moves scale ONLY each molecule's root position. Internal coordinates,
bond constraints and rigid orientations do not change. The NPT variable is
log(V); Gibbs uses log(V1/V2), hence the proposal-Jacobian exponents N+1.

Volume-changing ensembles here are bulk-fluid ensembles, with NO framework.
A potential_factory(cell) must construct the same target Hamiltonian at the
new cell and must provide total_energy(entries). It must not change force-field
parameters with density. If finite cutoffs constrain the minimum cell size,
the sampled volume distribution is correspondingly truncated. Classical
potentials also require face heights larger than molecular diameter + cutoff;
volume proposals outside either bound are rejected before potential creation.

TMMC is an explicit controller: call accept(...) for every macrostate-changing
proposal, or record(...) with its UNBIASED expected acceptance. Biases are log
weights added to log(pi). Freeze them before production reweighting.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .cbmc import CBMC
from .cell import validate_cell
from .units import KB


def _accept(log_ratio, rng):
    if math.isnan(log_ratio):
        raise FloatingPointError("NaN Metropolis ratio")
    return math.log(max(float(rng.random()), np.finfo(float).tiny)) < min(0.0, log_ratio)


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _copies(entries):
    return [(int(s), np.array(x, dtype=float, copy=True)) for s, x in entries]


def _validate_entries(entries, species):
    result = _copies(entries)
    for s, x in result:
        if s < 0 or s >= len(species):
            raise ValueError("species index out of range")
        if x.shape != (species[s].n_atoms, 3) or not np.isfinite(x).all():
            raise ValueError("molecule coordinates have wrong shape or are nonfinite")
        species[s].validate_positions(x)
    return result


def _no_framework(potential):
    if len(getattr(potential, "host_positions", [])):
        raise ValueError("NPT/Gibbs volume ensembles require a bulk fluid without framework atoms")
    if getattr(potential, "contains_framework", False):
        raise ValueError("framework-bearing potential is not supported by volume ensembles")


def _environment(potential, entries):
    return potential.prepare_environment(entries) if hasattr(potential, "prepare_environment") else entries


def molecular_scale(entries, old_cell, new_cell):
    """Scale fractional root positions, preserving unwrapped internal geometry."""
    old, inverse, _ = validate_cell(old_cell)
    new, _, _ = validate_cell(new_cell)
    result = []
    for s, x in entries:
        x = np.asarray(x, float)
        old_root = x[0]
        new_root = (old_root @ inverse) @ new
        result.append((int(s), x + (new_root - old_root)))
    return result


def npt_log_volume_ratio(n, temperature, pressure_pa, old_volume, new_volume, delta_energy):
    """Log MH ratio for a symmetric proposal in log(V), molecular coordinates."""
    _positive(temperature, "temperature")
    _positive(pressure_pa, "pressure")
    _positive(old_volume, "old volume"); _positive(new_volume, "new volume")
    work_kelvin = pressure_pa * (new_volume - old_volume) * 1e-30 / KB
    return -(delta_energy + work_kelvin) / temperature + (int(n)+1)*math.log(new_volume/old_volume)


def gibbs_transfer_log_ratio(n_source, n_destination, v_source, v_destination,
                             log_weight_new, log_weight_old):
    """Same-species transfer at equal T; gas-reference normalization cancels."""
    if n_source <= 0 or n_destination < 0:
        raise ValueError("invalid Gibbs component populations")
    _positive(v_source, "source volume"); _positive(v_destination, "destination volume")
    return (math.log(n_source) - math.log(n_destination+1)
            + math.log(v_destination/v_source) + log_weight_new-log_weight_old)


def gibbs_log_volume_ratio(n1, n2, old_v1, old_v2, new_v1, new_v2, delta_energy, temperature):
    """Symmetric proposal in log(V1/V2) at fixed total volume."""
    _positive(temperature, "temperature")
    for value in (old_v1, old_v2, new_v1, new_v2):
        _positive(value, "volume")
    if not math.isclose(old_v1+old_v2, new_v1+new_v2, rel_tol=1e-12):
        raise ValueError("Gibbs volume proposal must conserve total volume")
    return ((int(n1)+1)*math.log(new_v1/old_v1)
            + (int(n2)+1)*math.log(new_v2/old_v2)-delta_energy/temperature)


@dataclass
class MolecularBox:
    cell: np.ndarray
    entries: list
    potential: object
    energy: float
    minimum_face_height: float = 0.0

    @property
    def volume(self):
        return float(np.linalg.det(self.cell))

    @property
    def count(self):
        return len(self.entries)

    def component_count(self, species_index):
        return sum(s == species_index for s, _ in self.entries)


def _minimum_face_height(potential):
    """Cache known geometric support without treating arbitrary errors as rejections.

    ClassicalPotential requires both a unique finite-cutoff image and no
    short-range interaction with another image of the same molecule. The
    molecular bound is the rigid diameter or longest tree path using each
    allowed bond's upper bound, exactly as in that potential's constructor.
    Custom potentials retain only their declared cutoff constraint.
    """
    from .potentials import ClassicalPotential
    cutoff = getattr(potential, "cutoff", None)
    if cutoff is None:
        return 0.0
    cutoff = float(cutoff)
    lower = 2*cutoff
    if isinstance(potential, ClassicalPotential):
        for molecule in potential.species:
            if molecule.is_rigid:
                body = (molecule.rigid_positions if molecule.rigid_positions is not None
                        else np.zeros((1, 3)))
                diameter = float(np.linalg.norm(body[:, None, :]-body[None, :, :], axis=2).max())
            else:
                n = molecule.n_atoms
                path = np.full((n, n), np.inf)
                np.fill_diagonal(path, 0.)
                for i, parent in enumerate(molecule.parents):
                    if parent >= 0:
                        length = (molecule.bond_ranges[i, 1] if molecule.bond_ranges is not None
                                  else molecule.bond_lengths[i])
                        path[i, parent] = path[parent, i] = length
                for k in range(n):
                    path = np.minimum(path, path[:, k, None]+path[None, k, :])
                diameter = float(path.max())
            lower = max(lower, diameter+cutoff)
    return float(lower)


def _make_box(species, cell, entries, potential_factory, require_no_framework=True):
    cell, _, _ = validate_cell(cell)
    entries = _validate_entries(entries, species)
    potential = potential_factory(cell.copy())
    if require_no_framework:
        _no_framework(potential)
    energy = float(potential.total_energy(entries))
    if not math.isfinite(energy):
        raise ValueError("initial configuration must have finite total energy")
    return MolecularBox(cell.copy(), entries, potential, energy,
                        minimum_face_height=_minimum_face_height(potential))


def _trial_box(box, new_cell, potential_factory):
    new_cell, _, heights = validate_cell(new_cell)
    # Outside the known Hamiltonian support, the proposal is a rejected
    # self-loop. Do this before constructing a new potential; unrelated
    # potential_factory exceptions must still propagate to the caller.
    if min(heights) <= box.minimum_face_height:
        return None
    entries = molecular_scale(box.entries, box.cell, new_cell)
    potential = potential_factory(new_cell.copy())
    _no_framework(potential)
    energy = float(potential.total_energy(entries))
    if math.isnan(energy) or energy == -math.inf:
        raise FloatingPointError("invalid energy in volume proposal")
    return MolecularBox(new_cell.copy(), entries, potential, energy,
                        minimum_face_height=box.minimum_face_height)


def _regrow(box, species, temperature, trials, rng):
    if not box.entries:
        return False
    index = int(rng.integers(len(box.entries)))
    s, old_positions = box.entries[index]
    background = box.entries[:index]+box.entries[index+1:]
    env = _environment(box.potential, background)
    cbmc = CBMC(temperature, box.cell, box.potential, trials=trials)
    old = cbmc.retrace(species[s], old_positions, env, rng)
    new = cbmc.grow(species[s], env, rng)
    if not old.valid:
        raise FloatingPointError("current molecule has invalid retrace weight")
    if not new.valid or not _accept(new.log_weight-old.log_weight, rng):
        return False
    box.entries[index] = (s, new.positions.copy())
    box.energy += new.energy-old.energy
    return True


class MolecularNPT:
    """Fixed-composition bulk NPT: log-volume moves + full CBMC regrowth.

    run() returns instantaneous volume and energy samples, including rejected
    moves. No automatic parameter adaptation occurs during production.
    """
    def __init__(self, species, temperature, pressure_pa, cell, entries,
                 potential_factory, trials=8, log_volume_step=0.2,
                 volume_probability=0.1, seed=12345):
        self.species = list(species)
        if not self.species:
            raise ValueError("at least one species definition is required")
        self.temperature = _positive(temperature, "temperature")
        self.pressure_pa = _positive(pressure_pa, "pressure")
        self.log_volume_step = _positive(log_volume_step, "log volume step")
        if not 0 <= volume_probability <= 1:
            raise ValueError("volume_probability must lie in [0,1]")
        self.volume_probability = float(volume_probability)
        self.potential_factory = potential_factory
        self.trials = trials
        self.rng = np.random.default_rng(seed)
        self.box = _make_box(self.species, cell, entries, potential_factory)
        self.stats = {"volume_attempted": 0, "volume_accepted": 0,
                      "regrow_attempted": 0, "regrow_accepted": 0}

    def volume_move(self):
        self.stats["volume_attempted"] += 1
        log_ratio_v = float(self.rng.uniform(-self.log_volume_step, self.log_volume_step))
        scale = math.exp(log_ratio_v/3)
        proposed = _trial_box(self.box, self.box.cell*scale, self.potential_factory)
        if proposed is None:
            return False
        log_ratio = npt_log_volume_ratio(self.box.count, self.temperature, self.pressure_pa,
                                        self.box.volume, proposed.volume, proposed.energy-self.box.energy)
        if not _accept(log_ratio, self.rng):
            return False
        self.box = proposed
        self.stats["volume_accepted"] += 1
        return True

    def step(self):
        if self.rng.random() < self.volume_probability:
            return self.volume_move()
        self.stats["regrow_attempted"] += 1
        accepted = _regrow(self.box, self.species, self.temperature, self.trials, self.rng)
        self.stats["regrow_accepted"] += int(accepted)
        return accepted

    def run(self, steps, burn_in=0, sample_every=1):
        """Run burn_in attempts, then steps production attempts (GCMC convention)."""
        steps, burn_in, sample_every = _run_arguments(steps, burn_in, sample_every)
        for _ in range(burn_in):
            self.step()
        volumes, energies = [], []
        for i in range(steps):
            self.step()
            if (i+1) % sample_every == 0:
                volumes.append(self.box.volume); energies.append(self.box.energy)
        return {"volume_A3": np.array(volumes), "energy_K": np.array(energies)}


def _run_arguments(steps, burn_in, sample_every):
    if (int(steps) != steps or int(burn_in) != burn_in or int(sample_every) != sample_every
            or steps < 1 or burn_in < 0 or sample_every < 1):
        raise ValueError("require integer steps >= 1, burn_in >= 0 and sample_every >= 1")
    return int(steps), int(burn_in), int(sample_every)


class GibbsNVT:
    """Two bulk boxes, common T, conserved N_s and total volume.

    A species and donor box are chosen uniformly from fixed sets. Empty-donor
    attempts are rejected self-loops; selection probabilities therefore cancel.
    Each transfer uses fresh destination growth and source retracing after
    excluding the source molecule. Molecular internal reference factors cancel.
    """
    def __init__(self, species, temperature, cells, entries, potential_factory,
                 trials=8, log_volume_step=0.2, transfer_probability=0.45,
                 volume_probability=0.1, seed=12345):
        self.species = list(species)
        if not self.species or len(cells) != 2 or len(entries) != 2:
            raise ValueError("Gibbs needs two boxes and at least one species")
        self.temperature = _positive(temperature, "temperature")
        self.log_volume_step = _positive(log_volume_step, "log volume step")
        if not (0 <= transfer_probability <= 1 and 0 <= volume_probability <= 1
                and transfer_probability+volume_probability <= 1):
            raise ValueError("Gibbs move probabilities must be nonnegative and sum to <= 1")
        self.transfer_probability = float(transfer_probability)
        self.volume_probability = float(volume_probability)
        self.potential_factory = potential_factory
        self.trials = trials
        self.rng = np.random.default_rng(seed)
        self.boxes = [_make_box(self.species, cells[i], entries[i], potential_factory) for i in range(2)]
        self.total_volume = sum(box.volume for box in self.boxes)
        self.total_counts = np.array([sum(b.component_count(s) for b in self.boxes)
                                      for s in range(len(self.species))], int)
        self.stats = {name+suffix: 0 for name in ("transfer", "volume", "regrow")
                      for suffix in ("_attempted", "_accepted")}

    def transfer_move(self):
        self.stats["transfer_attempted"] += 1
        donor = int(self.rng.integers(2)); acceptor = 1-donor
        source, dest = self.boxes[donor], self.boxes[acceptor]
        s = int(self.rng.integers(len(self.species)))
        eligible = [i for i, (species_index, _) in enumerate(source.entries) if species_index == s]
        if not eligible:
            return False
        index = int(self.rng.choice(eligible))
        old_positions = source.entries[index][1]
        background = source.entries[:index]+source.entries[index+1:]
        old_cbmc = CBMC(self.temperature, source.cell, source.potential, self.trials)
        new_cbmc = CBMC(self.temperature, dest.cell, dest.potential, self.trials)
        old = old_cbmc.retrace(self.species[s], old_positions,
                              _environment(source.potential, background), self.rng)
        if not old.valid:
            raise FloatingPointError("source molecule has invalid retrace weight")
        new = new_cbmc.grow(self.species[s], _environment(dest.potential, dest.entries), self.rng)
        if not new.valid:
            return False
        ratio = gibbs_transfer_log_ratio(len(eligible), dest.component_count(s),
                                        source.volume, dest.volume, new.log_weight, old.log_weight)
        if not _accept(ratio, self.rng):
            return False
        source.entries = background
        dest.entries.append((s, new.positions.copy()))
        source.energy -= old.energy
        dest.energy += new.energy
        self.stats["transfer_accepted"] += 1
        return True

    def volume_move(self):
        self.stats["volume_attempted"] += 1
        first, second = self.boxes
        x = math.log(first.volume/second.volume)
        x += float(self.rng.uniform(-self.log_volume_step, self.log_volume_step))
        # Stable logistic and endpoint protection; no clipping of a valid move.
        exp_minus_abs = math.exp(-abs(x))
        fraction = 1/(1+exp_minus_abs) if x >= 0 else exp_minus_abs/(1+exp_minus_abs)
        v1 = self.total_volume*fraction; v2 = self.total_volume-v1
        if min(v1, v2) <= 0:
            return False
        new_first = _trial_box(first, first.cell*(v1/first.volume)**(1/3), self.potential_factory)
        if new_first is None:
            return False
        new_second = _trial_box(second, second.cell*(v2/second.volume)**(1/3), self.potential_factory)
        if new_second is None:
            return False
        delta = new_first.energy+new_second.energy-first.energy-second.energy
        ratio = gibbs_log_volume_ratio(first.count, second.count, first.volume, second.volume,
                                      new_first.volume, new_second.volume, delta, self.temperature)
        if not _accept(ratio, self.rng):
            return False
        self.boxes = [new_first, new_second]
        self.stats["volume_accepted"] += 1
        return True

    def step(self):
        u = self.rng.random()
        if u < self.transfer_probability:
            return self.transfer_move()
        if u < self.transfer_probability+self.volume_probability:
            return self.volume_move()
        self.stats["regrow_attempted"] += 1
        box = self.boxes[int(self.rng.integers(2))]
        accepted = _regrow(box, self.species, self.temperature, self.trials, self.rng)
        self.stats["regrow_accepted"] += int(accepted)
        return accepted

    def run(self, steps, burn_in=0, sample_every=1):
        """Run burn_in attempts, then steps production attempts (GCMC convention)."""
        steps, burn_in, sample_every = _run_arguments(steps, burn_in, sample_every)
        for _ in range(burn_in):
            self.step()
        volumes, counts, energies = [], [], []
        for i in range(steps):
            self.step()
            if (i+1) % sample_every == 0:
                volumes.append([b.volume for b in self.boxes])
                energies.append([b.energy for b in self.boxes])
                counts.append([[b.component_count(s) for s in range(len(self.species))]
                               for b in self.boxes])
        return {"volume_A3": np.array(volumes), "energy_K": np.array(energies),
                "counts": np.array(counts, dtype=int)}


class CanonicalReplica:
    """Fixed-cell, fixed-composition replica with CBMC configuration updates."""
    def __init__(self, species, temperature, cell, entries, potential_factory,
                 trials=8, seed=12345):
        self.species = list(species)
        self.temperature = _positive(temperature, "temperature")
        self.box = _make_box(self.species, cell, entries, potential_factory,
                             require_no_framework=False)
        self.trials = trials
        self.rng = np.random.default_rng(seed)

    def step(self):
        return _regrow(self.box, self.species, self.temperature, self.trials, self.rng)


class ReplicaExchange:
    """Exact canonical temperature/Hamiltonian exchange with crossed energies.

    Cells, composition, constraints and host geometry must be shared. Potential
    parameters may differ because both cross-Hamiltonian energies are evaluated.
    This class does not perform grand-canonical or pressure replica exchange.
    """
    def __init__(self, replicas, seed=12345):
        self.replicas = list(replicas)
        if len(self.replicas) < 2:
            raise ValueError("replica exchange requires at least two replicas")
        self.rng = np.random.default_rng(seed)
        first = self.replicas[0]
        first_counts = [first.box.component_count(s) for s in range(len(first.species))]
        topology_fields = ("parents", "bond_lengths", "angle_refs", "angles", "dihedral_refs", "rigid_positions",
                           "bond_ranges", "angle_ranges")
        for replica in self.replicas[1:]:
            if not np.allclose(first.box.cell, replica.box.cell, rtol=0, atol=1e-12):
                raise ValueError("canonical exchange requires identical cells")
            if len(first.species) != len(replica.species):
                raise ValueError("replicas must share species topology")
            for a, b in zip(first.species, replica.species):
                if a.n_atoms != b.n_atoms:
                    raise ValueError("replicas must share species topology")
                for field in topology_fields:
                    x, y = getattr(a, field), getattr(b, field)
                    if (x is None) != (y is None) or (x is not None and not np.array_equal(x, y)):
                        raise ValueError("replicas must share molecular constraints")
            counts = [replica.box.component_count(s) for s in range(len(replica.species))]
            if counts != first_counts:
                raise ValueError("canonical exchange requires identical component populations")
            x = np.asarray(getattr(first.box.potential, "host_positions", []))
            y = np.asarray(getattr(replica.box.potential, "host_positions", []))
            if x.shape != y.shape or not np.array_equal(x, y):
                raise ValueError("replicas must share host geometry")
        self.attempted = 0; self.accepted = 0

    def attempt(self, i, j):
        if i == j or min(i, j) < 0 or max(i, j) >= len(self.replicas):
            raise ValueError("choose two different valid replicas")
        a, b = self.replicas[i], self.replicas[j]
        ea_cross = float(a.box.potential.total_energy(b.box.entries))
        eb_cross = float(b.box.potential.total_energy(a.box.entries))
        ratio = -(ea_cross-a.box.energy)/a.temperature-(eb_cross-b.box.energy)/b.temperature
        self.attempted += 1
        if not _accept(ratio, self.rng):
            return False
        a.box.entries, b.box.entries = b.box.entries, a.box.entries
        a.box.energy, b.box.energy = ea_cross, eb_cross
        self.accepted += 1
        return True


def activity_exchange_log_ratio(count_a,count_b,activity_a,activity_b):
    """Log acceptance ratio for swapping two equal-T, equal-Hamiltonian labels."""
    na,nb,za,zb=(np.asarray(a,dtype=float) for a in (count_a,count_b,activity_a,activity_b))
    if na.ndim!=1 or any(x.shape!=na.shape for x in (nb,za,zb)):
        raise ValueError('all four arrays must have one entry per component')
    if any(not np.isfinite(x).all() or np.any(x<0) for x in (na,nb,za,zb)):
        raise ValueError('finite nonnegative counts and activities are required')
    if np.any((na>0)&(za==0)) or np.any((nb>0)&(zb==0)):
        raise ValueError('current state lies outside a reservoir support')
    if np.any((nb>0)&(za==0)) or np.any((na>0)&(zb==0)):
        return -math.inf
    active=(za>0)&(zb>0)
    return float((nb[active]-na[active])@(np.log(za[active])-np.log(zb[active])))


def osmotic_phase_probabilities(temperature,pressure_pa,host_free_energies_kelvin,volumes_a3,
                                log_grand_partitions,degeneracies=None):
    """Normalize fixed-phase osmotic weights from independently supplied host free energies."""
    from scipy.special import logsumexp

    temp=_positive(temperature,'temperature')
    pressure=float(pressure_pa)
    f,v,ln_xi=(np.asarray(x,float) for x in (host_free_energies_kelvin,volumes_a3,log_grand_partitions))
    g=np.ones(f.shape) if degeneracies is None else np.asarray(degeneracies,float)
    if (f.ndim!=1 or not f.size or any(x.shape!=f.shape or not np.isfinite(x).all() for x in (f,v,ln_xi,g))
            or not math.isfinite(pressure) or pressure<0 or np.any(v<=0) or np.any(g<=0)):
        raise ValueError('finite free energies and positive per-phase volumes and degeneracies are required')
    log_weight=ln_xi+np.log(g)-(f+pressure*v*(1e-30/KB))/temp
    return np.exp(log_weight-logsumexp(log_weight))


class GrandCanonicalReplicaExchange:
    """Independent fixed-cell walkers with neighbor exchanges of activity labels.

    All walkers share one target instance, temperature and molecular measure.
    A round saves samples under their pre-exchange label. Exchange rounds use a
    fixed alternating schedule that depends on completed moves, never on time.
    """
    def __init__(self,walkers,seed=12345):
        self.walkers=list(walkers)
        if len(self.walkers)<2 or len({id(s) for s in self.walkers})!=len(self.walkers):
            raise ValueError('provide at least two distinct walkers')
        first=self.walkers[0]
        definitions=[m.to_dict() for m in first.species]
        for s in self.walkers:
            if (type(s) is not type(first) or s.potential is not first.potential
                    or s.target_potential is not first.target_potential
                    or s.temperature!=first.temperature or not np.array_equal(s.cell,first.cell)
                    or getattr(s,'count_bias',None) is not None
                    or [m.to_dict() for m in s.species]!=definitions):
                raise ValueError('activity exchange requires the same unbiased physical model')
            if hasattr(s,'_fingerprint') and s._fingerprint()!=first._fingerprint():
                raise ValueError('activity exchange requires identical native reciprocal Hamiltonians')
            if np.any((s.activities==0)&(s.counts>0)):
                raise ValueError('occupied zero-activity component')
        self.activities=np.array([s.activities for s in self.walkers])
        self.activities.flags.writeable=False
        self.occupants=list(range(len(self.walkers)))
        self.rng=np.random.default_rng(seed)
        self.round=0
        self.attempts=np.zeros(len(self.walkers)-1,dtype=int)
        self.accepts=np.zeros_like(self.attempts)

    @staticmethod
    def _set_activity(sim,values):
        if hasattr(sim,'set_activities'):
            sim.set_activities(values)
        else:
            sim.activities=values.copy()

    def advance(self,steps,sample_every=10,record_energy=False):
        steps,sample_every=int(steps),int(sample_every)
        if not 1<=sample_every<=steps or steps%sample_every:
            raise ValueError('exchange steps must be a positive multiple of sample_every')
        samples=[self.walkers[w].run(steps,sample_every=sample_every,record_energy=record_energy)
                 for w in self.occupants]
        for i in range(self.round%2,len(self.walkers)-1,2):
            j=i+1
            a,b=(self.walkers[self.occupants[label]] for label in (i,j))
            ratio=activity_exchange_log_ratio(a.counts,b.counts,self.activities[i],self.activities[j])
            self.attempts[i]+=1
            if _accept(ratio,self.rng):
                self._set_activity(a,self.activities[j])
                self._set_activity(b,self.activities[i])
                self.occupants[i],self.occupants[j]=self.occupants[j],self.occupants[i]
                self.accepts[i]+=1
        self.round+=1
        return samples


class TransitionMatrix:
    """Finite nearest-neighbour macrostate TMMC with explicit bias lifecycle.

    C[n,m] accumulates expected UNBIASED transitions. Row normalization removes
    the biased visitation frequencies. The reversible birth-death stationary
    distribution follows log(pi[n+1]/pi[n])=log(T[n,n+1]/T[n+1,n]).
    Missing edges raise instead of fabricating probabilities with pseudocounts.
    """
    def __init__(self, minimum, maximum):
        if int(minimum) != minimum or int(maximum) != maximum or maximum <= minimum:
            raise ValueError("TMMC requires integer maximum > minimum")
        self.minimum, self.maximum = int(minimum), int(maximum)
        size = self.maximum-self.minimum+1
        self.collection = np.zeros((size, size), dtype=float)
        self.log_bias = np.zeros(size, dtype=float)
        self.frozen = False

    def _index(self, count):
        if int(count) != count or not self.minimum <= count <= self.maximum:
            raise ValueError("macrostate outside TMMC window")
        return int(count)-self.minimum

    def record(self, old_count, new_count, unbiased_acceptance):
        i = self._index(old_count)
        a = float(unbiased_acceptance)
        if not math.isfinite(a) or not 0 <= a <= 1:
            raise ValueError("unbiased acceptance must lie in [0,1]")
        if int(new_count) != new_count or abs(new_count-old_count) > 1:
            raise ValueError("this TMMC controller supports only nearest-neighbour macrostate moves")
        if not self.minimum <= new_count <= self.maximum or old_count == new_count:
            self.collection[i, i] += 1
            return
        j = self._index(new_count)
        self.collection[i, j] += a
        self.collection[i, i] += 1-a

    def accept(self, old_count, new_count, log_unbiased_ratio, rng):
        if math.isnan(log_unbiased_ratio):
            raise FloatingPointError("NaN unbiased log acceptance")
        a = math.exp(min(0.0, log_unbiased_ratio))
        self.record(old_count, new_count, a)
        if not self.minimum <= new_count <= self.maximum:
            return False
        i, j = self._index(old_count), self._index(new_count)
        return _accept(log_unbiased_ratio+self.log_bias[j]-self.log_bias[i], rng)

    def transition_probabilities(self):
        rows = self.collection.sum(axis=1)
        if np.any(rows == 0):
            raise RuntimeError("not every macrostate has transition observations")
        return self.collection/rows[:, None]

    def stationary_log_probabilities(self):
        t = self.transition_probabilities()
        logs = np.zeros(len(t))
        for i in range(len(t)-1):
            if t[i, i+1] <= 0 or t[i+1, i] <= 0:
                raise RuntimeError("TMMC transition graph is not connected by observed edges")
            logs[i+1] = logs[i]+math.log(t[i, i+1])-math.log(t[i+1, i])
        maximum = max(logs)
        return logs-maximum-math.log(float(np.exp(logs-maximum).sum()))

    def update_bias(self):
        if self.frozen:
            raise RuntimeError("production bias is frozen")
        self.log_bias = -self.stationary_log_probabilities()
        self.log_bias -= self.log_bias[0]
        return self.log_bias.copy()

    def freeze(self):
        self.frozen = True
        self.log_bias.setflags(write=False)
        return self.log_bias.copy()

    def reweight(self, counts, values=None):
        """Return unbiased histogram, or weighted scalar/vector observable mean."""
        if not self.frozen:
            raise RuntimeError("freeze bias before collecting/reweighting production samples")
        raw = np.asarray(counts)
        if raw.ndim != 1 or len(raw) == 0:
            raise ValueError("counts must be a nonempty one-dimensional series")
        indices = np.array([self._index(n) for n in raw], int)
        logw = -self.log_bias[indices]
        weights = np.exp(logw-max(logw)); weights /= weights.sum()
        if values is None:
            return np.bincount(indices, weights=weights, minlength=len(self.log_bias))
        values = np.asarray(values, float)
        if values.ndim == 0 or len(values) != len(indices) or not np.isfinite(values).all():
            raise ValueError("values must be finite and aligned with counts")
        return np.tensordot(weights, values, axes=(0, 0))
