"""Grand canonical Monte Carlo with rigid/flexible CBMC and exact correction.

A step is one attempted move, not an ill-defined cycle. Move and species
probabilities are fixed; an empty-species move is a self transition. Optional
step-size adaptation happens only during burn-in and is frozen in production.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
import math
import time
from pathlib import Path
import numpy as np
from .topology import Molecule, local_rotation
from .cbmc import CBMC, molecule_energy
from .statistics import summarize


@dataclass
class Particle:
    species: int
    positions: np.ndarray

    def to_dict(self):
        return {"species": self.species, "positions": self.positions.tolist()}


class ZeroPotential:
    """Exactly noninteracting reference, useful for ensemble validation."""
    def incremental_energy(self, prefix, trials, atom_index, molecule, environment):
        return np.zeros(len(trials))
    def molecule_energy(self, coordinates, molecule, environment):
        return 0.0
    def molecule_energies(self, coordinates, molecule, environment):
        return np.zeros(len(coordinates))
    def total_energy(self, entries):
        return 0.0


class GCMC:
    MOVE_NAMES = ("insert", "delete", "translate", "rotate", "regrow")

    def __init__(self, species, potential, cell, temperature, activities,
                 seed=12345, trials=8, move_weights=None,
                 species_probabilities=None, max_translation=1.0,
                 max_rotation=0.5, target_potential=None, max_particles=100000, root_proposal=None, count_bias=None):
        self.count_bias = count_bias
        self.species = list(species)
        if not self.species:
            raise ValueError("at least one molecular species is required")
        self.potential = potential
        self.target_potential = target_potential
        self.cell = np.asarray(cell, float).copy()
        if self.cell.shape == (3,):
            self.cell = np.diag(self.cell)
        if self.cell.shape != (3, 3) or not np.isfinite(self.cell).all():
            raise ValueError("cell must be a finite 3x3 row-lattice matrix")
        self.volume = float(np.linalg.det(self.cell))
        if self.volume <= 0:
            raise ValueError("cell must be right handed with positive volume")
        self.inverse_cell = np.linalg.inv(self.cell)
        self.temperature = float(temperature)
        if self.temperature <= 0 or not math.isfinite(self.temperature):
            raise ValueError("temperature must be positive and finite")
        self.activities = np.asarray(activities, float).reshape(-1)
        if self.activities.shape != (len(species),) or np.any(self.activities < 0) or not np.isfinite(self.activities).all():
            raise ValueError("one finite nonnegative activity per species is required")
        self.rng = np.random.default_rng(seed)
        self.trials = int(trials) if np.ndim(trials)==0 else list(map(int,trials))
        if np.any(np.asarray(self.trials) < 1):
            raise ValueError("all trial counts must be positive")
        if np.ndim(self.trials) != 0 and len(self.trials) < max(m.n_atoms for m in self.species):
            raise ValueError("per-atom trial counts must cover every molecular growth index")
        self.root_proposal=root_proposal
        self.cbmc = CBMC(self.temperature, self.cell, potential, self.trials,root_proposal)
        weights = {"insert": .25, "delete": .25, "translate": .2, "rotate": .1, "regrow": .2}
        if move_weights is not None:
            if not set(move_weights).issubset(self.MOVE_NAMES):
                raise ValueError("unknown move name")
            weights = {m: float(move_weights.get(m, 0)) for m in self.MOVE_NAMES}
        w = np.array([weights[m] for m in self.MOVE_NAMES], float)
        if np.any(w<0) or not np.isfinite(w).all() or w.sum()<=0:
            raise ValueError("invalid move weights")
        if bool(w[0]) != bool(w[1]):
            raise ValueError("insertion and deletion must both have positive weight or both be disabled")
        self.move_probabilities = w / w.sum()
        if species_probabilities is None:
            species_probabilities = np.ones(len(species))
        sp = np.asarray(species_probabilities, float)
        if sp.shape != (len(species),) or np.any(sp<=0) or not np.isfinite(sp).all():
            raise ValueError("species probabilities must be finite and positive")
        self.species_probabilities = sp/sp.sum()
        self.max_translation = float(max_translation)
        self.max_rotation = float(max_rotation)
        if not 0 < self.max_translation or not 0 < self.max_rotation <= math.pi:
            raise ValueError("positive displacement and rotation in (0,pi] required")
        self.max_particles = int(max_particles)
        if self.max_particles < 1:
            raise ValueError("max_particles must be positive")
        self.particles: list[Particle] = []
        self.step_count = 0
        self.counters = {m: {"attempted": 0, "accepted": 0, "stage1_passed": 0,
                            "stage2_attempted": 0, "stage2_passed": 0} for m in self.MOVE_NAMES}
        self.count_trace: list[list[int]] = []
        self.energy_trace: list[float] = []
        self.sample_steps: list[int] = []
        self.adaptation_updates = 0

    @property
    def counts(self):
        return np.bincount([p.species for p in self.particles], minlength=len(self.species))

    def entries(self, exclude=None):
        return [(p.species, p.positions) for j, p in enumerate(self.particles) if j != exclude]

    @staticmethod
    def _prepare(potential, entries):
        return potential.prepare_environment(entries) if hasattr(potential, "prepare_environment") else entries

    def _wrap_molecule(self, coordinates):
        """Wrap only the root; keep intramolecular vectors unwrapped and intact."""
        return coordinates - np.floor(coordinates[0] @ self.inverse_cell) @ self.cell

    def add_particle(self, species, positions):
        species = int(species)
        if not 0 <= species < len(self.species):
            raise ValueError("invalid species index")
        positions = np.asarray(positions, float)
        if positions.shape != (self.species[species].n_atoms, 3) or not np.isfinite(positions).all():
            raise ValueError("invalid particle coordinates")
        self.species[species].validate_positions(positions)
        if len(self.particles) >= self.max_particles:
            raise RuntimeError("max_particles exceeded; capacity is not a physical occupancy constraint")
        coordinates = self._wrap_molecule(positions).copy()
        self._validate_current_energy(self.species[species], coordinates, self.entries())
        self.particles.append(Particle(species, coordinates))

    def _validate_current_energy(self, molecule, coordinates, entries):
        """Initialization/restart states must have finite surrogate and target U."""
        potentials = [self.potential]
        if self.target_potential is not None:
            potentials.append(self.target_potential)
        for potential in potentials:
            environment = self._prepare(potential, entries)
            if not math.isfinite(molecule_energy(potential, coordinates, molecule, environment)):
                raise ValueError("initial or restored molecule has nonfinite energy under the configured Hamiltonian")

    def _accept_log(self, log_ratio):
        if math.isnan(log_ratio):
            raise FloatingPointError("NaN Monte Carlo acceptance ratio")
        return math.log(max(float(self.rng.random()), np.finfo(float).tiny)) < min(0.0, log_ratio)

    def _accept_candidate(self, move, first_log_ratio, molecule, old, new, entries, env, surrogate_delta=None):
        if self.count_bias is not None:
            before = self.counts
            after = before.copy()
            if move in ("insert", "delete"):
                sid = next(i for i, m in enumerate(self.species) if m is molecule)
                after[sid] += 1 if move == "insert" else -1
            old_bias = float(self.count_bias(before))
            new_bias = float(self.count_bias(after))
            if not math.isfinite(old_bias):
                raise ValueError("current state is outside the frozen bias window")
            if not math.isfinite(new_bias):
                return False
            first_log_ratio -= new_bias - old_bias
        if not self._accept_log(first_log_ratio):
            return False
        self.counters[move]["stage1_passed"] += 1
        if self.target_potential is None:
            return True
        self.counters[move]["stage2_attempted"] += 1
        target_env = self._prepare(self.target_potential, entries)
        if surrogate_delta is None:
            old_s = molecule_energy(self.potential, old, molecule, env) if old is not None else 0.0
            new_s = molecule_energy(self.potential, new, molecule, env) if new is not None else 0.0
        else:
            old_s=0.;new_s=float(surrogate_delta)
        old_t = molecule_energy(self.target_potential, old, molecule, target_env) if old is not None else 0.0
        new_t = molecule_energy(self.target_potential, new, molecule, target_env) if new is not None else 0.0
        if not math.isfinite(old_s) or not math.isfinite(new_s) or not math.isfinite(old_t):
            raise FloatingPointError("current or surrogate selected configuration has nonfinite energy")
        if math.isnan(new_t) or new_t == -math.inf:
            raise FloatingPointError("target candidate has invalid energy")
        correction = -((new_t-old_t)-(new_s-old_s))/self.temperature
        if not self._accept_log(correction):
            return False
        self.counters[move]["stage2_passed"] += 1
        return True

    def step(self, move=None, species_index=None):
        """Attempt a move. Explicit arguments are for controlled tests/schedules.

        Acceptance selection factors always use configured move probabilities.
        A manually altered scheduling distribution must match those weights.
        """
        if move is None:
            mi = int(self.rng.choice(len(self.MOVE_NAMES), p=self.move_probabilities))
            move = self.MOVE_NAMES[mi]
        elif move not in self.MOVE_NAMES:
            raise ValueError("unknown move")
        if self.move_probabilities[self.MOVE_NAMES.index(move)] == 0:
            raise ValueError("cannot manually attempt a zero-probability move")
        s = int(self.rng.choice(len(self.species), p=self.species_probabilities)) if species_index is None else int(species_index)
        if not 0 <= s < len(self.species):
            raise ValueError("invalid species index")
        self.step_count += 1
        self.counters[move]["attempted"] += 1
        molecule = self.species[s]
        members = [i for i,p in enumerate(self.particles) if p.species == s]
        n = len(members)
        activity = float(self.activities[s])
        if move == "insert":
            if activity == 0:
                return False
            if len(self.particles) >= self.max_particles:
                # Silently rejecting would change the target grand ensemble.
                raise RuntimeError("max_particles reached; increase capacity to preserve the target ensemble")
            entries = self.entries()
            env = self._prepare(self.potential, entries)
            growth = self.cbmc.grow(molecule, env, self.rng)
            if not growth.valid:
                return False
            ratio = (math.log(activity)+math.log(self.volume)-math.log(n+1)+growth.log_weight
                     +math.log(self.move_probabilities[1]/self.move_probabilities[0]))
            accepted = self._accept_candidate(move, ratio, molecule, None, growth.positions, entries, env, growth.energy)
            if accepted:
                self.particles.append(Particle(s, self._wrap_molecule(growth.positions)))
        else:
            if n == 0:
                return False
            index = int(self.rng.choice(members))
            old = self.particles[index].positions
            entries = self.entries(exclude=index)
            env = self._prepare(self.potential, entries)
            if move == "delete":
                retrace = self.cbmc.retrace(molecule, old, env, self.rng)
                if not retrace.valid:
                    raise FloatingPointError("cannot retrace an invalid current molecule")
                ratio = math.inf if activity==0 else (math.log(n)-math.log(activity)-math.log(self.volume)-retrace.log_weight
                         + math.log(self.move_probabilities[0]/self.move_probabilities[1]))
                accepted = self._accept_candidate(move, ratio, molecule, old, None, entries, env, -retrace.energy)
                if accepted:
                    del self.particles[index]
            elif move == "regrow":
                start = 0 if molecule.is_rigid else int(self.rng.integers(molecule.n_atoms))
                growth = self.cbmc.grow(molecule, env, self.rng, prefix=old, start=start)
                if not growth.valid:
                    return False
                retrace = self.cbmc.retrace(molecule, old, env, self.rng, start=start)
                if not retrace.valid:
                    raise FloatingPointError("cannot regrow an invalid current molecule")
                ratio = growth.log_weight-retrace.log_weight
                accepted = self._accept_candidate(move, ratio, molecule, old, growth.positions, entries, env, growth.energy-retrace.energy)
                if accepted:
                    self.particles[index].positions = self._wrap_molecule(growth.positions)
            else:
                if move == "translate":
                    candidate = old + self.rng.uniform(-self.max_translation, self.max_translation, 3)
                else:
                    if molecule.n_atoms == 1:
                        return False
                    center = old.mean(axis=0)
                    candidate = (old-center) @ local_rotation(self.rng, self.max_rotation).T + center
                old_energy = molecule_energy(self.potential, old, molecule, env)
                new_energy = molecule_energy(self.potential, candidate, molecule, env)
                ratio = -(new_energy-old_energy)/self.temperature
                accepted = self._accept_candidate(move, ratio, molecule, old, candidate, entries, env, new_energy-old_energy)
                if accepted:
                    self.particles[index].positions = self._wrap_molecule(candidate)
        if accepted:
            self.counters[move]["accepted"] += 1
        return bool(accepted)

    def total_energy(self):
        potential = self.target_potential if self.target_potential is not None else self.potential
        entries = self.entries()
        if hasattr(potential, "total_energy"):
            return float(potential.total_energy(entries))
        total = 0.0
        for i, p in enumerate(self.particles):
            env = self._prepare(potential, entries[:i])
            total += molecule_energy(potential, p.positions, self.species[p.species], env)
        return total

    def _adapt(self, before):
        self.adaptation_updates += 1
        gain = min(0.25, 1/math.sqrt(self.adaptation_updates))
        for move, name, target in (("translate", "max_translation", .35), ("rotate", "max_rotation", .35)):
            attempts = self.counters[move]["attempted"]-before[move][0]
            accepted = self.counters[move]["accepted"]-before[move][1]
            if attempts:
                value = getattr(self, name)*math.exp(gain*(accepted/attempts-target))
                limit = math.pi if move=="rotate" else float(np.linalg.norm(self.cell,axis=1).min())
                setattr(self, name, min(limit, max(1e-6, value)))

    def run(self, steps, burnin=0, sample_every=10, adapt=False, record_energy=True):
        steps, burnin, sample_every = int(steps), int(burnin), int(sample_every)
        if steps < 0 or burnin < 0 or sample_every < 1:
            raise ValueError("invalid run lengths or sampling interval")
        wall_start = time.perf_counter()
        for block in range(0, burnin, 100):
            before = {m:(c["attempted"],c["accepted"]) for m,c in self.counters.items()}
            for _ in range(min(100, burnin-block)):
                self.step()
            if adapt:
                self._adapt(before)
        production_start = time.perf_counter()
        trace, energies, sample_steps = [], [], []
        for i in range(steps):
            self.step()
            if (i+1)%sample_every == 0:
                count = self.counts.tolist()
                trace.append(count); self.count_trace.append(count)
                sample_steps.append(self.step_count); self.sample_steps.append(self.step_count)
                if record_energy:
                    energy = self.total_energy()
                    energies.append(energy); self.energy_trace.append(energy)
        production_seconds = time.perf_counter()-production_start
        elapsed = time.perf_counter()-wall_start
        result = {"steps": steps, "burnin": burnin, "sample_every": sample_every,
                  "production_seconds": production_seconds, "elapsed_seconds": elapsed,
                  "attempted_moves_per_second": steps/production_seconds if production_seconds>0 else 0,
                  "counts": trace, "energies_kelvin": energies, "sample_steps": sample_steps,
                  "acceptance": {m:dict(c, rate=c["accepted"]/c["attempted"] if c["attempted"] else 0.0)
                                 for m,c in self.counters.items()},
                  "species_statistics": {mol.name: summarize(np.array(trace)[:,j],production_seconds)
                                         for j,mol in enumerate(self.species)} if trace else {},
                  "energy_statistics": summarize(energies,production_seconds) if energies else {},
                  "final_counts": self.counts.tolist(), "adaptation_frozen": True,
                  "max_translation": self.max_translation, "max_rotation": self.max_rotation,
                  "delayed_acceptance": self.target_potential is not None}
        return result

    def checkpoint(self):
        from .free_energy import HarmonicCountBias
        if self.count_bias is not None and not isinstance(self.count_bias, HarmonicCountBias):
            raise TypeError("checkpoint requires a serializable HarmonicCountBias")
        return {"count_bias": None if self.count_bias is None else self.count_bias.__dict__, "schema": "poremc-gcmc-checkpoint-v1", "root_proposal":self.root_proposal.to_dict() if self.root_proposal is not None else None, "species": [s.to_dict() for s in self.species],
                "cell": self.cell.tolist(), "temperature": self.temperature, "activities": self.activities.tolist(),
                "trials": self.trials, "move_weights": dict(zip(self.MOVE_NAMES, self.move_probabilities.tolist())),
                "species_probabilities": self.species_probabilities.tolist(),
                "max_translation": self.max_translation, "max_rotation": self.max_rotation,
                "max_particles": self.max_particles, "particles": [p.to_dict() for p in self.particles],
                "rng_state": self.rng.bit_generator.state, "step_count": self.step_count,
                "counters": self.counters, "count_trace": self.count_trace,
                "energy_trace": self.energy_trace, "sample_steps": self.sample_steps,
                "adaptation_updates": self.adaptation_updates,
                "requires_target_potential": self.target_potential is not None}

    def save_checkpoint(self, path):
        path = Path(path)
        tmp = path.with_name(path.name+".tmp")
        tmp.write_text(json.dumps(self.checkpoint(), indent=2), encoding="utf8")
        tmp.replace(path)

    @classmethod
    def from_checkpoint(cls, path, potential, target_potential=None):
        """Restore exact RNG and state; caller supplies the identical Hamiltonian."""
        data = json.loads(Path(path).read_text(encoding="utf8"))
        if data.get("schema") != "poremc-gcmc-checkpoint-v1":
            raise ValueError("unsupported checkpoint schema")
        if data["requires_target_potential"] != (target_potential is not None):
            raise ValueError("checkpoint target-potential mode differs")
        from .proposals import VoxelProposal
        from .free_energy import HarmonicCountBias
        bias_data = data.get("count_bias")
        obj = cls([Molecule.from_dict(d) for d in data["species"]], potential, data["cell"],
                  data["temperature"], data["activities"], trials=data["trials"],
                  move_weights=data["move_weights"], species_probabilities=data["species_probabilities"],
                  max_translation=data["max_translation"], max_rotation=data["max_rotation"],
                  target_potential=target_potential, max_particles=data["max_particles"],root_proposal=VoxelProposal.from_dict(data.get("root_proposal")),
                  count_bias=None if bias_data is None else HarmonicCountBias(**bias_data))
        obj.particles = [Particle(int(p["species"]), np.asarray(p["positions"],float)) for p in data["particles"]]
        if len(obj.particles) > obj.max_particles:
            raise ValueError("checkpoint exceeds max_particles")
        for particle in obj.particles:
            if not 0 <= particle.species < len(obj.species):
                raise ValueError("checkpoint contains an invalid species index")
            obj.species[particle.species].validate_positions(particle.positions)
        for index, particle in enumerate(obj.particles):
            obj._validate_current_energy(obj.species[particle.species], particle.positions, obj.entries(exclude=index))
        obj.rng.bit_generator.state = data["rng_state"]
        obj.step_count = data["step_count"]
        obj.counters = data["counters"]
        obj.count_trace = data["count_trace"]
        obj.energy_trace = data["energy_trace"]
        obj.sample_steps = data["sample_steps"]
        obj.adaptation_updates = data["adaptation_updates"]
        return obj
