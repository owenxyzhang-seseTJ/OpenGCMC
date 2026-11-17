"""Configurational-bias growth with explicitly normalized Rosenbluth weights.

Trial proposals are draws from Molecule's normalized reference measure. The
weight for every step is MEAN(exp(-incremental_energy/T)), so K=1 is ordinary
unbiased proposal Monte Carlo. Retracing includes the old position plus K-1
fresh proposals at every growth stage. All arithmetic is in the log domain.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from .topology import Molecule, propose_atom, random_rotation


@dataclass
class GrowthResult:
    positions: np.ndarray
    log_weight: float
    energy: float
    valid: bool = True

    @property
    def log_rosenbluth(self):
        return self.log_weight


def logsumexp(x):
    x = np.asarray(x, float)
    m = float(np.max(x))
    if m == -math.inf:
        return -math.inf
    if not math.isfinite(m):
        raise FloatingPointError("nonfinite positive CBMC weight")
    return m + math.log(float(np.exp(x-m).sum()))


def _energies(potential, prefix, trials, i, molecule, environment):
    values = np.asarray(potential.incremental_energy(prefix, trials, i, molecule, environment), float)
    if values.shape != (len(trials),):
        raise ValueError("incremental_energy must return one energy per trial")
    if np.any(np.isnan(values)) or np.any(np.isneginf(values)):
        raise FloatingPointError("energy must not be NaN or negative infinity")
    return values


def molecule_energy(potential, coordinates, molecule, environment):
    if hasattr(potential, "molecule_energy"):
        return float(potential.molecule_energy(coordinates, molecule, environment))
    return sum(float(_energies(potential, coordinates[:i], coordinates[i:i+1], i,
                               molecule, environment)[0]) for i in range(molecule.n_atoms))


class CBMC:
    def __init__(self, temperature, cell, potential, trials=8, root_proposal=None):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)
        self.cell = np.asarray(cell, float)
        self.potential = potential
        self.trials = trials
        self.root_proposal = root_proposal
        if root_proposal is not None and not np.array_equal(root_proposal.cell,self.cell):
            raise ValueError('proposal and CBMC cells differ')

    def _k(self, i):
        k = int(self.trials if np.ndim(self.trials) == 0 else self.trials[i])
        if k < 1:
            raise ValueError("CBMC trial counts must be positive")
        return k

    def _rigid_trials(self, molecule, rng, k):
        body = np.zeros((1, 3)) if molecule.rigid_positions is None else molecule.rigid_positions
        if k == 0:
            return np.empty((0, molecule.n_atoms, 3))
        translations = rng.random((k, 3)) @ self.cell if self.root_proposal is None else self.root_proposal.sample(rng,k)
        if molecule.n_atoms == 1:
            return translations[:, None, :]
        return np.stack([(body if self.root_proposal is None else body-body[0]) @ random_rotation(rng).T + translations[j] for j in range(k)])

    def _whole_energies(self, coordinates, molecule, environment):
        if hasattr(self.potential, "molecule_energies"):
            energies = np.asarray(self.potential.molecule_energies(coordinates, molecule, environment), float)
        else:
            energies = np.array([molecule_energy(self.potential, x, molecule, environment) for x in coordinates])
        if energies.shape != (len(coordinates),) or np.isnan(energies).any() or np.isneginf(energies).any():
            raise FloatingPointError("invalid whole molecule trial energies")
        return energies

    def grow(self, molecule: Molecule, environment, rng, prefix=None, start=0):
        if molecule.is_rigid:
            if start != 0:
                raise ValueError("rigid growth starts at atom zero")
            k = self._k(0)
            trial_coords = self._rigid_trials(molecule, rng, k)
            energies = self._whole_energies(trial_coords, molecule, environment)
            lw = -energies / self.temperature
            if self.root_proposal is not None:lw -= self.root_proposal.log_relative_density(trial_coords[:,0,:])
            ls = logsumexp(lw)
            if ls == -math.inf:
                return GrowthResult(trial_coords[0].copy(), -math.inf, math.inf, False)
            idx = int(rng.choice(k, p=np.exp(lw-ls)))
            return GrowthResult(trial_coords[idx].copy(), ls-math.log(k), float(energies[idx]))
        if start < 0 or start >= molecule.n_atoms:
            raise ValueError("invalid CBMC growth start")
        positions = np.empty((molecule.n_atoms, 3))
        if start:
            if prefix is None or np.asarray(prefix).shape[0] < start:
                raise ValueError("growth prefix does not cover fixed atoms")
            positions[:start] = np.asarray(prefix)[:start]
        log_w, energy = 0.0, 0.0
        for i in range(start, molecule.n_atoms):
            k = self._k(i)
            candidates = (self.root_proposal.sample(rng,k) if i==0 and self.root_proposal is not None
                          else propose_atom(molecule, positions[:i], i, k, self.cell, rng))
            values = _energies(self.potential, positions[:i], candidates, i, molecule, environment)
            lw = -values / self.temperature
            if i==0 and self.root_proposal is not None:lw -= self.root_proposal.log_relative_density(candidates)
            ls = logsumexp(lw)
            if ls == -math.inf:
                positions[i:] = np.nan
                return GrowthResult(positions, -math.inf, math.inf, False)
            selected = int(rng.choice(k, p=np.exp(lw-ls)))
            positions[i] = candidates[selected]
            energy += float(values[selected])
            log_w += ls - math.log(k)
        return GrowthResult(positions, log_w, energy)

    def retrace(self, molecule: Molecule, old_positions, environment, rng, start=0):
        old = np.asarray(old_positions, float)
        if old.shape != (molecule.n_atoms, 3):
            raise ValueError("old molecule positions have wrong shape")
        if molecule.is_rigid:
            if start != 0:
                raise ValueError("rigid retracing starts at atom zero")
            k = self._k(0)
            trial_coords = self._rigid_trials(molecule, rng, k-1)
            trial_coords = np.concatenate([old[None, :, :], trial_coords], axis=0)
            energies = self._whole_energies(trial_coords, molecule, environment)
            lw=-energies/self.temperature
            if self.root_proposal is not None:lw-=self.root_proposal.log_relative_density(trial_coords[:,0,:])
            ls = logsumexp(lw)
            return GrowthResult(old.copy(), ls-math.log(k), float(energies[0]),
                                math.isfinite(ls) and math.isfinite(float(energies[0])))
        if start < 0 or start >= molecule.n_atoms:
            raise ValueError("invalid CBMC retrace start")
        log_w, energy = 0.0, 0.0
        for i in range(start, molecule.n_atoms):
            k = self._k(i)
            fresh = (self.root_proposal.sample(rng,k-1) if i==0 and self.root_proposal is not None
                     else propose_atom(molecule, old[:i], i, k-1, self.cell, rng))
            candidates = np.concatenate([old[i:i+1], fresh], axis=0)
            values = _energies(self.potential, old[:i], candidates, i, molecule, environment)
            lw=-values/self.temperature
            if i==0 and self.root_proposal is not None:lw-=self.root_proposal.log_relative_density(candidates)
            ls = logsumexp(lw)
            log_w += ls - math.log(k)
            energy += float(values[0])
        return GrowthResult(old.copy(), log_w, energy, math.isfinite(log_w) and math.isfinite(energy))
