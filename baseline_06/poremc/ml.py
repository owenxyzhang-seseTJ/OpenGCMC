"""Total-energy ML potential adapters with explicit units and many-body deltas.

A nonlocal ML model cannot safely be treated as a sum of new-atom site energies.
This adapter recomputes the total combined system and subtracts the unchanged
background. It is slow but correct for any deterministic total-energy model,
including non-additive framework relaxation corrections if the model itself
returns a well-defined energy of the specified coordinates.
"""
from dataclasses import dataclass
import numpy as np
from .units import KJ_MOL_TO_K
EV_TO_K = 1.602176634e-19 / 1.380649e-23

@dataclass
class AtomicConfiguration:
    positions: np.ndarray
    atomic_numbers: np.ndarray
    cell: np.ndarray

@dataclass
class TotalEnvironment:
    entries: list
    positions: np.ndarray
    numbers: np.ndarray
    energy: float

class CallableTotalPotential:
    """energy_function(AtomicConfiguration)->scalar in declared unit.

    The function must include all desired intra-, guest-host and guest-guest
    interactions. No classical term is added implicitly. For a delta-trained
    model, supply a callback that explicitly adds its classical baseline.
    Full-state energies must be mutually consistent across molecule counts.
    """
    def __init__(self,species,cell,atomic_numbers,energy_function,unit='eV',
                 host_positions=None,host_atomic_numbers=None):
        self.species=list(species); self.cell=np.asarray(cell,float)
        if len(atomic_numbers)!=len(species): raise ValueError('atomic_numbers per species required')
        self.atomic_numbers=[np.asarray(z,int) for z in atomic_numbers]
        for m,z in zip(species,self.atomic_numbers):
            if z.shape!=(m.n_atoms,) or (z<1).any(): raise ValueError('invalid molecular atomic_numbers')
        self.host_positions=np.asarray(host_positions if host_positions is not None else [],float).reshape(-1,3)
        self.host_numbers=np.asarray(host_atomic_numbers if host_atomic_numbers is not None else [],int)
        if len(self.host_positions)!=len(self.host_numbers): raise ValueError('host position/element size mismatch')
        if unit not in ('eV','K','kJ/mol'): raise ValueError('energy unit must be eV, K or kJ/mol')
        self.factor={'eV':EV_TO_K,'K':1.0,'kJ/mol':KJ_MOL_TO_K}[unit]
        self.function=energy_function
        self.host_energy=self._energy(self.host_positions,self.host_numbers)
    def _energy(self,p,z):
        value=np.asarray(self.function(AtomicConfiguration(p,z,self.cell)))
        if value.size!=1: raise ValueError('ML callback must return one total scalar energy')
        value=float(value.reshape(-1)[0])*self.factor
        if np.isnan(value) or value==-np.inf: raise FloatingPointError('invalid total ML energy')
        return value
    def prepare_environment(self,entries):
        entries=list(entries)
        p=np.concatenate([self.host_positions]+[np.asarray(x) for _,x in entries])
        z=np.concatenate([self.host_numbers]+[self.atomic_numbers[s] for s,_ in entries])
        return TotalEnvironment(entries,p,z,self._energy(p,z))
    def _species_index(self,molecule):
        for i,m in enumerate(self.species):
            if m is molecule or m.name==molecule.name: return i
        raise ValueError('unknown molecule')
    def molecule_energy(self,coords,molecule,environment):
        env=environment if isinstance(environment,TotalEnvironment) else self.prepare_environment(environment)
        s=self._species_index(molecule)
        return self._energy(np.concatenate([env.positions,coords]),np.concatenate([env.numbers,self.atomic_numbers[s]]))-env.energy
    def incremental_energy(self,prefix,trials,atom_index,molecule,environment):
        env=environment if isinstance(environment,TotalEnvironment) else self.prepare_environment(environment)
        s=self._species_index(molecule); numbers=self.atomic_numbers[s]
        old_p=np.concatenate([env.positions,prefix]); old_z=np.concatenate([env.numbers,numbers[:atom_index]])
        old_e=self._energy(old_p,old_z)
        return np.array([self._energy(np.concatenate([old_p,p[None,:]]),np.r_[old_z,numbers[atom_index]])-old_e for p in trials])
    def total_energy(self,entries): return self.prepare_environment(entries).energy-self.host_energy

class TorchScriptEnergy:
    """Adapter for a USER-EXPORTED TorchScript model with one strict signature.

    forward(positions[N,3], atomic_numbers[N], cell[3,3]) -> scalar energy.
    Inputs use Angstrom, atomic numbers; output units are handled by the
    CallableTotalPotential wrapper. This does not claim arbitrary MACE/NequIP/
    Allegro checkpoints share that signature. Export an adapter for the model.
    """
    def __init__(self,path,device='cpu',dtype='float64'):
        try: import torch
        except ImportError as exc: raise RuntimeError('TorchScript requires the optional torch package') from exc
        if dtype not in ('float32','float64'): raise ValueError('unsupported Torch dtype')
        self.torch=torch; self.device=device; self.dtype=getattr(torch,dtype)
        self.model=torch.jit.load(str(path),map_location=device).eval()
    def __call__(self,config):
        t=self.torch
        with t.inference_mode():
            y=self.model(t.as_tensor(config.positions,dtype=self.dtype,device=self.device),
                         t.as_tensor(config.atomic_numbers,dtype=t.long,device=self.device),
                         t.as_tensor(config.cell,dtype=self.dtype,device=self.device))
        if y.numel()!=1: raise ValueError('TorchScript must return scalar total energy')
        return float(y.detach().cpu().item())
