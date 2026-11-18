"""Three-dimensional tin-foil Ewald, full reciprocal sum, double precision.

No PME or net-charge assumption is hidden here. Intermediate CBMC prefixes may
be charged; the uniform background term is included and telescopes. Public
ClassicalPotential disallows charged COMPLETE systems unless explicitly asked.
"""
from dataclasses import dataclass
import itertools
import math
import numpy as np
from .cell import validate_cell, cutoff_image
from .units import COULOMB_KA
try:
    from scipy.special import erfc
except ImportError:
    erfc = np.vectorize(math.erfc, otypes=[float])

@dataclass
class EwaldSnapshot:
    positions: np.ndarray
    charges: np.ndarray
    structure: np.ndarray
    total_charge: float

class Ewald:
    def __init__(self, cell, cutoff, alpha=0.25, kmax=6, background=True):
        self.cell,self.inverse,self.heights=validate_cell(cell,cutoff)
        self.volume=float(np.linalg.det(self.cell)); self.cutoff=float(cutoff)
        if not math.isfinite(alpha) or alpha<=0: raise ValueError("Ewald alpha must be positive")
        self.alpha=float(alpha); self.background=bool(background)
        km=np.broadcast_to(np.asarray(kmax,dtype=int),(3,))
        if (km<1).any(): raise ValueError("kmax must be positive")
        self.kmax=km.copy()
        ns=np.array([n for n in itertools.product(*(range(-k,k+1) for k in km)) if n!=(0,0,0)],float)
        self.k=2*np.pi*ns@self.inverse.T
        k2=np.einsum('ij,ij->i',self.k,self.k)
        self.coefficient=2*np.pi/self.volume*np.exp(-k2/(4*alpha*alpha))/k2

    def snapshot(self,positions,charges):
        p=np.asarray(positions,float).reshape(-1,3); q=np.asarray(charges,float).reshape(-1)
        if len(p)!=len(q) or not np.isfinite(p).all() or not np.isfinite(q).all():
            raise ValueError("invalid Ewald positions/charges")
        # Bound temporary phase-array memory for independent reference checks on
        # large frameworks, retaining the complete rectangular reciprocal set.
        structure=np.zeros(len(self.k),dtype=np.complex128)
        block=max(1,(1 << 20)//len(self.k))
        for start in range(0,len(q),block):
            end=start+block
            phase=np.exp(1j*(p[start:end]@self.k.T))
            structure+=(q[start:end,None]*phase).sum(axis=0)
        return EwaldSnapshot(p,q,structure,float(q.sum()))

    def energy(self,positions,charges):
        s=self.snapshot(positions,charges)
        if not self.background and abs(s.total_charge)>1e-9:
            raise ValueError("non-neutral Ewald requires an explicit uniform background")
        energy=0.0
        for i in range(len(s.charges)):
            d=cutoff_image(s.positions[i]-s.positions[:i],self.cell)
            r=np.linalg.norm(d,axis=1)
            active=(r<self.cutoff)&(s.charges[:i]*s.charges[i]!=0)
            if np.any(r[active]<1e-12): return float('inf')
            r=r[active]
            energy+=np.sum(s.charges[i]*s.charges[:i][active]*erfc(self.alpha*r)/r)
        energy+=np.sum(self.coefficient*np.abs(s.structure)**2)
        energy-=self.alpha/np.sqrt(np.pi)*np.dot(s.charges,s.charges)
        if self.background:
            energy-=np.pi*s.total_charge*s.total_charge/(2*self.alpha*self.alpha*self.volume)
        return float(energy*COULOMB_KA)

    def insertion_energies(self,snapshot,trial_positions,charge,include_real=True):
        """Exact finite-Ewald Hamiltonian difference for adding one atom."""
        p=np.asarray(trial_positions,float).reshape(-1,3); q=float(charge)
        if q==0: return np.zeros(len(p))
        if not self.background and abs(snapshot.total_charge+q)>1e-9:
            raise ValueError("charged Ewald state without background")
        phase=np.exp(1j*(p@self.k.T))
        energy=np.sum(self.coefficient[None,:]*(2*q*np.real(snapshot.structure.conj()[None,:]*phase)+q*q),axis=1)
        energy-=self.alpha/np.sqrt(np.pi)*q*q
        if self.background:
            energy-=np.pi*(2*snapshot.total_charge*q+q*q)/(2*self.alpha*self.alpha*self.volume)
        if include_real and len(snapshot.charges):
            d=cutoff_image(p[:,None,:]-snapshot.positions[None,:,:],self.cell)
            r=np.linalg.norm(d,axis=2)
            active=(r<self.cutoff)&(snapshot.charges[None,:]*q!=0)
            safe=np.where(active & (r>1e-12),r,1.0)
            energy+=np.sum(np.where(active, q*snapshot.charges[None,:]*erfc(self.alpha*safe)/safe,0),axis=1)
            energy[np.any(active & (r<=1e-12),axis=1)]=np.inf
        return energy*COULOMB_KA
