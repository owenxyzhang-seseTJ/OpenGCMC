"""Local molecular energy with native batched LJ/DSF and optional exact Ewald.

Incremental contributions partition the complete molecular energy. Every
internal interaction is included exactly when its largest atom index appears.
"""
from dataclasses import dataclass
from collections import OrderedDict
import hashlib
import math
import numpy as np
from .cell import validate_cell
from .electrostatics import Ewald, EwaldSnapshot, erfc
from .units import COULOMB_KA

@dataclass
class PreparedEnvironment:
    entries: list
    positions: np.ndarray
    sigma: np.ndarray
    epsilon: np.ndarray
    charge: np.ndarray
    evaluator: object
    ewald_snapshot: object = None

class ZeroPotential:
    """Analytic ideal-gas target, also useful to isolate topology sampling."""
    def prepare_environment(self,entries): return entries
    def incremental_energy(self,prefix,trials,atom_index,molecule,environment):
        return np.zeros(len(trials))
    def molecule_energy(self,coords,molecule,environment): return 0.0
    def total_energy(self,entries): return 0.0

class ClassicalPotential:
    def __init__(self,species,cell,cutoff,host_positions=None,host_sigma=None,
                 host_epsilon=None,host_charge=None,backend='native',
                 algorithm='cell',coulomb='none',alpha=0.25,kmax=6,
                 shifted=True,threads=1,allow_charge_background=False,
                 scale14_lj=1.0,scale14_coulomb=1.0):
        from .kernels import NativeEnvironment, ReferenceEnvironment
        self._prepared_cache=OrderedDict();self.environment_cache_stats={'hits':0,'builds':0}
        self.species=list(species); self.cell,self.inverse,_=validate_cell(cell,cutoff)
        self.cutoff=float(cutoff); self.coulomb=coulomb; self.alpha=float(alpha)
        self.shifted=bool(shifted); self.backend=backend; self.algorithm=algorithm
        self.threads=threads
        hmin=float(1.0 / np.linalg.norm(self.inverse,axis=0).max())
        for m in self.species:
            if m.is_rigid:
                rp=m.rigid_positions if m.rigid_positions is not None else np.zeros((1,3))
                bound=float(np.linalg.norm(rp[:,None,:]-rp[None,:,:],axis=2).max())
            else:
                distances=np.full((m.n_atoms,m.n_atoms),np.inf); np.fill_diagonal(distances,0)
                for i,parent in enumerate(m.parents):
                    if parent>=0: distances[i,parent]=distances[parent,i]=(m.bond_ranges[i,1] if getattr(m,'bond_ranges',None) is not None else m.bond_lengths[i])
                for k in range(m.n_atoms): distances=np.minimum(distances,distances[:,k,None]+distances[None,k,:])
                bound=float(distances.max())
            if bound+self.cutoff>=hmin:
                raise ValueError(f'{m.name}: molecule diameter upper bound + cutoff must be less than minimum face height; enlarge supercell')
        self.scale14_lj=float(scale14_lj); self.scale14_coulomb=float(scale14_coulomb)
        if coulomb not in ('none','dsf','ewald'): raise ValueError('coulomb must be none, dsf or ewald')
        if backend=='native': self.environment_class=NativeEnvironment
        elif backend=='reference': self.environment_class=ReferenceEnvironment
        elif backend=='cuda':
            from .kernels import CudaEnvironment
            self.environment_class=CudaEnvironment
        else: raise ValueError('unknown backend')
        self.host_positions=np.array(host_positions if host_positions is not None else [],float,copy=True).reshape(-1,3)
        n=len(self.host_positions)
        self.host_sigma=np.zeros(n) if host_sigma is None else np.array(host_sigma,float,copy=True)
        self.host_epsilon=np.zeros(n) if host_epsilon is None else np.array(host_epsilon,float,copy=True)
        self.host_charge=np.zeros(n) if host_charge is None else np.array(host_charge,float,copy=True)
        for a in [self.host_sigma,self.host_epsilon,self.host_charge]:
            if a.shape!=(n,) or not np.isfinite(a).all(): raise ValueError('invalid host parameters')
            a.flags.writeable=False
        self.host_positions.flags.writeable=False
        if (self.host_sigma<0).any() or (self.host_epsilon<0).any(): raise ValueError('negative host LJ parameters')
        if not allow_charge_background and coulomb=='ewald':
            if abs(self.host_charge.sum())>1e-8 or any(abs(np.sum(m.charge))>1e-8 for m in self.species):
                raise ValueError('charged host/molecule requires allow_charge_background=true; neutral molecular GCMC is the default')
        self.ewald=Ewald(self.cell,cutoff,alpha,kmax,background=True) if coulomb=='ewald' else None
        self.host_evaluator=self._new_evaluator(self.host_positions,self.host_sigma,self.host_epsilon,self.host_charge)
        self._distances={id(m):self._graph_distances(m) for m in self.species}
        self._exclusions={id(m):frozenset(tuple(sorted(pair)) for pair in (m.exclusions or [])) for m in self.species}
        self._host_snapshot=self.ewald.snapshot(self.host_positions,self.host_charge) if self.ewald is not None else None

    def _new_evaluator(self,p,s,e,q):
        return self.environment_class(p,s,e,q,self.cell,self.cutoff,
             coulomb=('ewald-real' if self.coulomb=='ewald' else self.coulomb), alpha=self.alpha,
             shifted=self.shifted,algorithm=self.algorithm,threads=self.threads)

    @staticmethod
    def _graph_distances(m):
        n=m.n_atoms; d=np.full((n,n),n+1,dtype=int); np.fill_diagonal(d,0)
        for i,p in enumerate(m.parents):
            if p>=0: d[i,p]=d[p,i]=1
        for k in range(n): d=np.minimum(d,d[:,k,None]+d[None,k,:])
        return d

    def prepare_environment(self,entries):
        entries=[(int(s),np.asarray(p,float)) for s,p in entries]
        digest=hashlib.sha256()
        for sid,xyz in entries:
            digest.update(np.int64(sid).tobytes());digest.update(str(xyz.shape).encode());digest.update(xyz.tobytes())
        key=digest.digest()
        if key in self._prepared_cache:
            self.environment_cache_stats['hits']+=1
            self._prepared_cache.move_to_end(key)
            return self._prepared_cache[key]
        self.environment_cache_stats['builds']+=1
        entries=[(sid,xyz.copy()) for sid,xyz in entries]
        p=np.concatenate([x for _,x in entries],axis=0) if entries else np.empty((0,3))
        def join(name): return np.concatenate([getattr(self.species[s],name) for s,_ in entries]) if entries else np.empty(0)
        s,e,q=join('sigma'),join('epsilon'),join('charge')
        evaluator=self._new_evaluator(p,s,e,q)
        snap=None
        if self.ewald is not None:
            guest_snap=self.ewald.snapshot(p,q)
            host_snap=self._host_snapshot
            snap=EwaldSnapshot(np.concatenate([self.host_positions,p]),np.concatenate([self.host_charge,q]),
                               host_snap.structure+guest_snap.structure,host_snap.total_charge+guest_snap.total_charge)
        prepared=PreparedEnvironment(entries,p,s,e,q,evaluator,snap)
        self._prepared_cache[key]=prepared
        if len(self._prepared_cache)>8:self._prepared_cache.popitem(last=False)
        return prepared

    def _excluded(self,m,i,j):
        if m.is_rigid: return True
        explicit=self._exclusions.get(id(m))
        if explicit is None:
            explicit=frozenset(tuple(sorted(pair)) for pair in (m.exclusions or []))
            self._exclusions[id(m)]=explicit
        return (min(i,j),max(i,j)) in explicit

    def _internal_pair(self,prefix,trials,i,m):
        out=np.zeros(len(trials))
        graph=self._distances.get(id(m))
        if graph is None: graph=self._graph_distances(m); self._distances[id(m)]=graph
        for j in range(i):
            # Molecules are stored unwrapped. Topological distances are not
            # replaced by periodic distances for bonded exclusions.
            excluded=self._excluded(m,i,j)
            if excluded and self.coulomb!='ewald': continue
            r=np.linalg.norm(trials-prefix[j],axis=1)
            slj=0 if excluded else (self.scale14_lj if graph[i,j]==3 else 1.0)
            sc=0 if excluded else (self.scale14_coulomb if graph[i,j]==3 else 1.0)
            sigma=(m.sigma[i]+m.sigma[j])/2; epsilon=np.sqrt(m.epsilon[i]*m.epsilon[j])
            mask=(r<self.cutoff)&(r>1e-12)
            if slj and epsilon and sigma:
                sr6=(sigma/np.where(mask,r,1))**6
                en=4*epsilon*(sr6*sr6-sr6)
                if self.shifted:
                    c=(sigma/self.cutoff)**6; en-=4*epsilon*(c*c-c)
                out+=slj*np.where(mask,en,0)
                out[r<=1e-12]=np.inf
            qq=m.charge[i]*m.charge[j]
            if self.coulomb=='dsf' and sc and qq:
                a=self.alpha; rc=self.cutoff
                slope=math.erfc(a*rc)/(rc*rc)+2*a/math.sqrt(math.pi)*math.exp(-(a*rc)**2)/rc
                safe=np.where(mask,r,1)
                u=erfc(a*safe)/safe-math.erfc(a*rc)/rc+(safe-rc)*slope
                out+=sc*COULOMB_KA*qq*np.where(mask,u,0)
                out[r<=1e-12]=np.inf
            if self.coulomb=='ewald' and qq:
                safe=np.where(mask,r,1.0)
                out+=COULOMB_KA*qq*np.where(mask,erfc(self.alpha*safe)/safe,0.0)
            if self.coulomb=='ewald' and qq and sc!=1:
                # Bare Coulomb exclusion is needed in addition to Ewald, to
                # remove same-molecule primary-cell interactions exactly.
                out-=(1-sc)*COULOMB_KA*qq/np.maximum(r,1e-12)
        return out

    @staticmethod
    def torsion_angles(a,b,c,d):
        b0=a-b; b1=c-b; b2=d-c
        norm=np.linalg.norm(b1,axis=-1,keepdims=True)
        if np.any(norm<1e-12): raise ValueError('undefined torsion: zero central bond')
        b1=b1/norm
        v=b0-np.sum(b0*b1,axis=-1,keepdims=True)*b1
        w=b2-np.sum(b2*b1,axis=-1,keepdims=True)*b1
        return np.arctan2(np.sum(np.cross(b1,v)*w,axis=-1),np.sum(v*w,axis=-1))

    def incremental_energy(self,prefix,trials,atom_index,molecule,environment):
        if not isinstance(environment,PreparedEnvironment): environment=self.prepare_environment(environment)
        m=molecule; i=int(atom_index); x=np.asarray(trials,float).reshape(-1,3); p=np.asarray(prefix,float).reshape(-1,3)
        if i!=len(p) or i>=m.n_atoms: raise ValueError('incremental atom must follow its prefix')
        out=self._host_energy(x[:,None,:],m.sigma[i:i+1],m.epsilon[i:i+1],m.charge[i:i+1])
        out=out+environment.evaluator.energy(x[:,None,:],m.sigma[i:i+1],m.epsilon[i:i+1],m.charge[i:i+1])
        out=out+self._internal_pair(p,x,i,m)
        if self.ewald is not None:
            snap=environment.ewald_snapshot
            if i:
                part=self.ewald.snapshot(p,m.charge[:i])
                snap=EwaldSnapshot(np.concatenate([snap.positions,p]),np.concatenate([snap.charges,m.charge[:i]]),
                                   snap.structure+part.structure,snap.total_charge+part.total_charge)
            out=out+self.ewald.insertion_energies(snap,x,m.charge[i],include_real=False)
        if not m.is_rigid and i>0 and (m.bond_force_constants[i]!=0 or m.angle_force_constants[i]!=0):
            parent=int(m.parents[i]); bond=x-p[parent]
            length=np.linalg.norm(bond,axis=1)
            kb=float(m.bond_force_constants[i])
            out+=0.5*kb*(length-m.bond_lengths[i])**2
            ref=int(m.angle_refs[i])
            if ref>=0:
                axis=p[ref]-p[parent]; norm=np.linalg.norm(axis)
                cosine=np.sum(bond*axis,axis=1)/np.maximum(length*norm,1e-300)
                angle=np.arccos(np.clip(cosine,-1,1))
                out+=0.5*float(m.angle_force_constants[i])*(angle-m.angles[i])**2
        for torsion in ([] if m.is_rigid else m.torsions):
            ids=torsion['indices']
            if max(ids)!=i: continue
            pts=[x if j==i else np.broadcast_to(p[j],x.shape) for j in ids]
            phi=self.torsion_angles(*pts)
            out+=float(torsion['k'])*(1+np.cos(int(torsion['periodicity'])*phi-float(torsion.get('phase',0.0))))
        return out

    def _host_energy(self,coordinates,sigma,epsilon,charge):
        return self.host_evaluator.energy(coordinates,sigma,epsilon,charge)

    def molecule_energies(self,coords,molecule,environment):
        if not isinstance(environment,PreparedEnvironment): environment=self.prepare_environment(environment)
        x=np.asarray(coords,float)
        if x.ndim!=3 or x.shape[1:]!=(molecule.n_atoms,3): raise ValueError('batch shape must be (B,n_atoms,3)')
        if molecule.is_rigid and self.ewald is None:
            m=molecule
            return self._host_energy(x,m.sigma,m.epsilon,m.charge)+environment.evaluator.energy(x,m.sigma,m.epsilon,m.charge)
        return np.array([self.molecule_energy(y,molecule,environment) for y in x])

    def molecule_energy(self,coords,molecule,environment):
        if not isinstance(environment,PreparedEnvironment): environment=self.prepare_environment(environment)
        x=np.asarray(coords,float)
        return float(sum(self.incremental_energy(x[:i],x[i:i+1],i,molecule,environment)[0] for i in range(len(x))))

    def total_energy(self,entries):
        """Guest contribution relative to the empty host, counted once."""
        total=0.0; previous=[]
        for s,x in entries:
            total+=self.molecule_energy(x,self.species[s],self.prepare_environment(previous))
            previous.append((s,x))
        return float(total)

class IsolatedMoleculePotential:
    """Gas-reference internal energy, without host or periodic-image terms.

    Uses the target's finite-cutoff LJ convention and explicit bonded terms.
    Charge interactions use scaled bare Coulomb, as for an isolated molecule.
    The geometric reference measure is supplied by Molecule/CBMC, not hidden
    in this energy function. Its Q integral must be calibrated at the target T.
    """
    is_isolated_internal = True
    def __init__(self,species,cutoff=12.0,shifted=True,scale14_lj=1.0,
                 scale14_coulomb=1.0,electrostatics='bare'):
        if electrostatics not in ('bare','none'): raise ValueError('isolated electrostatics must be bare or none')
        self.species=list(species); self.electrostatics=electrostatics
        max_span=max(sum((m.bond_ranges[i,1] if m.bond_ranges is not None else m.bond_lengths[i])
                         for i in range(m.n_atoms)) if not m.is_rigid else
                     (float(np.linalg.norm(m.rigid_positions,axis=1).max())*2 if m.rigid_positions is not None else 0)
                     for m in species)
        self.helper=ClassicalPotential(species,np.eye(3)*(2*(max_span+cutoff+1)),cutoff,
              backend='reference',coulomb='none',shifted=shifted,scale14_lj=scale14_lj,
              scale14_coulomb=scale14_coulomb)
        self.empty=self.helper.prepare_environment([])
    def prepare_environment(self,entries):
        if entries: raise ValueError('isolated internal reference cannot contain other molecules')
        return self.empty
    def incremental_energy(self,prefix,trials,atom_index,molecule,environment=None):
        m=molecule; i=atom_index; x=np.asarray(trials,float)
        out=self.helper.incremental_energy(prefix,x,i,m,self.empty)
        if self.electrostatics=='bare' and not m.is_rigid:
            graph=self.helper._distances[id(m)]
            for j in range(i):
                if self.helper._excluded(m,i,j): continue
                scale=self.helper.scale14_coulomb if graph[i,j]==3 else 1.0
                qq=m.charge[i]*m.charge[j]*scale
                if qq:
                    r=np.linalg.norm(x-prefix[j],axis=1)
                    out+=COULOMB_KA*qq/np.maximum(r,1e-12)
        return out
    def molecule_energy(self,coords,molecule,environment=None):
        x=np.asarray(coords,float)
        return float(sum(self.incremental_energy(x[:i],x[i:i+1],i,molecule)[0] for i in range(len(x))))
