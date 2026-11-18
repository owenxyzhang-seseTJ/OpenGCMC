"""Frozen full-support spatial proposals with explicit density corrections."""
from __future__ import annotations
import math
import time
import numpy as np
from .cache import ArrayCache, host_key

class VoxelProposal:
    """Vose alias sampling over periodic voxels, uniform inside each selected voxel.

    relative_density(x) is V*q(x), relative to uniform Cartesian positions.
    CBMC divides trial Boltzmann weights by this quantity, including retracing.
    """
    def __init__(self,cell,probabilities,normalize=True):
        self.cell=np.array(cell,dtype=float,copy=True)
        if self.cell.shape!=(3,3) or not np.isfinite(self.cell).all() or np.linalg.det(self.cell)<=0:
            raise ValueError('finite positive-volume 3x3 cell required')
        self.inverse=np.linalg.inv(self.cell)
        p=np.array(probabilities,dtype=float,copy=True)
        if p.ndim!=3 or min(p.shape)<1 or not np.isfinite(p).all() or np.any(p<=0):
            raise ValueError('three-dimensional strictly positive voxel probabilities required')
        total=p.sum()
        if not np.isfinite(total) or total<=0:raise ValueError('invalid probability sum')
        if normalize:p/=total
        elif not np.isclose(total,1,rtol=0,atol=1e-12):raise ValueError('checkpoint probabilities are not normalized')
        self.probabilities=p;self.shape=np.array(p.shape,dtype=int);self.n=p.size
        scaled=p.ravel()*self.n;self.threshold=np.ones(self.n);self.alias=np.arange(self.n,dtype=np.int64)
        small=list(np.flatnonzero(scaled<1));large=list(np.flatnonzero(scaled>=1))
        while small and large:
            lo=small.pop();hi=large.pop();self.threshold[lo]=scaled[lo];self.alias[lo]=hi
            scaled[hi]-=1-scaled[lo]
            (small if scaled[hi]<1 else large).append(hi)
        for a in (self.cell,self.inverse,self.probabilities,self.shape,self.threshold,self.alias):a.flags.writeable=False
    def sample(self,rng,count):
        count=int(count)
        if count<0:raise ValueError('count must be nonnegative')
        r=rng.random(count)*self.n;i=np.floor(r).astype(np.int64)
        j=np.where(r-i<self.threshold[i],i,self.alias[i])
        grid=np.column_stack(np.unravel_index(j,tuple(self.shape)))
        return ((grid+rng.random((count,3)))/self.shape)@self.cell
    def log_relative_density(self,positions):
        x=np.asarray(positions,dtype=float)
        f=np.mod(x@self.inverse,1.0);idx=np.floor(f*self.shape).astype(int)
        idx=np.minimum(idx,self.shape-1)
        return np.log(self.n*self.probabilities[idx[...,0],idx[...,1],idx[...,2]])
    def to_dict(self):return {'kind':'voxel_alias','cell':self.cell.tolist(),'probabilities':self.probabilities.tolist()}
    @classmethod
    def from_dict(cls,d):
        if d is None:return None
        if d.get('kind')!='voxel_alias':raise ValueError('unknown root proposal')
        return cls(d['cell'],d['probabilities'],normalize=False)
    @classmethod
    def from_energies(cls,cell,energies,temperature,uniform_fraction=.05):
        e=np.asarray(energies,float)
        if e.ndim!=3 or np.isnan(e).any() or np.isneginf(e).any():raise ValueError('invalid grid energies')
        if not np.isfinite(temperature) or temperature<=0 or not 0<uniform_fraction<=1:raise ValueError('invalid proposal controls')
        lw=-e/temperature
        if np.isfinite(lw).any():
            w=np.exp(lw-np.max(lw));w/=w.sum()
        else:w=np.full(e.shape,1/e.size)
        return cls(cell,uniform_fraction/e.size+(1-uniform_fraction)*w)

def build_host_proposal(exact,molecule,temperature,shape=(16,16,16),uniform_fraction=.05,cache_dir=None,batch_size=1024):
    """Cache root-site host energies; temperature only reweights the same grid.

    For a multi-site guest this is a proposal, not a molecular adsorption model.
    """
    start=time.perf_counter();shape=np.broadcast_to(np.asarray(shape,int),(3,)).copy()
    if (shape<1).any() or np.prod(shape)>2_000_000:raise ValueError('invalid proposal grid')
    if int(batch_size)<1:raise ValueError('batch_size must be positive')
    key=host_key(exact,'root-center-energy',{'shape':shape.tolist(),
          'site':[float(molecule.sigma[0]),float(molecule.epsilon[0]),float(molecule.charge[0])]})
    cache=ArrayCache(cache_dir) if cache_dir is not None else None
    saved=cache.get(key) if cache else None;hit=saved is not None
    if saved is None:
        frac=np.stack(np.meshgrid(*[(np.arange(n)+.5)/n for n in shape],indexing='ij'),axis=-1).reshape(-1,3)
        x=frac@exact.cell;values=[]
        for i in range(0,len(x),batch_size):
            values.append(exact.host_evaluator.energy(x[i:i+batch_size,None,:],molecule.sigma[:1],molecule.epsilon[:1],molecule.charge[:1]))
        energy=np.concatenate(values).reshape(tuple(shape))
        if cache:cache.put(key,{'energy':energy})
    else:energy=saved['energy']
    result=VoxelProposal.from_energies(exact.cell,energy,temperature,uniform_fraction)
    result.metadata={'kind':'full_support_alias','cache_hit':hit,'cache_key':key,'shape':shape.tolist(),
                     'uniform_fraction':uniform_fraction,'setup_seconds':time.perf_counter()-start}
    return result
