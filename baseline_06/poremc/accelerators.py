"""Periodic grid and fitted Fourier surrogates for EXACT delayed acceptance.

These are proposal energies. Pass target_potential=exact to GCMC. Surrogate
clipping, interpolation, or regression then cannot change the invariant target
measure; they can change mixing and stage-two acceptance. No model is fitted
or updated during production.
"""
import itertools
import time
import numpy as np
from .potentials import ClassicalPotential

class PeriodicGrid:
    def __init__(self,values,cell):
        self.values=np.asarray(values,float)
        if self.values.ndim!=3 or min(self.values.shape)<2 or not np.isfinite(self.values).all():
            raise ValueError('grid must be finite with at least two points per dimension')
        self.cell=np.asarray(cell,float); self.inverse=np.linalg.inv(self.cell)
        self.shape=np.array(self.values.shape)
    def __call__(self,positions):
        p=np.asarray(positions,float); scaled=np.mod(p@self.inverse,1.0)*self.shape
        lo=np.floor(scaled).astype(int); t=scaled-lo
        out=np.zeros(p.shape[:-1])
        for dx,dy,dz in itertools.product((0,1),repeat=3):
            corner=np.mod(lo+[dx,dy,dz],self.shape)
            weight=(t[...,0] if dx else 1-t[...,0])*(t[...,1] if dy else 1-t[...,1])*(t[...,2] if dz else 1-t[...,2])
            out+=weight*self.values[corner[...,0],corner[...,1],corner[...,2]]
        return out

class _HostSurrogate(ClassicalPotential):
    def _copy(self,exact):
        if not isinstance(exact,ClassicalPotential): raise TypeError('a ClassicalPotential target is required')
        self.__dict__.update(exact.__dict__)
        self.exact=exact
        self.site_parameters=sorted({tuple(map(float,v)) for m in self.species for v in zip(m.sigma,m.epsilon,m.charge)})
        self.fields={}; self.frozen=True
    def _host_energy(self,coordinates,sigma,epsilon,charge):
        x=np.asarray(coordinates,float); out=np.zeros(len(x))
        for i,key in enumerate(zip(sigma,epsilon,charge)):
            out+=self.fields[tuple(map(float,key))](x[:,i,:])
        return out

class GridHostSurrogate(_HostSurrogate):
    def __init__(self,exact,shape=(24,24,24),clip_kelvin=(-50000,50000),batch_size=512,cache_dir=None):
        start=time.perf_counter(); self._copy(exact)
        shape=np.broadcast_to(np.asarray(shape,int),(3,))
        if (shape<2).any() or int(np.prod(shape))>20_000_000: raise ValueError('invalid or excessively large grid')
        if clip_kelvin[0]>=clip_kelvin[1] or not np.isfinite(clip_kelvin).all(): raise ValueError('invalid finite clipping interval')
        if batch_size<1:raise ValueError('batch_size must be positive')
        from .cache import ArrayCache,host_key
        cache=ArrayCache(cache_dir) if cache_dir is not None else None
        key=host_key(exact,'periodic-trilinear-grid',{'shape':shape.tolist(),'clip':list(clip_kelvin),'types':self.site_parameters})
        saved=cache.get(key) if cache else None
        if saved is not None:
            for i,site in enumerate(self.site_parameters):self.fields[site]=PeriodicGrid(saved[f'field_{i}'],self.cell)
            self.metadata={'kind':'periodic_trilinear_host_grid','shape':shape.tolist(),'clip_kelvin':list(clip_kelvin),
              'site_types':len(self.site_parameters),'setup_seconds':time.perf_counter()-start,'frozen':True,
              'requires_target_correction':True,'cache_hit':True,'cache_key':key}
            return
        frac=np.stack(np.meshgrid(*(np.arange(n)/n for n in shape),indexing='ij'),axis=-1).reshape(-1,3)
        p=frac@self.cell
        for key in self.site_parameters:
            sigma,eps,q=key; values=[]
            for j in range(0,len(p),batch_size):
                e=self.host_evaluator.energy(p[j:j+batch_size,None,:],[sigma],[eps],[q])
                values.extend(np.clip(e,*clip_kelvin))
            self.fields[key]=PeriodicGrid(np.array(values).reshape(tuple(shape)),self.cell)
        cache_key=host_key(exact,'periodic-trilinear-grid',{'shape':shape.tolist(),'clip':list(clip_kelvin),'types':self.site_parameters})
        if cache:cache.put(cache_key,{f'field_{i}':self.fields[site].values for i,site in enumerate(self.site_parameters)})
        self.metadata={'kind':'periodic_trilinear_host_grid','shape':shape.tolist(),'clip_kelvin':list(clip_kelvin),
                       'site_types':len(self.site_parameters),'setup_seconds':time.perf_counter()-start,
                       'frozen':True,'requires_target_correction':True,'cache_hit':False,'cache_key':cache_key}

class FourierField:
    def __init__(self,cell,modes,coefficients):
        self.inverse=np.linalg.inv(cell); self.modes=np.asarray(modes,float); self.coefficients=np.asarray(coefficients,float)
    def design(self,p):
        phase=2*np.pi*(np.mod(np.asarray(p)@self.inverse,1.0)@self.modes.T)
        return np.concatenate([np.ones((*phase.shape[:-1],1)),np.cos(phase),np.sin(phase)],axis=-1)
    def __call__(self,p): return self.design(p)@self.coefficients

class LearnedHostSurrogate(_HostSurrogate):
    """Executable small ML model: ridge-fitted periodic Fourier energy model.

    This is a tested host-energy SURROGATE, not an ab-initio universal ML force
    field. It needs no network or pretrained weights. Validation RMSE measures
    clipped training-target error and is not a bound on GCMC thermodynamics.
    """
    def __init__(self,exact,n_train=1024,n_modes=40,max_frequency=4,ridge=1e-3,
                 seed=2026,clip_kelvin=(-5000,20000)):
        start=time.perf_counter(); self._copy(exact)
        if n_train<8 or n_modes<1 or max_frequency<1 or ridge<=0: raise ValueError('invalid ML hyperparameters')
        rng=np.random.default_rng(seed); choices=[]
        for n in itertools.product(range(-max_frequency,max_frequency+1),repeat=3):
            if n==(0,0,0): continue
            first=next(v for v in n if v)
            if first>0: choices.append(n)
        choices.sort(key=lambda n:sum(v*v for v in n)); modes=np.array(choices[:n_modes])
        positions=rng.random((n_train+256,3))@self.cell
        empty=FourierField(self.cell,modes,np.zeros(1+2*len(modes)))
        design=empty.design(positions); a=design[:n_train]; reg=np.eye(a.shape[1])*ridge; reg[0,0]=0
        inverse=np.linalg.solve(a.T@a+reg,a.T)
        errors={}
        for key in self.site_parameters:
            sigma,eps,q=key
            y=np.clip(self.host_evaluator.energy(positions[:,None,:],[sigma],[eps],[q]),*clip_kelvin)
            coef=inverse@y[:n_train]; field=FourierField(self.cell,modes,coef)
            self.fields[key]=field
            errors[str(key)]=float(np.sqrt(np.mean((design[n_train:]@coef-y[n_train:])**2)))
        self.metadata={'kind':'ridge_periodic_fourier_host_energy','n_train':n_train,'n_modes':len(modes),
                       'validation_clipped_rmse_K':errors,'clip_kelvin':list(clip_kelvin),
                       'setup_seconds':time.perf_counter()-start,'frozen':True,'requires_target_correction':True}
