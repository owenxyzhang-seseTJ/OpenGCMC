"""Device-resident neighbour plans, parameter tables and reusable CUDA workspaces.

`energy` is the validated NumPy boundary. `energy_device` avoids H2D/D2H and
requires finite, contiguous FP64 CuPy coordinates on this environment's device.
Returned device views are workspace-owned until the next call. One instance
belongs to one stream / one caller; use separate instances for concurrent calls.
"""
from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
import math
import numpy as np
from .kernels import _Environment,NativeEnvironment,COULOMB_K

class CudaEnvironment(_Environment):
    def __init__(self,*args,**kwargs):
        kwargs.setdefault('algorithm','cached');super().__init__(*args,**kwargs)
        try:import cupy as cp
        except ImportError as e:raise RuntimeError('CUDA requires CuPy and a working CUDA device') from e
        if cp.cuda.runtime.getDeviceCount()<1:raise RuntimeError('no CUDA device available')
        self.cp=cp;self.device=int(cp.cuda.runtime.getDevice());self.stream=cp.cuda.get_current_stream()
        self._owner=NativeEnvironment(self.pos,self.sigma,self.epsilon,self.charge,self.cell,self.cutoff,
              coulomb=self.coulomb,alpha=self.alpha,shifted=self.shifted,algorithm='direct' if self.algorithm=='direct' else 'cached')
        plan=self._owner.export_spatial_plan();self.plan=plan;order=plan['order']
        self._sorted=(self.sigma[order],self.epsilon[order],self.charge[order])
        self._resident=tuple(cp.asarray(v) for v in (np.ascontiguousarray(self.frac[order].T),self.cell,self.inv,
                plan['bins'],plan['offsets'],plan['indices']))
        source=(Path(__file__).with_name('cached_cuda.cu')).read_text()
        module=cp.RawModule(code=source,options=('--std=c++14','--fmad=false'))
        self._atoms=module.get_function('poremc_cached_atoms');self._sum=module.get_function('poremc_sum_atoms');self._select=module.get_function('poremc_select')
        self._module=module;self._bindings=OrderedDict();self._binding_bytes=0;self._capacity=0
        self._selection=cp.empty(3,dtype=cp.float64);self._empty=cp.empty(1,dtype=cp.float64)
        self._stats={'calls':0,'binding_hits':0,'binding_builds':0,'workspace_reallocations':0,'host_bytes_uploaded':0,'result_bytes_downloaded':0}
    def _check_context(self):
        if self._closed:raise RuntimeError('environment is closed')
        if self.cp.cuda.runtime.getDevice()!=self.device or self.cp.cuda.get_current_stream().ptr!=self.stream.ptr:
            raise RuntimeError('use the construction device and stream, or create a separate environment')
    def _bind(self,sigma,epsilon,charge):
        parameters=np.ascontiguousarray(np.column_stack([sigma,epsilon,charge]),dtype=np.float64)
        if not np.isfinite(parameters).all() or (parameters[:,:2]<0).any():raise ValueError('invalid query parameters')
        key=parameters.tobytes()
        if key in self._bindings:
            self._bindings.move_to_end(key);self._stats['binding_hits']+=1;return self._bindings[key]
        types,ids=np.unique(parameters,axis=0,return_inverse=True);hs,he,hq=self._sorted;m=len(hs)
        table=np.empty((len(types),4,m),dtype=np.float64)
        for i,(s,e,q) in enumerate(types):
            s2=(.5*(s+hs))**2;e4=4*np.sqrt(e)*np.sqrt(he);c3=(s2/(self.cutoff*self.cutoff))**3
            table[i]=np.array([s2,e4,e4*c3*(c3-1) if self.shifted else np.zeros(m),COULOMB_K*q*hq])
        binding=(self.cp.asarray(table),self.cp.asarray(ids.astype(np.int32)))
        size=sum(a.nbytes for a in binding)
        while self._bindings and self._binding_bytes+size>64*1024**2:
            _,old=self._bindings.popitem(last=False);self._binding_bytes-=sum(a.nbytes for a in old)
        self._bindings[key]=binding;self._binding_bytes+=size;self._stats['binding_builds']+=1
        return binding
    def _reserve(self,q):
        if q<=self._capacity:return
        self._capacity=max(q,2*self._capacity,128);cp=self.cp
        self._xyz=cp.empty((self._capacity,3),dtype=cp.float64)
        self._partial=cp.empty(self._capacity,dtype=cp.float64);self._out=cp.empty(self._capacity,dtype=cp.float64)
        self._stats['workspace_reallocations']+=1
    def energy_device(self,xyz,sigma,epsilon,charge):
        self._check_context();cp=self.cp
        if not isinstance(xyz,cp.ndarray) or xyz.dtype!=cp.float64 or xyz.ndim!=3 or xyz.shape[-1]!=3 or not xyz.flags.c_contiguous:
            raise ValueError('contiguous CuPy FP64 [B,A,3] input required')
        if xyz.device.id!=self.device:raise ValueError('coordinates reside on another device')
        b,a=xyz.shape[:2]
        if any(np.asarray(v).shape!=(a,) for v in (sigma,epsilon,charge)):raise ValueError('parameter count differs')
        if b==0:return cp.empty(0,dtype=cp.float64)
        if a==0:return cp.zeros(b,dtype=cp.float64)
        self._reserve(b*a);pair,ids=self._bind(sigma,epsilon,charge);self._stats['calls']+=1
        rc=self.cutoff;al=self.alpha;vcut=math.erfc(al*rc)/rc;slope=vcut/rc+2*al/math.sqrt(math.pi)*math.exp(-al*al*rc*rc)/rc
        diag=int(np.count_nonzero(self.cell-np.diag(np.diag(self.cell)))==0)
        self._atoms((b*a,),(128,),(np.int64(b*a),np.int32(a),np.int64(len(self.pos)),xyz,*self._resident,
                  pair,ids,np.int32(self.plan['indexed']),np.int32(diag),np.int32({'none':0,'dsf':1,'ewald-real':2}[self.coulomb]),
                  np.float64(rc*rc),np.float64(rc),np.float64(al),np.float64(vcut),np.float64(slope),self._partial))
        self._sum(((b+127)//128,),(128,),(np.int64(b),np.int32(a),self._partial,self._out))
        return self._out[:b]
    def energy(self,trials,sigma,epsilon,charge):
        xyz,sigma,epsilon,charge=self._trials(trials,sigma,epsilon,charge);self._check_context()
        b,a=xyz.shape[:2]
        if not b or not a:return np.zeros(b)
        self._reserve(b*a);device_xyz=self._xyz[:b*a].reshape(b,a,3);device_xyz.set(xyz,stream=self.stream)
        self._stats['host_bytes_uploaded']+=xyz.nbytes
        out=self.energy_device(device_xyz,sigma,epsilon,charge)
        self._stats['result_bytes_downloaded']+=8*b
        return self.cp.asnumpy(out,stream=self.stream)
    def select_device(self,energies,temperature,uniform,log_relative_density=None):
        """Return a device view [log(mean weight), selected_index, selected_energy]."""
        self._check_context();cp=self.cp
        if not np.isfinite(temperature) or temperature<=0 or not 0<=uniform<1:raise ValueError('invalid selection controls')
        if not isinstance(energies,cp.ndarray) or energies.dtype!=cp.float64 or energies.ndim!=1 or not energies.flags.c_contiguous or not len(energies):raise ValueError('nonempty contiguous device energies required')
        logq=self._empty if log_relative_density is None else log_relative_density
        if log_relative_density is not None and (not isinstance(logq,cp.ndarray) or logq.dtype!=cp.float64 or logq.shape!=energies.shape or not logq.flags.c_contiguous):raise ValueError('invalid proposal density vector')
        self._select((1,),(128,),(np.int64(len(energies)),energies,logq,np.int32(log_relative_density is not None),np.float64(temperature),np.float64(uniform),self._selection))
        return self._selection
    @property
    def profile(self):
        return {**self._stats,'parameter_cache_bytes':self._binding_bytes,'capacity_atoms':self._capacity,'spatial':self._owner.profile}
    def close(self):
        if not self._closed:
            self.stream.synchronize();self._owner.close();self._bindings.clear();self._resident=()
            for name in ('_xyz','_partial','_out','_selection','_empty'):
                if hasattr(self,name):setattr(self,name,None)
        super().close()
