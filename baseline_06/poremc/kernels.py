"""FP64 batched pair-energy evaluators in Kelvin and Angstrom.

`ReferenceEnvironment` is the independent NumPy implementation.
`NativeEnvironment` uses the explicitly built C++17 library.
`CudaEnvironment` uses an optional CuPy CUDA kernel (no silent CPU fallback).
The interaction Hamiltonian is identical across these backends: truncated LJ
with optional potential shifting, plus optionally damped shifted-force (DSF)
Coulomb or the bare real-space Ewald pair term (erfc(alpha*r)/r). DSF is a finite-range model, not an Ewald approximation with a specified
error tolerance. The host/environment is immutable after construction.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
import math
from pathlib import Path
import sys
import weakref
import numpy as np

COULOMB_K = 167100.94689828736  # e²/(4π eps0 kB), K Angstrom / e².


def _array(a, shape, name, nonnegative=False):
    value = np.array(a, dtype=np.float64, order='C', copy=True)
    if value.shape != shape:
        raise ValueError(f'{name}: expected shape {shape}, got {value.shape}')
    if not np.all(np.isfinite(value)):
        raise ValueError(f'{name}: all values must be finite')
    if nonnegative and np.any(value < 0):
        raise ValueError(f'{name}: values must be nonnegative')
    value.flags.writeable = False
    return value


class _Environment:
    def __init__(self, pos, sigma, epsilon, charge, cell, cutoff, coulomb='none',
                 alpha=0.2, shifted=True, algorithm='cell', threads=1):
        pos = np.asarray(pos, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError('environment positions must have shape [M,3]')
        m = len(pos)
        self.pos = _array(pos, (m,3), 'pos')
        self.sigma = _array(sigma, (m,), 'sigma', True)
        self.epsilon = _array(epsilon, (m,), 'epsilon', True)
        self.charge = _array(charge, (m,), 'charge')
        self.cell = _array(cell, (3,3), 'cell')
        if not np.linalg.det(self.cell) > 0:
            raise ValueError('cell must have a positive determinant')
        self.inv = np.linalg.inv(self.cell)
        self.heights = 1.0 / np.linalg.norm(self.inv, axis=0)
        self.cutoff = float(cutoff)
        if not np.isfinite(self.cutoff) or self.cutoff <= 0 or np.any(2*self.cutoff >= self.heights):
            raise ValueError('cutoff must be positive and strictly less than half every face height')
        if coulomb not in ('none','dsf','ewald-real'):
            raise ValueError("coulomb must be 'none', 'dsf' or 'ewald-real'; complete Ewald also requires reciprocal/self/background terms")
        if algorithm not in ('direct','cell','cached'):
            raise ValueError("algorithm must be 'direct', 'cell' or 'cached'")
        if not isinstance(threads,(int,np.integer)) or threads < 1:
            raise ValueError('threads must be a positive integer')
        self.alpha = float(alpha)
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError('alpha must be finite and nonnegative')
        self.coulomb, self.shifted, self.algorithm, self.threads = coulomb, bool(shifted), algorithm, int(threads)
        self.frac = self.pos @ self.inv
        self.frac -= np.floor(self.frac)
        self.frac.flags.writeable = False
        self._closed = False

    def _trials(self, trials, sigma, epsilon, charge):
        if self._closed:
            raise RuntimeError('environment is closed')
        trials = np.asarray(trials, dtype=np.float64, order='C')
        if trials.ndim == 2:
            trials = trials[None,:,:]
        if trials.ndim != 3 or trials.shape[-1] != 3:
            raise ValueError('trials must have shape [B,A,3] (or one [A,3] trial)')
        if not np.all(np.isfinite(trials)):
            raise ValueError('trial coordinates must be finite')
        atoms = trials.shape[1]
        params = [_array(v,(atoms,),n,n != 'charge') for v,n in zip((sigma,epsilon,charge),('sigma','epsilon','charge'))]
        return np.ascontiguousarray(trials), *params

    def close(self):
        self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError('environment is closed')
        return self

    def __exit__(self, *args):
        self.close()


class ReferenceEnvironment(_Environment):
    """NumPy direct-pair reference, independent of native data structures."""
    def energy(self, trials, sigma, epsilon, charge):
        xyz, sigma, epsilon, charge = self._trials(trials,sigma,epsilon,charge)
        out = np.zeros(len(xyz))
        rc, alpha = self.cutoff, self.alpha
        vcut = math.erfc(alpha*rc)/rc
        slope = math.erfc(alpha*rc)/(rc*rc)+2*alpha/math.sqrt(math.pi)*math.exp(-(alpha*rc)**2)/rc
        for b in range(len(xyz)):
            for a in range(xyz.shape[1]):
                delta = xyz[b,a]@self.inv-self.frac
                delta -= np.rint(delta)
                cart = delta@self.cell
                r2 = np.einsum('ij,ij->i',cart,cart)
                mask = r2 < rc*rc
                if not np.any(mask):
                    continue
                dist2 = r2[mask]
                sij = 0.5*(sigma[a]+self.sigma[mask])
                eij = np.sqrt(epsilon[a]*self.epsilon[mask])
                qij = charge[a]*self.charge[mask]
                pair = np.zeros_like(dist2)
                active = (eij>0)&(sij>0)
                ordinary = active&(dist2>0)
                with np.errstate(divide='ignore',over='ignore',invalid='ignore'):
                    t3 = (sij[ordinary]**2/dist2[ordinary])**3
                    pair[ordinary] = 4*eij[ordinary]*t3*(t3-1)
                    if self.shifted:
                        tc3 = (sij[ordinary]/rc)**6
                        pair[ordinary] -= 4*eij[ordinary]*tc3*(tc3-1)
                pair[active&(dist2==0)] = np.inf
                if self.coulomb != 'none':
                    use = (qij!=0)&~np.isinf(pair)&(dist2>0)
                    r = np.sqrt(dist2[use])
                    erfc = np.array([math.erfc(alpha*float(v)) for v in r])
                    radial=erfc/r
                    if self.coulomb == 'dsf':
                        radial=radial-vcut+slope*(r-rc)
                    pair[use] += COULOMB_K*qij[use]*radial
                    singular = (qij!=0)&~np.isinf(pair)&(dist2==0)
                    pair[singular] = np.copysign(np.inf,qij[singular])
                out[b] += np.sum(pair)
        return out


_DOUBLE = ctypes.POINTER(ctypes.c_double)


def _ptr(a):
    return a.ctypes.data_as(_DOUBLE)


@lru_cache(maxsize=1)
def _load_native():
    extension = '.dylib' if sys.platform == 'darwin' else '.so'
    library = Path(__file__).resolve().parent / ('libporemc'+extension)
    if not library.exists():
        raise RuntimeError('PoreMC native library is not built. Run python native/build.py from the project root; use ReferenceEnvironment explicitly for the NumPy backend.')
    lib = ctypes.CDLL(str(library))
    lib.poremc_last_error.restype = ctypes.c_char_p
    lib.poremc_kernel_version.restype = ctypes.c_char_p
    lib.poremc_has_openmp.restype = ctypes.c_int
    lib.poremc_environment_create.argtypes = [ctypes.c_size_t,_DOUBLE,_DOUBLE,_DOUBLE,_DOUBLE,_DOUBLE,ctypes.c_double,ctypes.c_int,ctypes.c_double,ctypes.c_int,ctypes.c_int,ctypes.c_int]
    lib.poremc_environment_create.restype = ctypes.c_void_p
    lib.poremc_environment_destroy.argtypes = [ctypes.c_void_p]
    lib.poremc_environment_destroy.restype = None
    lib.poremc_environment_energy.argtypes = [ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t,_DOUBLE,_DOUBLE,_DOUBLE,_DOUBLE,_DOUBLE]
    lib.poremc_environment_energy.restype = ctypes.c_int
    lib.poremc_environment_energy_excluding.argtypes = [ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t,_DOUBLE,_DOUBLE,_DOUBLE,_DOUBLE,ctypes.c_size_t,ctypes.c_size_t,_DOUBLE]
    lib.poremc_environment_energy_excluding.restype = ctypes.c_int
    lib.poremc_environment_profile.argtypes = [ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint64)]
    lib.poremc_environment_profile.restype = ctypes.c_int
    lib.poremc_environment_export_plan.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_uint32),ctypes.POINTER(ctypes.c_uint64),ctypes.POINTER(ctypes.c_uint32)]
    lib.poremc_environment_export_plan.restype=ctypes.c_int
    return lib


class NativeEnvironment(_Environment):
    """C++ FP64 direct or fractional-cell-list pair energies; explicit CPU path."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lib = _load_native()
        self._handle = self._lib.poremc_environment_create(len(self.pos),_ptr(self.pos),_ptr(self.sigma),_ptr(self.epsilon),_ptr(self.charge),_ptr(self.cell),self.cutoff,{'none':0,'dsf':1,'ewald-real':2}[self.coulomb],self.alpha,int(self.shifted),{'direct':0,'cell':1,'cached':2}[self.algorithm],self.threads)
        if not self._handle:
            raise RuntimeError(self._lib.poremc_last_error().decode())
        self._finalizer = weakref.finalize(self,self._lib.poremc_environment_destroy,self._handle)

    @property
    def build_info(self):
        return {'version': self._lib.poremc_kernel_version().decode(), 'openmp': bool(self._lib.poremc_has_openmp()), 'algorithm': self.algorithm, 'threads': self.threads}

    def energy(self,trials,sigma,epsilon,charge):
        xyz,sigma,epsilon,charge = self._trials(trials,sigma,epsilon,charge)
        out = np.empty(len(xyz),dtype=np.float64)
        rc = self._lib.poremc_environment_energy(self._handle,len(xyz),xyz.shape[1],_ptr(xyz),_ptr(sigma),_ptr(epsilon),_ptr(charge),_ptr(out))
        if rc:
            raise RuntimeError(self._lib.poremc_last_error().decode())
        return out

    def export_spatial_plan(self):
        """Copy immutable CSR search plan for a GPU upload or inspection."""
        info=self.profile
        bins=np.empty(3,dtype=np.int32);order=np.empty(len(self.pos),dtype=np.uint32)
        offsets=np.zeros(max(1,info['query_cells']+1),dtype=np.uint64)
        indices=np.empty(info['neighbour_entries'],dtype=np.uint32)
        rc=self._lib.poremc_environment_export_plan(self._handle,
            bins.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),order.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)))
        if rc:raise RuntimeError('cannot export spatial plan')
        return {'bins':bins,'order':order,'offsets':offsets,'indices':indices,'indexed':bool(info['expanded_plan'])}

    @property
    def profile(self):
        if self._closed: raise RuntimeError('environment is closed')
        out=np.empty(8,dtype=np.uint64)
        self._lib.poremc_environment_profile(self._handle,out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)))
        return dict(zip(('energy_calls','pair_cache_hits','pair_cache_misses','pair_cache_bytes',
                         'spatial_plan_bytes','neighbour_entries','query_cells','expanded_plan'),map(int,out)))

    def energy_excluding(self,trials,sigma,epsilon,charge,exclude_start,exclude_end):
        """Exclude original source indices [start,end), independently of sorting."""
        if not 0 <= exclude_start <= exclude_end <= len(self.pos): raise ValueError('invalid excluded atom interval')
        xyz,sigma,epsilon,charge=self._trials(trials,sigma,epsilon,charge)
        out=np.empty(len(xyz),dtype=np.float64)
        rc=self._lib.poremc_environment_energy_excluding(self._handle,len(xyz),xyz.shape[1],
            _ptr(xyz),_ptr(sigma),_ptr(epsilon),_ptr(charge),exclude_start,exclude_end,_ptr(out))
        if rc: raise RuntimeError(self._lib.poremc_last_error().decode())
        return out

    def close(self):
        if not self._closed:
            self._finalizer()
        super().close()


class LegacyCudaEnvironment(_Environment):
    """Optional CuPy CUDA FP64 pair reduction, one block per trial.

    This backend requires a working CUDA device and CuPy. It has no implicit
    fallback. Its kernel is delivered as source; this release's CPU-only build
    environment cannot validate CUDA execution or benchmark GPU speed.
    GPU algorithms use direct pair evaluation, not the CPU cell-list index.
    """
    def __init__(self,*args,**kwargs):
        if kwargs.get('algorithm','direct') != 'direct':
            raise ValueError("CudaEnvironment currently implements algorithm='direct' only")
        kwargs['algorithm'] = 'direct'
        super().__init__(*args,**kwargs)
        try:
            import cupy as cp
        except ImportError as e:
            raise RuntimeError('CUDA backend requires CuPy and a working CUDA device; no CPU fallback was selected.') from e
        self._cp = cp
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError('CUDA backend requested but no CUDA device is available')
        source = Path(__file__).resolve().parent/'energies_cuda.cu'
        if not source.exists():
            source = Path(__file__).resolve().parents[1]/'native'/'energies_cuda.cu'
        self._kernel = cp.RawKernel(source.read_text(),'poremc_trials',options=('--std=c++11',))
        self._gpu = tuple(cp.asarray(v) for v in (self.frac,self.sigma,self.epsilon,self.charge,self.cell,self.inv))

    def energy(self,trials,sigma,epsilon,charge):
        xyz,sigma,epsilon,charge = self._trials(trials,sigma,epsilon,charge)
        if not len(xyz):
            return np.zeros(0)
        cp=self._cp
        gx,gs,ge,gq=(cp.asarray(v) for v in (xyz,sigma,epsilon,charge))
        result=cp.empty(len(xyz),dtype=cp.float64)
        self._kernel((len(xyz),),(128,),(np.int64(len(xyz)),np.int64(xyz.shape[1]),np.int64(len(self.pos)),gx,gs,ge,gq,*self._gpu,np.float64(self.cutoff),np.float64(self.alpha),np.int32(self.shifted),np.int32({'none':0,'dsf':1,'ewald-real':2}[self.coulomb]),result),shared_mem=128*8)
        return cp.asnumpy(result)


# Import after NativeEnvironment is defined; the new CUDA wrapper reuses its immutable plan.
def __getattr__(name):
    if name=='CudaEnvironment':
        from .cuda import CudaEnvironment
        return CudaEnvironment
    raise AttributeError(name)
