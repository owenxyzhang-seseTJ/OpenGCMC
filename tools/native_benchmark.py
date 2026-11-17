"""Minimal single-site benchmark access to the production native C ABI.

The same native sampling loop is used by FastGCMC. This harness records arrays
directly and keeps validation outside the timed production loop.
"""
import ctypes as C
import numpy as np
from poremc.fast import _Config, _bind, _INT, _U64
from poremc.kernels import _load_native, _ptr


class NativePointSampler:
    def __init__(self,host,temperature,activity,seed,sigma=3.,epsilon=5.):
        self.host=host;self.lib=_bind(_load_native());self.arrays=[]
        def dp(a):
            x=np.ascontiguousarray(a,dtype=np.float64);self.arrays.append(x);return _ptr(x)
        def ip(a):
            x=np.ascontiguousarray(a,dtype=np.int32);self.arrays.append(x);return x.ctypes.data_as(_INT)
        config=_Config(atoms=1,torsions=0,max_particles=10000,seed=seed,
            temperature=temperature,activity=activity,cutoff=host.cutoff,alpha=host.alpha,
            max_translation=1.,max_rotation=.5,scale14_lj=1.,scale14_coulomb=1.,
            shifted=int(host.shifted),coulomb=0,algorithm={'direct':0,'cell':1,'cached':2}[host.algorithm],threads=1,rigid=1,
            cell=(C.c_double*9)(*host.cell.ravel()),move_weights=(C.c_double*5)(.5,.5,0.,0.,0.),
            sigma=dp([sigma]),epsilon=dp([epsilon]),charge=dp([0.]),bond_lengths=dp([0.]),angles=dp([0.]),
            rigid_positions=dp([0.,0.,0.]),bond_ranges=dp([0.,0.]),angle_ranges=dp([0.,0.]),
            bond_constants=dp([0.]),angle_constants=dp([0.]),parents=ip([-1]),angle_refs=ip([-1]),
            dihedral_refs=ip([-1]),trials=ip([1]),exclusions=ip([0]),
            torsion_indices=ip([]),torsion_periodicity=ip([]),torsion_k=dp([]),torsion_phase=dp([]),host=host._handle)
        self.handle=self.lib.poremc_fast_create(C.byref(config))
        if not self.handle:raise RuntimeError(self.lib.poremc_fast_last_error().decode())

    def check(self,status):
        if status:raise RuntimeError(self.lib.poremc_fast_last_error().decode())

    def add(self,positions):
        for point in np.asarray(positions).reshape(-1,3):self.check(self.lib.poremc_fast_add(self.handle,_ptr(np.ascontiguousarray(point))))

    def run(self,steps,burnin=0,sample_every=20):
        counts=np.empty(steps//sample_every,dtype=np.uint64)
        energy=np.empty(0);timing=np.empty(2)
        self.check(self.lib.poremc_fast_run(self.handle,burnin,steps,sample_every,counts.ctypes.data_as(_U64),_ptr(energy),0,_ptr(timing)))
        return counts,timing

    def positions(self):
        count=int(self.lib.poremc_fast_count(self.handle));x=np.empty((count,3))
        self.check(self.lib.poremc_fast_positions(self.handle,_ptr(x)));return x

    def energy(self):
        e=np.empty(1);self.check(self.lib.poremc_fast_total_energy(self.handle,_ptr(e)));return float(e[0])

    def counters(self):
        x=np.empty((5,2),dtype=np.uint64)
        self.check(self.lib.poremc_fast_counters(self.handle,x.ctypes.data_as(_U64)))
        return x

    def close(self):
        if self.handle:self.lib.poremc_fast_destroy(self.handle);self.handle=None

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
