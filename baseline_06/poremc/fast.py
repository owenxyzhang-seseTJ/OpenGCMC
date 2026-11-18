"""Full C++ GCMC loop for one classical rigid or tree-flexible component.

Supports LJ, DSF, native incremental Ewald or cubic particle-mesh Ewald,
rigid bodies, bounded or fixed bonds/angles, Fourier torsions and CBMC.
The complete move loop and reciprocal candidate batches remain in C++.
Native reference-CBMC and delayed acceptance reuse fixed-host grids.
Multi-component sampling, ML potentials and online adaptation use
:class:`poremc.engine.GCMC`.
The reference measures and activity units match the Python engine.
"""
from __future__ import annotations
import ctypes as C
import hashlib
import json
import math
from pathlib import Path
import weakref
import numpy as np
from .kernels import NativeEnvironment, _load_native, _ptr
from .topology import Molecule
from .potentials import ClassicalPotential
from .statistics import summarize
from .engine import Particle

_DOUBLE=C.POINTER(C.c_double)
_INT=C.POINTER(C.c_int32)
_U64=C.POINTER(C.c_uint64)

class _Config(C.Structure):
    _fields_=[(n,C.c_uint64) for n in ('atoms','torsions','max_particles','seed')]+[
        (n,C.c_double) for n in ('temperature','activity','cutoff','alpha','max_translation','max_rotation','scale14_lj','scale14_coulomb')]+[
        (n,C.c_int32) for n in ('shifted','coulomb','algorithm','threads','rigid')]+[
        ('cell',C.c_double*9),('move_weights',C.c_double*5)]+[
        (n,_DOUBLE) for n in ('sigma','epsilon','charge','bond_lengths','angles','rigid_positions','bond_ranges','angle_ranges','bond_constants','angle_constants')]+[
        (n,_INT) for n in ('parents','angle_refs','dihedral_refs','trials','exclusions','torsion_indices','torsion_periodicity')]+[
        ('torsion_k',_DOUBLE),('torsion_phase',_DOUBLE),('host',C.c_void_p),
        ('ewald_kmax',C.c_int32*3),('pme_mesh',C.c_int32*3),
        ('reciprocal_method',C.c_int32),('reciprocal_backend',C.c_int32),
        ('host_atoms',C.c_uint64),('host_positions',_DOUBLE),('host_charge',_DOUBLE),
        ('cuda_library',C.c_char_p),
        ('ewald_half_spectrum',C.c_int32),('sampling_mode',C.c_int32),('host_grid_mesh',C.c_int32*3),
        ('ewald_k_cutoff',C.c_double),('host_grid_cap',C.c_double),('grid_root_bias',C.c_double),
        ('host_grid_cache',C.c_char_p),('host_grid_key',C.c_char_p),('host_grid_steep',C.c_int32)]


def _bind(lib):
    if not hasattr(lib,'poremc_fast_create'):
        raise RuntimeError('Rebuild with python native/build.py to enable the full C++ FastGCMC loop.')
    lib.poremc_fast_config_size.restype=C.c_size_t
    if lib.poremc_fast_config_size()!=C.sizeof(_Config):
        raise RuntimeError('FastGCMC C ABI layout mismatch; rebuild the native library')
    lib.poremc_fast_last_error.restype=C.c_char_p
    lib.poremc_fast_create.argtypes=[C.POINTER(_Config)];lib.poremc_fast_create.restype=C.c_void_p
    lib.poremc_fast_destroy.argtypes=[C.c_void_p];lib.poremc_fast_destroy.restype=None
    lib.poremc_fast_add.argtypes=[C.c_void_p,_DOUBLE];lib.poremc_fast_add.restype=C.c_int
    lib.poremc_fast_add_raw.argtypes=[C.c_void_p,_DOUBLE];lib.poremc_fast_add_raw.restype=C.c_int
    lib.poremc_fast_count.argtypes=[C.c_void_p];lib.poremc_fast_count.restype=C.c_size_t
    lib.poremc_fast_step_count.argtypes=[C.c_void_p];lib.poremc_fast_step_count.restype=C.c_uint64
    lib.poremc_fast_positions.argtypes=[C.c_void_p,_DOUBLE];lib.poremc_fast_positions.restype=C.c_int
    lib.poremc_fast_counters.argtypes=[C.c_void_p,_U64];lib.poremc_fast_counters.restype=C.c_int
    lib.poremc_fast_run.argtypes=[C.c_void_p,C.c_uint64,C.c_uint64,C.c_uint64,_U64,_DOUBLE,C.c_int,_DOUBLE];lib.poremc_fast_run.restype=C.c_int
    lib.poremc_fast_total_energy.argtypes=[C.c_void_p,_DOUBLE];lib.poremc_fast_total_energy.restype=C.c_int
    lib.poremc_fast_rng.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t];lib.poremc_fast_rng.restype=C.c_size_t
    lib.poremc_fast_restore_bookkeeping.argtypes=[C.c_void_p,C.c_char_p,C.c_uint64,_U64];lib.poremc_fast_restore_bookkeeping.restype=C.c_int
    lib.poremc_fast_set_root.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int,_DOUBLE];lib.poremc_fast_set_root.restype=C.c_int
    lib.poremc_fast_profile.argtypes=[C.c_void_p,_U64];lib.poremc_fast_profile.restype=C.c_int
    for name in ('poremc_fast_longrange_state','poremc_fast_longrange_info','poremc_fast_acceleration_info'):
        function=getattr(lib,name)
        function.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t];function.restype=C.c_size_t
    lib.poremc_fast_restore_longrange.argtypes=[C.c_void_p,C.c_char_p]
    lib.poremc_fast_restore_longrange.restype=C.c_int
    lib.poremc_fast_site_energies.argtypes=[C.c_void_p,C.c_size_t,C.c_size_t,_DOUBLE,C.c_size_t,_DOUBLE,_DOUBLE]
    lib.poremc_fast_site_energies.restype=C.c_int
    lib.poremc_fast_molecule_energies.argtypes=[C.c_void_p,C.c_size_t,C.c_size_t,_DOUBLE,_DOUBLE]
    lib.poremc_fast_molecule_energies.restype=C.c_int
    lib.poremc_fast_surrogate_site_energies.argtypes=lib.poremc_fast_site_energies.argtypes
    lib.poremc_fast_surrogate_site_energies.restype=C.c_int
    lib.poremc_fast_surrogate_molecule_energies.argtypes=lib.poremc_fast_molecule_energies.argtypes
    lib.poremc_fast_surrogate_molecule_energies.restype=C.c_int
    for name in ('poremc_fast_acceleration_counters','poremc_fast_restore_acceleration_counters'):
        function=getattr(lib,name);function.argtypes=[C.c_void_p,_U64];function.restype=C.c_int
    return lib


class FastGCMC:
    """One-component classical GCMC with no Python code inside the move loop.

    Constructor and run signatures mirror the Python GCMC engine where
    supported. `regrow` means a full molecule regrowth, whereas the Python
    implementation can also choose a partial suffix. `seed` initializes a
    native mt19937_64 generator, so seeds do not promise cross-engine traces.
    """
    MOVE_NAMES=('insert','delete','translate','rotate','regrow')
    def __init__(self,species,potential,cell,temperature,activities,seed=12345,trials=8,
                 move_weights=None,max_translation=1.,max_rotation=.5,max_particles=100000,
                 target_potential=None,species_probabilities=None,root_proposal=None,
                 reciprocal_method='direct',pme_mesh=32,reciprocal_backend='cpu',cuda_library=None,
                 sampling_mode='exact',host_grid_mesh=32,host_grid_cap=1e5,host_grid_cache=None,
                 grid_root_bias=0.,ewald_half_spectrum=True,ewald_k_cutoff=0.,host_grid_steep='boltzmann'):
        self.species=[species] if isinstance(species,Molecule) else list(species)
        if len(self.species)!=1:
            raise ValueError('FastGCMC supports exactly one component; use GCMC for mixtures')
        if type(potential) is not ClassicalPotential or potential.coulomb not in ('none','dsf','ewald') or target_potential is not None:
            raise ValueError('FastGCMC requires ClassicalPotential with LJ, DSF or Ewald; use GCMC for external ML targets')
        if sampling_mode not in ('exact','delayed'):
            raise ValueError('sampling_mode must be exact or delayed')
        self.sampling_mode=sampling_mode
        self.ewald_half_spectrum=bool(ewald_half_spectrum)
        self.ewald_k_cutoff=float(ewald_k_cutoff)
        if not math.isfinite(self.ewald_k_cutoff) or self.ewald_k_cutoff<0:
            raise ValueError('ewald_k_cutoff must be finite and nonnegative (inverse angstrom)')
        if self.ewald_k_cutoff and (potential.coulomb!='ewald' or reciprocal_method!='direct'):
            raise ValueError('ewald_k_cutoff applies to direct Ewald only')
        gm=np.broadcast_to(np.asarray(host_grid_mesh),(3,))
        if not np.isfinite(gm).all() or np.any(gm!=np.floor(gm)) or np.any(gm<4):
            raise ValueError('host_grid_mesh must contain three integers of at least4')
        self.host_grid_mesh=tuple(map(int,gm))
        if sampling_mode=='delayed' and potential.coulomb=='ewald' and any(n&(n-1) for n in self.host_grid_mesh):
            raise ValueError('Ewald host grids use power-of-two dimensions')
        self.host_grid_cap=float(host_grid_cap)
        if not math.isfinite(self.host_grid_cap) or self.host_grid_cap<0:
            raise ValueError('host_grid_cap must be finite and nonnegative')
        if host_grid_steep not in ('linear','boltzmann'):
            raise ValueError('host_grid_steep must be linear or boltzmann')
        self.host_grid_steep=host_grid_steep
        self.grid_root_bias=float(grid_root_bias)
        if not 0<=self.grid_root_bias<1:
            raise ValueError('grid_root_bias must be in [0,1)')
        if self.grid_root_bias and (sampling_mode!='delayed' or root_proposal is not None):
            raise ValueError('grid_root_bias requires delayed mode without an explicit root_proposal')
        self.host_grid_cache=str(Path(host_grid_cache).expanduser().resolve()) if host_grid_cache is not None else None
        if sampling_mode=='delayed' and self.host_grid_cache:
            Path(self.host_grid_cache).parent.mkdir(parents=True,exist_ok=True)
        if reciprocal_method not in ('direct','pme'):
            raise ValueError('reciprocal_method must be direct or pme')
        if reciprocal_backend not in ('cpu','auto','cuda'):
            raise ValueError('reciprocal_backend must be cpu, auto or cuda')
        if sampling_mode=='delayed' and reciprocal_backend=='cuda':
            raise ValueError('CUDA candidate batches use sampling_mode=exact; delayed selected corrections run in native CPU')
        if potential.coulomb!='ewald' and (reciprocal_method!='direct' or reciprocal_backend!='cpu'):
            raise ValueError('reciprocal options require coulomb=ewald')
        self.reciprocal_method=reciprocal_method
        self.reciprocal_backend=reciprocal_backend
        mesh=np.broadcast_to(np.asarray(pme_mesh),(3,))
        if not np.isfinite(mesh).all() or np.any(mesh!=np.floor(mesh)):
            raise ValueError('pme_mesh must contain three integers')
        self.pme_mesh=tuple(map(int,mesh))
        if reciprocal_method=='pme' and any(n<8 or n&(n-1) for n in self.pme_mesh):
            raise ValueError('PME uses power-of-two mesh dimensions of at least 8')
        extension='.dylib' if __import__('sys').platform=='darwin' else '.so'
        default_cuda=Path(__file__).resolve().parent/('libporemc_ewald_cuda'+extension)
        self._cuda_library_explicit=cuda_library is not None
        self.cuda_library=str(Path(cuda_library).expanduser().resolve()) if cuda_library else str(default_cuda)
        if len(potential.species)!=1 or potential.species[0].to_dict()!=self.species[0].to_dict():
            raise ValueError('potential and FastGCMC molecule definitions differ')
        if species_probabilities is not None and (len(species_probabilities)!=1 or not np.isfinite(species_probabilities).all() or species_probabilities[0]<=0):
            raise ValueError('one positive component probability is required')
        self.species=[Molecule.from_dict(self.species[0].to_dict())]
        self.potential=potential;self.target_potential=None;self.cell=np.array(cell,float,copy=True)
        if self.cell.shape==(3,):self.cell=np.diag(self.cell)
        if self.cell.shape!=(3,3) or not np.array_equal(self.cell,potential.cell):
            raise ValueError('FastGCMC cell must equal the potential cell')
        self.volume=float(np.linalg.det(self.cell))
        self.temperature=float(temperature);self.activities=np.array(activities,float,copy=True).reshape(-1)
        if self.temperature<=0 or not math.isfinite(self.temperature) or self.activities.shape!=(1,) or not np.isfinite(self.activities).all() or self.activities[0]<0:
            raise ValueError('finite positive temperature and one nonnegative activity are required')
        self.max_translation=float(max_translation);self.max_rotation=float(max_rotation)
        if not math.isfinite(self.max_translation) or self.max_translation<=0 or not 0<self.max_rotation<=math.pi:
            raise ValueError('invalid translation/rotation step')
        self.max_particles=int(max_particles)
        if self.max_particles<1:raise ValueError('max_particles must be positive')
        m=self.species[0];n=m.n_atoms
        self.trials=np.full(n,trials,dtype=np.int32) if np.ndim(trials)==0 else np.array(trials,dtype=np.int32,copy=True)
        if self.trials.shape!=(n,) or np.any(self.trials<1):raise ValueError('positive trial count per atom is required')
        weights={'insert':.25,'delete':.25,'translate':.2,'rotate':.1,'regrow':.2} if move_weights is None else move_weights
        if not set(weights).issubset(self.MOVE_NAMES):raise ValueError('unknown move name')
        self.move_weights={name:float(weights.get(name,0)) for name in self.MOVE_NAMES}
        p=np.array(list(self.move_weights.values()))
        if not np.isfinite(p).all() or (p<0).any() or p.sum()<=0 or bool(p[0])!=bool(p[1]):raise ValueError('invalid move weights; insert/delete must both be enabled or both disabled')
        if potential.algorithm not in ('cell','direct','cached'):raise ValueError('FastGCMC uses native CPU cell/direct kernels')
        if not 0<=int(seed)<2**64:raise ValueError('seed must fit an unsigned 64-bit integer')
        self.cell.flags.writeable=False;self.activities.flags.writeable=False;self.trials.flags.writeable=False
        self._lib=_bind(_load_native());self._closed=False
        self._owns_host=not isinstance(potential.host_evaluator,NativeEnvironment)
        if not self._owns_host:
            if potential.host_evaluator._closed:raise RuntimeError('potential host environment is closed')
            self._host=potential.host_evaluator
        else:
            self._host=NativeEnvironment(potential.host_positions,potential.host_sigma,potential.host_epsilon,potential.host_charge,
                 self.cell,potential.cutoff,coulomb='ewald-real' if potential.coulomb=='ewald' else potential.coulomb,alpha=potential.alpha,shifted=potential.shifted,
                 algorithm=potential.algorithm,threads=potential.threads)
        keep=[]
        def dp(a):
            x=np.ascontiguousarray(a,dtype=np.float64);keep.append(x);return _ptr(x)
        def ip(a):
            x=np.ascontiguousarray(a,dtype=np.int32);keep.append(x);return x.ctypes.data_as(_INT)
        exclusions=np.zeros((n,n),dtype=np.int32)
        for i,j in m.exclusions:exclusions[i,j]=exclusions[j,i]=1
        br=np.column_stack([m.bond_lengths,m.bond_lengths]) if m.bond_ranges is None else m.bond_ranges
        ar=np.column_stack([m.angles,m.angles]) if m.angle_ranges is None else m.angle_ranges
        torsions=m.torsions
        config=_Config(atoms=n,torsions=len(torsions),max_particles=self.max_particles,seed=int(seed),
             temperature=self.temperature,activity=float(self.activities[0]),cutoff=potential.cutoff,alpha=potential.alpha,
             max_translation=self.max_translation,max_rotation=self.max_rotation,scale14_lj=potential.scale14_lj,scale14_coulomb=potential.scale14_coulomb,
             shifted=int(potential.shifted),coulomb=({'none':0,'dsf':1,'ewald':3 if reciprocal_method=='pme' else 2}[potential.coulomb]),algorithm={'direct':0,'cell':1,'cached':2}[potential.algorithm],threads=potential.threads,rigid=int(m.is_rigid),
             cell=(C.c_double*9)(*self.cell.ravel()),move_weights=(C.c_double*5)(*p),
             sigma=dp(m.sigma),epsilon=dp(m.epsilon),charge=dp(m.charge),bond_lengths=dp(m.bond_lengths),angles=dp(m.angles),
             rigid_positions=dp(np.zeros((n,3)) if m.rigid_positions is None else m.rigid_positions),
             bond_ranges=dp(br),angle_ranges=dp(ar),bond_constants=dp(m.bond_force_constants),angle_constants=dp(m.angle_force_constants),
             parents=ip(m.parents),angle_refs=ip(m.angle_refs),dihedral_refs=ip(m.dihedral_refs),trials=ip(self.trials),exclusions=ip(exclusions),
             torsion_indices=ip([t['indices'] for t in torsions]),torsion_periodicity=ip([t['periodicity'] for t in torsions]),
             torsion_k=dp([t['k'] for t in torsions]),torsion_phase=dp([t.get('phase',0) for t in torsions]),host=self._host._handle,
             ewald_kmax=(C.c_int32*3)(*(map(int,potential.ewald.kmax) if potential.ewald is not None else (1,1,1))),
             pme_mesh=(C.c_int32*3)(*self.pme_mesh),reciprocal_method=int(reciprocal_method=='pme'),
             reciprocal_backend=0 if sampling_mode=='delayed' else {'cpu':0,'auto':1,'cuda':2}[reciprocal_backend],
             host_atoms=len(potential.host_positions),host_positions=dp(potential.host_positions),host_charge=dp(potential.host_charge),
             cuda_library=self.cuda_library.encode(),
             ewald_half_spectrum=int(self.ewald_half_spectrum),sampling_mode=int(self.sampling_mode=='delayed'),
             host_grid_mesh=(C.c_int32*3)(*self.host_grid_mesh),ewald_k_cutoff=self.ewald_k_cutoff,
             host_grid_cap=self.host_grid_cap,grid_root_bias=self.grid_root_bias,
             host_grid_cache=self.host_grid_cache.encode() if self.host_grid_cache else None,
             host_grid_key=self._grid_cache_key().encode(),host_grid_steep=int(self.host_grid_steep=='boltzmann'))
        self._handle=self._lib.poremc_fast_create(C.byref(config))
        if not self._handle:
            if self._owns_host:self._host.close()
            raise RuntimeError(self._lib.poremc_fast_last_error().decode())
        self._finalizer=weakref.finalize(self,self._lib.poremc_fast_destroy,self._handle)
        self.root_proposal=root_proposal
        if root_proposal is not None:
            if not np.array_equal(root_proposal.cell,self.cell):raise ValueError('root proposal cell differs')
            self._check(self._lib.poremc_fast_set_root(self._handle,*map(int,root_proposal.shape),_ptr(root_proposal.probabilities)))
        self.count_trace=[];self.energy_trace=[];self.sample_steps=[]
        self._frozen_fingerprint=self._fingerprint()
        self._backend_signature={'algorithm':potential.algorithm,'threads':potential.threads,'library_sha256':hashlib.sha256(Path(self._lib._name).read_bytes()).hexdigest(),
                                 'reciprocal_backend':self.long_range_profile.get('backend','none')}
        if 'cuda' in self._backend_signature['reciprocal_backend']:
            self._backend_signature['cuda_library_sha256']=hashlib.sha256(Path(self.cuda_library).read_bytes()).hexdigest()
        self._frozen_metadata=self._metadata()
    def _metadata(self):
        return json.dumps({'species':self.species[0].to_dict(),'cell':self.cell.tolist(),'temperature':self.temperature,
           'activities':self.activities.tolist(),'trials':self.trials.tolist(),'move_weights':self.move_weights,
           'max_translation':self.max_translation,'max_rotation':self.max_rotation,'max_particles':self.max_particles,
           'reciprocal_method':self.reciprocal_method,'pme_mesh':self.pme_mesh,'reciprocal_backend':self.reciprocal_backend,'cuda_library':self.cuda_library,
           'acceleration':self._acceleration_options()},sort_keys=True)
    def _acceleration_options(self):
        return dict(sampling_mode=self.sampling_mode,host_grid_mesh=self.host_grid_mesh,
                    host_grid_cap=self.host_grid_cap,host_grid_cache=self.host_grid_cache,
                    grid_root_bias=self.grid_root_bias,ewald_half_spectrum=self.ewald_half_spectrum,
                    ewald_k_cutoff=self.ewald_k_cutoff,host_grid_steep=self.host_grid_steep)
    def _grid_cache_key(self):
        """Only immutable field inputs: reuse across pressures, temperatures and seeds."""
        p=self.potential;m=self.species[0]
        fields={'format':'poremc-host-grid-v1','cell':self.cell.tolist(),
                'cutoff':p.cutoff,'alpha':p.alpha,'coulomb':p.coulomb,'shifted':p.shifted,
                'host_positions':p.host_positions.tolist(),'host_sigma':p.host_sigma.tolist(),
                'host_epsilon':p.host_epsilon.tolist(),'host_charge':p.host_charge.tolist(),
                'guest_sigma':m.sigma.tolist(),'guest_epsilon':m.epsilon.tolist(),'guest_charge':m.charge.tolist(),
                'reciprocal_method':self.reciprocal_method,'pme_mesh':self.pme_mesh,
                'kmax':p.ewald.kmax.tolist() if p.ewald is not None else None,
                'half_spectrum':self.ewald_half_spectrum,'k_cutoff':self.ewald_k_cutoff,
                'grid_mesh':self.host_grid_mesh,'cap':self.host_grid_cap}
        return hashlib.sha256(json.dumps(fields,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    def _ensure_config(self):
        if self._metadata()!=self._frozen_metadata:raise RuntimeError('FastGCMC configuration was mutated after native construction; construct a new engine')
    def _ensure(self):
        if self._closed:raise RuntimeError('FastGCMC is closed')
    def _check(self,code):
        if code:raise RuntimeError(self._lib.poremc_fast_last_error().decode())
    @property
    def step_count(self):
        self._ensure();return int(self._lib.poremc_fast_step_count(self._handle))
    @property
    def counts(self):
        self._ensure();return np.array([self._lib.poremc_fast_count(self._handle)],dtype=int)
    @property
    def particles(self):
        self._ensure();x=np.empty((int(self.counts[0]),self.species[0].n_atoms,3))
        self._check(self._lib.poremc_fast_positions(self._handle,_ptr(x)))
        return [Particle(0,p.copy()) for p in x]
    @property
    def counters(self):
        self._ensure();raw=np.empty((5,2),dtype=np.uint64)
        self._check(self._lib.poremc_fast_counters(self._handle,raw.ctypes.data_as(_U64)))
        return {name:{'attempted':int(raw[i,0]),'accepted':int(raw[i,1])} for i,name in enumerate(self.MOVE_NAMES)}
    def entries(self):return [(p.species,p.positions) for p in self.particles]
    def add_particle(self,species,positions):
        self._ensure();self._ensure_config()
        if int(species)!=0:raise ValueError('FastGCMC has only species 0')
        x=np.ascontiguousarray(self.species[0].validate_positions(positions),dtype=np.float64)
        self._check(self._lib.poremc_fast_add(self._handle,_ptr(x)))
    def total_energy(self):
        self._ensure();value=np.empty(1)
        self._check(self._lib.poremc_fast_total_energy(self._handle,_ptr(value)));return float(value[0])
    def _native_string(self,name):
        self._ensure();function=getattr(self._lib,name)
        size=function(self._handle,None,0)
        if not size:raise RuntimeError(self._lib.poremc_fast_last_error().decode())
        buffer=C.create_string_buffer(size)
        if not function(self._handle,buffer,size):raise RuntimeError(self._lib.poremc_fast_last_error().decode())
        return buffer.value.decode()
    @property
    def long_range_profile(self):
        """Actual reciprocal method/backend and native cache/work counters."""
        return json.loads(self._native_string('poremc_fast_longrange_info'))
    @property
    def acceleration_profile(self):
        """Native stage counters, fixed-host grid reuse and interpolation costs."""
        return json.loads(self._native_string('poremc_fast_acceleration_info'))
    def _excluded_index(self,exclude):
        if exclude is None:return C.c_size_t(-1).value
        i=int(exclude)
        if i!=exclude or not 0<=i<int(self.counts[0]):raise IndexError('excluded particle index is out of range')
        return i
    def trial_molecule_energies(self,trials,exclude=None):
        """Energy to insert each complete trial into the current environment.

        The selected existing particle may be excluded, as in a rotation or
        deletion retrace. This executes the same C++ batch path as MC and
        does not consume random numbers or commit a trial. PME returns its
        own discrete mesh Hamiltonian, with identical mesh in every trial.
        """
        self._ensure();x=np.ascontiguousarray(trials,dtype=np.float64)
        if x.ndim==2:x=x[None,:,:]
        if x.ndim!=3 or x.shape[1:]!=(self.species[0].n_atoms,3) or not np.isfinite(x).all():
            raise ValueError('trials must have shape (B,n_atoms,3) and finite coordinates')
        out=np.empty(len(x))
        self._check(self._lib.poremc_fast_molecule_energies(self._handle,self._excluded_index(exclude),len(x),_ptr(x),_ptr(out)))
        return out
    def trial_site_energies(self,prefix,trials,exclude=None):
        """Native CBMC atom energies including the charged growth prefix."""
        self._ensure();prefix=np.ascontiguousarray(prefix,dtype=np.float64).reshape(-1,3)
        x=np.ascontiguousarray(trials,dtype=np.float64).reshape(-1,3)
        if len(prefix)>=self.species[0].n_atoms or not np.isfinite(prefix).all() or not np.isfinite(x).all():
            raise ValueError('invalid prefix length or nonfinite candidate coordinates')
        out=np.empty(len(x))
        self._check(self._lib.poremc_fast_site_energies(self._handle,self._excluded_index(exclude),len(prefix),_ptr(prefix),len(x),_ptr(x),_ptr(out)))
        return out
    def trial_surrogate_molecule_energies(self,trials,exclude=None):
        """Reference energy U0 used by delayed CBMC, without target correction."""
        self._ensure();x=np.ascontiguousarray(trials,dtype=np.float64)
        if x.ndim==2:x=x[None,:,:]
        if x.ndim!=3 or x.shape[1:]!=(self.species[0].n_atoms,3) or not np.isfinite(x).all():
            raise ValueError('trials must have shape (B,n_atoms,3) and finite coordinates')
        out=np.empty(len(x))
        self._check(self._lib.poremc_fast_surrogate_molecule_energies(self._handle,self._excluded_index(exclude),len(x),_ptr(x),_ptr(out)))
        return out
    def trial_surrogate_site_energies(self,prefix,trials,exclude=None):
        """One reference-CBMC site increment, excluding proposal log densities."""
        self._ensure();prefix=np.ascontiguousarray(prefix,dtype=np.float64).reshape(-1,3)
        x=np.ascontiguousarray(trials,dtype=np.float64).reshape(-1,3)
        if len(prefix)>=self.species[0].n_atoms or not np.isfinite(prefix).all() or not np.isfinite(x).all():
            raise ValueError('invalid prefix or candidate coordinates')
        out=np.empty(len(x))
        self._check(self._lib.poremc_fast_surrogate_site_energies(self._handle,self._excluded_index(exclude),len(prefix),_ptr(prefix),len(x),_ptr(x),_ptr(out)))
        return out
    def run(self,steps,burnin=0,sample_every=10,adapt=False,record_energy=True):
        self._ensure();self._ensure_config();steps,burnin,sample_every=int(steps),int(burnin),int(sample_every)
        if min(steps,burnin)<0 or sample_every<1:raise ValueError('invalid run length')
        if adapt:raise ValueError('FastGCMC step adaptation is not implemented; choose fixed steps or Python GCMC burn-in adaptation')
        count=np.empty(steps//sample_every,dtype=np.uint64);energies=np.empty(len(count) if record_energy else 0);timing=np.empty(2)
        before=self.step_count
        self._check(self._lib.poremc_fast_run(self._handle,burnin,steps,sample_every,count.ctypes.data_as(_U64),_ptr(energies),int(record_energy),_ptr(timing)))
        trace=count[:,None].tolist();energy=energies.tolist();sample_steps=(before+burnin+sample_every*np.arange(1,len(count)+1)).tolist()
        self.count_trace.extend(trace);self.energy_trace.extend(energy);self.sample_steps.extend(sample_steps)
        sec=float(timing[0]);counters=self.counters
        return {'engine':'FastGCMC C++17','steps':steps,'burnin':burnin,'sample_every':sample_every,
             'production_seconds':sec,'elapsed_seconds':float(timing.sum()),'attempted_moves_per_second':steps/sec if sec>0 else 0,
             'counts':trace,'energies_kelvin':energy,'sample_steps':sample_steps,'final_counts':self.counts.tolist(),
             'acceptance':{m:dict(c,rate=c['accepted']/c['attempted'] if c['attempted'] else 0.) for m,c in counters.items()},
             'species_statistics':{self.species[0].name:summarize(count,sec)} if len(count) else {},
             'energy_statistics':summarize(energies,sec) if len(energies) else {},
             'adaptation_frozen':True,'delayed_acceptance':self.sampling_mode=='delayed','regrowth':'full molecule','rng':'std::mt19937_64',
             'long_range':self.long_range_profile,'acceleration':self.acceleration_profile}
    @property
    def profile(self):
        self._ensure();out=np.empty(3,dtype=np.uint64)
        self._check(self._lib.poremc_fast_profile(self._handle,out.ctypes.data_as(_U64)))
        return {**dict(zip(('guest_environment_builds','guest_environment_reuses','attempted_moves'),map(int,out))),
                'host':self._host.profile,'long_range':self.long_range_profile,'acceleration':self.acceleration_profile}
    def _fingerprint(self):
        if hasattr(self,'_frozen_fingerprint'):return self._frozen_fingerprint
        p=self.potential
        fields={'species':self.species[0].to_dict(),'cell':self.cell.tolist(),'cutoff':p.cutoff,'alpha':p.alpha,'coulomb':p.coulomb,
                'shifted':p.shifted,'scale14_lj':p.scale14_lj,'scale14_coulomb':p.scale14_coulomb,
                'host_positions':p.host_positions.tolist(),'host_sigma':p.host_sigma.tolist(),'host_epsilon':p.host_epsilon.tolist(),'host_charge':p.host_charge.tolist()}
        if p.coulomb=='ewald':
            fields.update(kmax=p.ewald.kmax.tolist(),background=p.ewald.background,
                          reciprocal_method=self.reciprocal_method,
                          pme_mesh=list(self.pme_mesh) if self.reciprocal_method=='pme' else None,
                          pme_order=4 if self.reciprocal_method=='pme' else None,
                          ewald_k_cutoff=self.ewald_k_cutoff)
        return hashlib.sha256(json.dumps(fields,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    def checkpoint(self):
        self._ensure();self._ensure_config();size=self._lib.poremc_fast_rng(self._handle,None,0);buffer=C.create_string_buffer(size)
        self._lib.poremc_fast_rng(self._handle,buffer,size)
        acceleration_counts=np.empty(20,dtype=np.uint64)
        self._check(self._lib.poremc_fast_acceleration_counters(self._handle,acceleration_counts.ctypes.data_as(_U64)))
        return {'schema':'poremc-fast-checkpoint-v3','root_proposal':self.root_proposal.to_dict() if self.root_proposal is not None else None,'species':self.species[0].to_dict(),'cell':self.cell.tolist(),
          'temperature':self.temperature,'activities':self.activities.tolist(),'trials':self.trials.tolist(),
          'move_weights':self.move_weights,'max_translation':self.max_translation,'max_rotation':self.max_rotation,
          'max_particles':self.max_particles,'particles':[p.to_dict() for p in self.particles],
          'rng_state':buffer.value.decode(),'step_count':self.step_count,'counters':self.counters,
          'long_range_state':self._native_string('poremc_fast_longrange_state'),
          'long_range_options':{'reciprocal_method':self.reciprocal_method,'reciprocal_backend':self.reciprocal_backend,
                                'pme_mesh':list(self.pme_mesh),'cuda_library':self.cuda_library if self._cuda_library_explicit else None},
          'acceleration_options':self._acceleration_options(),'acceleration_counters':acceleration_counts.tolist(),
          'potential_fingerprint':self._fingerprint(),'backend_signature':self._backend_signature,'count_trace':self.count_trace,'energy_trace':self.energy_trace,'sample_steps':self.sample_steps}
    def save_checkpoint(self,path):
        path=Path(path);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(self.checkpoint(),indent=2));tmp.replace(path)
    @classmethod
    def from_checkpoint(cls,path,potential):
        data=json.loads(Path(path).read_text())
        if data.get('schema') not in ('poremc-fast-checkpoint-v1','poremc-fast-checkpoint-v2','poremc-fast-checkpoint-v3'):raise ValueError('unsupported FastGCMC checkpoint')
        from .proposals import VoxelProposal
        obj=cls([Molecule.from_dict(data['species'])],potential,data['cell'],data['temperature'],data['activities'],
            trials=data['trials'],move_weights=data['move_weights'],max_translation=data['max_translation'],
            max_rotation=data['max_rotation'],max_particles=data['max_particles'],root_proposal=VoxelProposal.from_dict(data.get('root_proposal')),
            **data.get('long_range_options',{}),**data.get('acceleration_options',{}))
        if obj._fingerprint()!=data['potential_fingerprint']:
            obj.close();raise ValueError('checkpoint Hamiltonian fingerprint differs')
        if obj._backend_signature!=data.get('backend_signature'):
            obj.close();raise ValueError('exact FastGCMC restart requires the same backend, thread configuration and native binary')
        for particle in data['particles']:
            if int(particle['species'])!=0:raise ValueError('invalid checkpoint species')
            coordinates=np.ascontiguousarray(obj.species[0].validate_positions(particle['positions']),dtype=np.float64)
            obj._check(obj._lib.poremc_fast_add_raw(obj._handle,_ptr(coordinates)))
        if 'long_range_state' in data:
            obj._check(obj._lib.poremc_fast_restore_longrange(obj._handle,data['long_range_state'].encode()))
        counters=np.array([[data['counters'][m]['attempted'],data['counters'][m]['accepted']] for m in cls.MOVE_NAMES],dtype=np.uint64)
        obj._check(obj._lib.poremc_fast_restore_bookkeeping(obj._handle,data['rng_state'].encode(),data['step_count'],counters.ctypes.data_as(_U64)))
        if 'acceleration_counters' in data:
            acceleration_counts=np.ascontiguousarray(data['acceleration_counters'],dtype=np.uint64)
            if acceleration_counts.shape!=(20,):raise ValueError('invalid acceleration checkpoint counters')
            obj._check(obj._lib.poremc_fast_restore_acceleration_counters(obj._handle,acceleration_counts.ctypes.data_as(_U64)))
        obj.count_trace=data.get('count_trace',[]);obj.energy_trace=data.get('energy_trace',[]);obj.sample_steps=data.get('sample_steps',[])
        return obj
    def close(self):
        if not self._closed:
            self._finalizer()
            if self._owns_host:self._host.close()
            self._closed=True
    def __enter__(self):self._ensure();return self
    def __exit__(self,*args):self.close()
