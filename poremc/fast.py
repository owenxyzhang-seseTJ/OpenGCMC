"""Full C++ GCMC loop for one classical rigid or tree-flexible component.

Supports LJ, DSF, rigid bodies, bounded or fixed bonds/angles, Fourier torsions,
CBMC insertion/deletion/full regrowth, translations and rotations. Ewald,
multi-component sampling, ML/delayed acceptance and online adaptation are
explicitly rejected; use :class:`poremc.engine.GCMC` for those workflows.
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
        ('torsion_k',_DOUBLE),('torsion_phase',_DOUBLE),('host',C.c_void_p)]


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
                 target_potential=None,species_probabilities=None,root_proposal=None):
        self.species=[species] if isinstance(species,Molecule) else list(species)
        if len(self.species)!=1:
            raise ValueError('FastGCMC supports exactly one component; use GCMC for mixtures')
        if type(potential) is not ClassicalPotential or potential.coulomb not in ('none','dsf') or target_potential is not None:
            raise ValueError('FastGCMC supports an exact ClassicalPotential with LJ/DSF only; use GCMC for Ewald, ML or delayed acceptance')
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
                 self.cell,potential.cutoff,coulomb=potential.coulomb,alpha=potential.alpha,shifted=potential.shifted,
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
             shifted=int(potential.shifted),coulomb=int(potential.coulomb=='dsf'),algorithm={'direct':0,'cell':1,'cached':2}[potential.algorithm],threads=potential.threads,rigid=int(m.is_rigid),
             cell=(C.c_double*9)(*self.cell.ravel()),move_weights=(C.c_double*5)(*p),
             sigma=dp(m.sigma),epsilon=dp(m.epsilon),charge=dp(m.charge),bond_lengths=dp(m.bond_lengths),angles=dp(m.angles),
             rigid_positions=dp(np.zeros((n,3)) if m.rigid_positions is None else m.rigid_positions),
             bond_ranges=dp(br),angle_ranges=dp(ar),bond_constants=dp(m.bond_force_constants),angle_constants=dp(m.angle_force_constants),
             parents=ip(m.parents),angle_refs=ip(m.angle_refs),dihedral_refs=ip(m.dihedral_refs),trials=ip(self.trials),exclusions=ip(exclusions),
             torsion_indices=ip([t['indices'] for t in torsions]),torsion_periodicity=ip([t['periodicity'] for t in torsions]),
             torsion_k=dp([t['k'] for t in torsions]),torsion_phase=dp([t.get('phase',0) for t in torsions]),host=self._host._handle)
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
        self._backend_signature={'algorithm':potential.algorithm,'threads':potential.threads,'library_sha256':hashlib.sha256(Path(self._lib._name).read_bytes()).hexdigest()}
        self._frozen_metadata=self._metadata()
    def _metadata(self):
        return json.dumps({'species':self.species[0].to_dict(),'cell':self.cell.tolist(),'temperature':self.temperature,
           'activities':self.activities.tolist(),'trials':self.trials.tolist(),'move_weights':self.move_weights,
           'max_translation':self.max_translation,'max_rotation':self.max_rotation,'max_particles':self.max_particles},sort_keys=True)
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
             'adaptation_frozen':True,'delayed_acceptance':False,'regrowth':'full molecule','rng':'std::mt19937_64'}
    @property
    def profile(self):
        self._ensure();out=np.empty(3,dtype=np.uint64)
        self._check(self._lib.poremc_fast_profile(self._handle,out.ctypes.data_as(_U64)))
        return {**dict(zip(('guest_environment_builds','guest_environment_reuses','attempted_moves'),map(int,out))),
                'host':self._host.profile}
    def _fingerprint(self):
        if hasattr(self,'_frozen_fingerprint'):return self._frozen_fingerprint
        p=self.potential
        fields={'species':self.species[0].to_dict(),'cell':self.cell.tolist(),'cutoff':p.cutoff,'alpha':p.alpha,'coulomb':p.coulomb,
                'shifted':p.shifted,'scale14_lj':p.scale14_lj,'scale14_coulomb':p.scale14_coulomb,
                'host_positions':p.host_positions.tolist(),'host_sigma':p.host_sigma.tolist(),'host_epsilon':p.host_epsilon.tolist(),'host_charge':p.host_charge.tolist()}
        return hashlib.sha256(json.dumps(fields,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    def checkpoint(self):
        self._ensure();self._ensure_config();size=self._lib.poremc_fast_rng(self._handle,None,0);buffer=C.create_string_buffer(size)
        self._lib.poremc_fast_rng(self._handle,buffer,size)
        return {'schema':'poremc-fast-checkpoint-v1','root_proposal':self.root_proposal.to_dict() if self.root_proposal is not None else None,'species':self.species[0].to_dict(),'cell':self.cell.tolist(),
          'temperature':self.temperature,'activities':self.activities.tolist(),'trials':self.trials.tolist(),
          'move_weights':self.move_weights,'max_translation':self.max_translation,'max_rotation':self.max_rotation,
          'max_particles':self.max_particles,'particles':[p.to_dict() for p in self.particles],
          'rng_state':buffer.value.decode(),'step_count':self.step_count,'counters':self.counters,
          'potential_fingerprint':self._fingerprint(),'backend_signature':self._backend_signature,'count_trace':self.count_trace,'energy_trace':self.energy_trace,'sample_steps':self.sample_steps}
    def save_checkpoint(self,path):
        path=Path(path);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(self.checkpoint(),indent=2));tmp.replace(path)
    @classmethod
    def from_checkpoint(cls,path,potential):
        data=json.loads(Path(path).read_text())
        if data.get('schema')!='poremc-fast-checkpoint-v1':raise ValueError('unsupported FastGCMC checkpoint')
        from .proposals import VoxelProposal
        obj=cls([Molecule.from_dict(data['species'])],potential,data['cell'],data['temperature'],data['activities'],
            trials=data['trials'],move_weights=data['move_weights'],max_translation=data['max_translation'],
            max_rotation=data['max_rotation'],max_particles=data['max_particles'],root_proposal=VoxelProposal.from_dict(data.get('root_proposal')))
        if obj._fingerprint()!=data['potential_fingerprint']:
            obj.close();raise ValueError('checkpoint Hamiltonian fingerprint differs')
        if obj._backend_signature!=data.get('backend_signature'):
            obj.close();raise ValueError('exact FastGCMC restart requires the same backend, thread configuration and native binary')
        for particle in data['particles']:
            if int(particle['species'])!=0:raise ValueError('invalid checkpoint species')
            coordinates=np.ascontiguousarray(obj.species[0].validate_positions(particle['positions']),dtype=np.float64)
            obj._check(obj._lib.poremc_fast_add_raw(obj._handle,_ptr(coordinates)))
        counters=np.array([[data['counters'][m]['attempted'],data['counters'][m]['accepted']] for m in cls.MOVE_NAMES],dtype=np.uint64)
        obj._check(obj._lib.poremc_fast_restore_bookkeeping(obj._handle,data['rng_state'].encode(),data['step_count'],counters.ctypes.data_as(_U64)))
        obj.count_trace=data.get('count_trace',[]);obj.energy_trace=data.get('energy_trace',[]);obj.sample_steps=data.get('sample_steps',[])
        return obj
    def close(self):
        if not self._closed:
            self._finalizer()
            if self._owns_host:self._host.close()
            self._closed=True
    def __enter__(self):self._ensure();return self
    def __exit__(self,*args):self.close()
