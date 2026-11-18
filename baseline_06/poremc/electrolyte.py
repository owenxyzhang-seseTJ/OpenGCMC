"""Primitive-electrolyte GCMC and constant-potential interfacial GCMC.

Point-ion species with Lennard-Jones cores, explicit stoichiometric exchange,
FP64 Ewald increments, cached native spatial environments and optional Gaussian
electrodes. Solvent is a uniform dielectric, not implicit claims of hydration.
Energy U/k_B is kelvin; valence is e; activity is inverse Angstrom cubed.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json,math,time
import numpy as np
from scipy.special import erfc,gammaln,logsumexp
from .cell import validate_cell,cutoff_image
from .electrostatics import Ewald
from .kernels import NativeEnvironment,ReferenceEnvironment
from .cache import content_key
from .units import COULOMB_KA,NA

@dataclass(frozen=True)
class Ion:
    name: str
    valence: float
    sigma: float=3.
    epsilon: float=50.
    def __post_init__(self):
        if not self.name or not np.isfinite([self.valence,self.sigma,self.epsilon]).all() or self.sigma<0 or self.epsilon<0:raise ValueError('invalid ion')

@dataclass(frozen=True)
class ExchangeChannel:
    name: str
    stoichiometry: tuple
    probability: float=1.
    insert_probability: float=.5
    def __post_init__(self):
        n=np.asarray(self.stoichiometry)
        if n.ndim!=1 or np.any(n<0) or not np.array_equal(n,n.astype(int)) or n.sum()==0:raise ValueError('stoichiometry must be nonnegative integers')
        if self.probability<=0 or not np.isfinite(self.probability) or not 0<self.insert_probability<1:raise ValueError('invalid channel probabilities')

class FractionalRegion:
    """Uniform full cell or a fractional slit. Restricted axes never wrap."""
    def __init__(self,cell,lower=(0,0,0),upper=(1,1,1)):
        self.cell=np.asarray(cell,float);self.inverse=np.linalg.inv(self.cell)
        self.lower=np.asarray(lower,float);self.upper=np.asarray(upper,float)
        if self.cell.shape!=(3,3) or self.lower.shape!=(3,) or self.upper.shape!=(3,) or not np.isfinite(self.cell).all() or np.any(self.lower<0) or np.any(self.upper>1) or np.any(self.upper<=self.lower):raise ValueError('invalid region')
        self.volume=float(np.linalg.det(self.cell)*np.prod(self.upper-self.lower))
        if self.volume<=0:raise ValueError('nonpositive volume')
        self.periodic=(self.lower==0)&(self.upper==1)
    def sample(self,rng,n=1):return (self.lower+(self.upper-self.lower)*rng.random((n,3)))@self.cell
    def wrap(self,positions):
        f=np.asarray(positions,float)@self.inverse;f[...,self.periodic]-=np.floor(f[...,self.periodic]);return f@self.cell
    def contains(self,positions):
        f=np.asarray(positions,float).reshape(-1,3)@self.inverse
        return bool(np.all(f>=self.lower-1e-12) and np.all(f<self.upper+1e-12))


def log_exchange_ratio(counts,stoichiometry,log_activities,log_densities,delta_u,temperature,insert=True,insert_probability=.5,delta_bias=0.):
    """Uniform-order independent proposals; includes all factorial / q factors.

    log_densities contains one spatial log proposal density per inserted or
    deleted particle. No extra nu! for ordered selection without replacement.
    """
    n=np.asarray(counts,int);nu=np.asarray(stoichiometry,int);la=np.asarray(log_activities,float)
    if n.shape!=nu.shape or la.shape!=nu.shape or temperature<=0 or np.any(n<0) or np.any(nu<0) or not 0<insert_probability<1:raise ValueError('invalid exchange inputs')
    if len(log_densities)!=nu.sum():raise ValueError('one density per ion required')
    if not insert and np.any(n<nu):return -np.inf
    if insert:
        pref=float(nu@la-np.sum(gammaln(n+nu+1)-gammaln(n+1))-np.sum(log_densities))
        pref+=math.log((1-insert_probability)/insert_probability)
    else:
        pref=float(-nu@la+np.sum(gammaln(n+1)-gammaln(n-nu+1))+np.sum(log_densities))
        pref+=math.log(insert_probability/(1-insert_probability))
    return pref-float(delta_u)/temperature-float(delta_bias)


def ideal_salt_distribution(stoichiometry,log_z_volume,max_units=150,reference_counts=None):
    nu=np.asarray(stoichiometry,int);lz=np.asarray(log_z_volume,float)
    n0=np.zeros_like(nu) if reference_counts is None else np.asarray(reference_counts,int)
    ns=np.arange(max_units+1);counts=n0+ns[:,None]*nu
    logp=counts@lz-gammaln(counts+1).sum(axis=1);p=np.exp(logp-logsumexp(logp))
    if p[-1]>1e-10:raise ValueError('increase maximum units')
    mean=p@counts;d=counts-mean
    return {'units':ns,'counts':counts,'probability':p,'mean':mean,'covariance':(d*p[:,None]).T@d}


def log_activity_from_molarity(concentration_mol_L,excess_mu_kelvin,temperature):
    """z = rho exp(mu_ex/kBT); requires a bulk calibration with the SAME model."""
    c=np.asarray(concentration_mol_L,float)
    if np.any(c<=0) or not np.isfinite(c).all() or temperature<=0:raise ValueError('positive concentration and T required')
    return np.log(c*NA*1e-27)+np.asarray(excess_mu_kelvin)/temperature


class IonicHamiltonian:
    def __init__(self,species,cell,cutoff=8.,alpha=.3,kmax=5,dielectric=78.4,coulomb='ewald',
                 shifted=True,backend='native',algorithm='cached',electrodes=None,
                 host_positions=None,host_sigma=None,host_epsilon=None,host_charge=None,slab=False):
        self.species=list(species);self.cell,self.inverse,_=validate_cell(cell,cutoff)
        self.cutoff=float(cutoff);self.alpha=float(alpha);self.kmax=np.broadcast_to(kmax,(3,)).astype(int)
        self.dielectric=float(dielectric);self.shifted=bool(shifted);self.coulomb=coulomb;self.slab=bool(slab)
        if not len(species) or dielectric<=0 or not np.isfinite(dielectric) or coulomb not in ('none','ewald'):raise ValueError('invalid Hamiltonian')
        self.valences=np.array([s.valence for s in species]);self.sigma=np.array([s.sigma for s in species]);self.epsilon=np.array([s.epsilon for s in species])
        self.host_positions=np.asarray([] if host_positions is None else host_positions,float).reshape(-1,3)
        m=len(self.host_positions)
        self.host_sigma=np.zeros(m) if host_sigma is None else np.asarray(host_sigma,float)
        self.host_epsilon=np.zeros(m) if host_epsilon is None else np.asarray(host_epsilon,float)
        self.host_charge=np.zeros(m) if host_charge is None else np.asarray(host_charge,float)
        if any(x.shape!=(m,) or not np.isfinite(x).all() for x in (self.host_sigma,self.host_epsilon,self.host_charge)):raise ValueError('invalid host')
        if np.any(self.host_sigma<0) or np.any(self.host_epsilon<0):raise ValueError('invalid host LJ')
        self.K=COULOMB_KA/dielectric
        self.ewald=Ewald(cell,cutoff,alpha,kmax,background=True) if coulomb=='ewald' else None
        self.electrodes=electrodes
        if electrodes is not None:
            e=electrodes
            if coulomb!='ewald' or not np.allclose(e.cell,self.cell) or e.dielectric!=dielectric or e.slab!=slab or e.ewald.alpha!=alpha or e.ewald.cutoff!=cutoff or not np.array_equal(e.ewald.kmax,self.kmax):raise ValueError('electrode and ionic electrostatic kernels must match')
        if slab and (self.ewald is None or not np.allclose(self.cell,np.diag(np.diag(self.cell)))):raise ValueError('slab requires orthorhombic Ewald geometry')
        if backend not in ('native','reference'):raise ValueError('new ionic path supports native or reference CPU')
        self.Env=NativeEnvironment if backend=='native' else ReferenceEnvironment
        self.backend=backend;self.algorithm=algorithm
        self.host_env=self._environment(self.host_positions,self.host_sigma,self.host_epsilon,self.host_charge)
        self.host_structure=self._structure(self.host_positions,self.host_charge)
        self.host_b=electrodes.fields(self.host_positions,self.host_charge).sum(axis=0) if electrodes else None
        self._env=None;self.positions=np.empty((0,3));self.types=np.empty(0,int)
        self.profile={'environment_builds':0,'environment_reuses':0,'proposals':0,'commits':0,'electrode_matrix_factorizations':int(electrodes is not None)}
        self.version=0;self.set_state([],np.empty((0,3)))
    def _environment(self,p,s,e,q):
        return self.Env(p,s,e,np.asarray(q)/np.sqrt(self.dielectric),self.cell,self.cutoff,
                        coulomb='ewald-real' if self.ewald else 'none',alpha=self.alpha,
                        shifted=self.shifted,algorithm=self.algorithm,threads=1)
    def _structure(self,p,q):
        return np.sum(np.asarray(q)[:,None]*np.exp(1j*(np.asarray(p).reshape(-1,3)@self.ewald.k.T)),axis=0) if self.ewald else np.empty(0,complex)
    def _guest_env(self):
        if self._env is None:
            self._env=self._environment(self.positions,self.sigma[self.types],self.epsilon[self.types],self.valences[self.types]);self.profile['environment_builds']+=1
        else:self.profile['environment_reuses']+=1
        return self._env
    def _pairs(self,p,t,x,u):
        """One small explicit cross block, used only for the changed atoms."""
        if not len(p) or not len(x):return 0.
        d=cutoff_image(p[:,None,:]-x[None,:,:],self.cell);r=np.linalg.norm(d,axis=2)
        sig=(self.sigma[t,None]+self.sigma[u][None,:])/2;eps=np.sqrt(self.epsilon[t,None]*self.epsilon[u][None,:])
        qq=self.valences[t,None]*self.valences[u][None,:] if self.ewald else np.zeros_like(r)
        active=r<self.cutoff;interaction=(eps*sig!=0)|(qq!=0)
        if np.any((r<1e-12)&active&interaction):return np.inf
        safe=np.where(active&(r>1e-12),r,1.);s6=(sig/safe)**6
        en=4*eps*(s6*s6-s6)
        if self.shifted:
            c=(sig/self.cutoff)**6;en-=4*eps*(c*c-c)
        if self.ewald:en+=self.K*qq*erfc(self.alpha*safe)/safe
        return float(np.sum(np.where(active,en,0)))
    def _within(self,p,t):
        return sum(self._pairs(p[i:i+1],t[i:i+1],p[:i],t[:i]) for i in range(len(p)))
    def _against(self,env,p,t):
        if not len(p):return 0.
        return sum(float(env.energy(x[None,None,:],[self.sigma[s]],[self.epsilon[s]],[self.valences[s]/np.sqrt(self.dielectric)])[0]) for x,s in zip(p,t))
    def set_state(self,types,positions):
        t=np.asarray(types,int).reshape(-1);p=np.asarray(positions,float).reshape(-1,3)
        if len(p)!=len(t) or np.any(t<0) or np.any(t>=len(self.species)) or not np.isfinite(p).all():raise ValueError('invalid ionic state')
        self.positions=p.copy();self.types=t.copy();q=self.valences[t]
        self.structure=self.host_structure+self._structure(p,q)
        self.total_charge=float(self.host_charge.sum()+q.sum())
        self.dipole_z=float(self.host_charge@self.host_positions[:,2]+q@p[:,2])
        self.fields=self.electrodes.fields(p,q) if self.electrodes else None
        self.electrode_state=self.electrodes.response.solve(self.host_b+self.fields.sum(axis=0),self.electrodes.potentials,-self.total_charge) if self.electrodes else None
        if self._env is not None:self._env.close()
        self._env=None;self.energy=self.full_energy(t,p);self.version+=1
        if not np.isfinite(self.energy):raise ValueError('initial state has nonfinite energy')
    def full_energy(self,types=None,positions=None):
        t=self.types if types is None else np.asarray(types,int);p=self.positions if positions is None else np.asarray(positions,float).reshape(-1,3)
        # Independent energy path: direct LJ reference + full Ewald, not delta bookkeeping.
        with ReferenceEnvironment(self.host_positions,self.host_sigma,self.host_epsilon,np.zeros(len(self.host_charge)),self.cell,self.cutoff,shifted=self.shifted,algorithm='direct') as env:
            energy=self._against(env,p,t)
        for i in range(len(t)):
            if i:
                with ReferenceEnvironment(p[:i],self.sigma[t[:i]],self.epsilon[t[:i]],np.zeros(i),self.cell,self.cutoff,shifted=self.shifted,algorithm='direct') as env:
                    energy+=self._against(env,p[i:i+1],t[i:i+1])
        q=self.valences[t];P=np.concatenate([self.host_positions,p]);Q=np.r_[self.host_charge,q]
        if self.ewald:energy+=(self.ewald.energy(P,Q)-self.ewald.energy(self.host_positions,self.host_charge))/self.dielectric
        if self.slab:energy+=2*np.pi*self.K/self.ewald.volume*((Q@P[:,2])**2-(self.host_charge@self.host_positions[:,2])**2)
        if self.electrodes:energy+=self.electrodes.solve(P,Q)['free_energy']
        return float(energy)
    def propose(self,remove=(),add_types=(),add_positions=None):
        remove=np.asarray(remove,int);at=np.asarray(add_types,int).reshape(-1)
        ap=np.asarray([] if add_positions is None else add_positions,float).reshape(-1,3)
        if len(ap)!=len(at) or len(set(remove.tolist()))!=len(remove) or np.any(remove<0) or np.any(remove>=len(self.types)) or np.any(at<0) or np.any(at>=len(self.species)) or not np.isfinite(ap).all():raise ValueError('invalid proposal')
        self.profile['proposals']+=1;rp=self.positions[remove];rt=self.types[remove]
        ideal=self.ewald is None and not np.any(self.epsilon*self.sigma) and not np.any(self.host_epsilon*self.host_sigma)
        short=0.
        if not ideal:
            env=self._guest_env();add_survivors=0.
            # Never subtract the spurious new--old self interaction: nearly
            # coincident translations would catastrophically cancel LJ energies.
            if len(ap) and len(remove)>0 and np.array_equal(np.sort(remove),np.arange(int(np.min(remove)),int(np.max(remove))+1)) and hasattr(env,'energy_excluding'):
                j=int(np.min(remove)); stop=int(np.max(remove))+1
                for x,s in zip(ap,at):
                    add_survivors+=float(env.energy_excluding(x[None,None,:],[self.sigma[s]],[self.epsilon[s]],[self.valences[s]/np.sqrt(self.dielectric)],j,stop)[0])
            elif len(ap) and len(remove):
                keep=np.ones(len(self.types),bool);keep[remove]=False
                with self._environment(self.positions[keep],self.sigma[self.types[keep]],self.epsilon[self.types[keep]],self.valences[self.types[keep]]) as remaining:
                    add_survivors=self._against(remaining,ap,at)
            else:add_survivors=self._against(env,ap,at)
            remove_all=0.
            for idx,x,s in zip(remove,rp,rt):
                if hasattr(env,'energy_excluding'):
                    remove_all+=float(env.energy_excluding(x[None,None,:],[self.sigma[s]],[self.epsilon[s]],[self.valences[s]/np.sqrt(self.dielectric)],int(idx),int(idx)+1)[0])
                else:
                    keep=np.arange(len(self.types))!=idx
                    with self._environment(self.positions[keep],self.sigma[self.types[keep]],self.epsilon[self.types[keep]],self.valences[self.types[keep]]) as sub:
                        remove_all+=self._against(sub,x[None,:],np.array([s]))
            short=add_survivors+self._within(ap,at)-remove_all+self._within(rp,rt)
            if len(self.host_positions):short+=self._against(self.host_env,ap,at)-self._against(self.host_env,rp,rt)
        aq=self.valences[at];rq=self.valences[rt];dQ=float(aq.sum()-rq.sum());dM=float(aq@ap[:,2]-rq@rp[:,2])
        ds=self._structure(ap,aq)-self._structure(rp,rq);du=short
        if self.ewald:
            e=self.ewald
            du+=self.K*(np.sum(e.coefficient*(2*np.real(self.structure.conj()*ds)+np.abs(ds)**2))-e.alpha/np.sqrt(np.pi)*(aq@aq-rq@rq)-np.pi*(2*self.total_charge*dQ+dQ*dQ)/(2*e.alpha**2*e.volume))
        if self.slab:du+=2*np.pi*self.K/self.ewald.volume*(2*self.dipole_z*dM+dM*dM)
        af=None;estate=None
        if self.electrodes:
            af=self.electrodes.fields(ap,aq);db=af.sum(axis=0)-self.fields[remove].sum(axis=0)
            df,estate=self.electrodes.response.delta(self.electrode_state,db,-dQ);du+=df
        if np.isnan(du) or du==-np.inf:raise FloatingPointError('invalid energy difference')
        return {'remove':remove,'add_types':at,'add_positions':ap,'delta_energy':float(du),'delta_structure':ds,'delta_charge':dQ,'delta_dipole':dM,'add_fields':af,'electrode_state':estate,'version':self.version}
    def commit(self,trial):
        if trial['version']!=self.version:raise ValueError('stale proposal')
        if not np.isfinite(trial['delta_energy']):raise ValueError('cannot commit invalid energy')
        keep=np.ones(len(self.types),bool);keep[trial['remove']]=False
        self.types=np.r_[self.types[keep],trial['add_types']];self.positions=np.concatenate([self.positions[keep],trial['add_positions']])
        self.structure+=trial['delta_structure'];self.total_charge+=trial['delta_charge'];self.dipole_z+=trial['delta_dipole'];self.energy+=trial['delta_energy']
        if self.electrodes:
            self.fields=np.concatenate([self.fields[keep],trial['add_fields']]);self.electrode_state=trial['electrode_state']
        if self._env is not None:self._env.close()
        self._env=None;self.version+=1;self.profile['commits']+=1
    def fingerprint(self):
        meta={'kind':'ionic-hamiltonian-v1','species':[s.__dict__ for s in self.species], 'cutoff':self.cutoff,'alpha':self.alpha,'kmax':self.kmax.tolist(),'dielectric':self.dielectric,'coulomb':self.coulomb,'shifted':self.shifted,'slab':self.slab}
        arrays={'cell':self.cell,'host':self.host_positions,'hs':self.host_sigma,'he':self.host_epsilon,'hq':self.host_charge}
        if self.electrodes:
            meta.update({'width':self.electrodes.width,'hardness':self.electrodes.hardness})
            arrays.update({'electrodes':self.electrodes.positions,'potentials':self.electrodes.potentials,'A':self.electrodes.response.A})
        return content_key(meta,arrays)
    def close(self):
        if self._env is not None:self._env.close();self._env=None
        self.host_env.close()


class ElectrolyteGCMC:
    def __init__(self,hamiltonian,temperature,log_activities,channels,seed=123,region=None,
                 exchange_probability=.6,max_translation=1.5,bias=None,allow_single_ion=False,
                 max_particles=100000,proposals=None):
        self.h=hamiltonian;self.temperature=float(temperature);self.log_activities=np.asarray(log_activities,float)
        self.channels=list(channels);self.rng=np.random.default_rng(seed)
        self.region=region or FractionalRegion(self.h.cell);self.bias=bias;self.allow_single_ion=bool(allow_single_ion)
        if temperature<=0 or not np.isfinite(temperature) or self.log_activities.shape!=(len(self.h.species),) or not np.isfinite(self.log_activities).all():raise ValueError('invalid T / activities')
        if not self.channels or not 0<=exchange_probability<=1 or max_translation<=0 or max_particles<1:raise ValueError('invalid moves')
        if not np.allclose(self.region.cell,self.h.cell):raise ValueError('region cell mismatch')
        for c in self.channels:
            if len(c.stoichiometry)!=len(self.h.species):raise ValueError('stoichiometric dimension mismatch')
            if abs(np.dot(c.stoichiometry,self.h.valences))>1e-10 and not (allow_single_ion and self.h.electrodes):raise ValueError('charged exchange requires compensating constant-potential electrodes')
        if allow_single_ion and self.h.electrodes is None:raise ValueError('single-ion reservoir needs electrodes')
        self.channel_probs=np.array([c.probability for c in self.channels]);self.channel_probs/=self.channel_probs.sum()
        self.exchange_probability=exchange_probability;self.max_translation=max_translation;self.max_particles=int(max_particles)
        self.proposals={} if proposals is None else dict(proposals)
        if self.proposals and not np.all(self.region.periodic):raise ValueError('voxel proposals currently require the full periodic cell')
        for s,p in self.proposals.items():
            if not 0<=s<len(self.h.species) or not np.allclose(p.cell,self.h.cell):raise ValueError('invalid species proposal')
        self.steps=0;self.counters={'insert':[0,0],'delete':[0,0],'translate':[0,0]}
        self.check_charge()
    @property
    def counts(self):return np.bincount(self.h.types,minlength=len(self.h.species))
    def check_charge(self):
        Q=self.h.total_charge
        if self.h.electrodes:Q+=self.h.electrode_state['q'].sum()
        if abs(Q)>1e-8:raise ValueError('complete state is not electroneutral')
        if not self.region.contains(self.h.positions):raise ValueError('ions outside sampled region')
    def _bias(self,counts):return float(self.bias(counts)) if self.bias is not None else 0.
    def _logq(self,s,x):
        return float(self.proposals[s].log_relative_density(x)-np.log(np.linalg.det(self.h.cell))) if s in self.proposals else -math.log(self.region.volume)
    def _sample(self,s):return self.proposals[s].sample(self.rng,1)[0] if s in self.proposals else self.region.sample(self.rng,1)[0]
    def _accept(self,ratio):
        if np.isnan(ratio):raise FloatingPointError('NaN acceptance')
        return math.log(max(self.rng.random(),np.finfo(float).tiny))<min(0.,ratio)
    def step(self):
        self.steps+=1;n=self.counts;b0=self._bias(n)
        if not np.isfinite(b0):raise ValueError('initial/current state lies outside bias window')
        if self.rng.random()<self.exchange_probability:
            c=self.channels[int(self.rng.choice(len(self.channels),p=self.channel_probs))];nu=np.array(c.stoichiometry,int)
            insert=self.rng.random()<c.insert_probability;name='insert' if insert else 'delete';self.counters[name][0]+=1
            if insert:
                if len(self.h.types)+nu.sum()>self.max_particles:raise RuntimeError('increase max_particles; capacity is not a physical constraint')
                types=np.repeat(np.arange(len(n)),nu);pos=np.array([self._sample(int(s)) for s in types]);logq=[self._logq(int(s),x) for s,x in zip(types,pos)]
                newn=n+nu;remove=[]
            else:
                if np.any(n<nu):return False
                remove=np.concatenate([self.rng.choice(np.flatnonzero(self.h.types==s),v,replace=False) for s,v in enumerate(nu)]).astype(int)
                logq=[self._logq(int(self.h.types[i]),self.h.positions[i]) for i in remove]
                newn=n-nu;types=[];pos=np.empty((0,3))
            db=self._bias(newn)-b0
            if not np.isfinite(db):return False
            trial=self.h.propose(remove,types,pos)
            ratio=log_exchange_ratio(n,nu,self.log_activities,logq,trial['delta_energy'],self.temperature,insert,c.insert_probability,db)
        else:
            name='translate';self.counters[name][0]+=1
            if not len(self.h.types):return False
            i=int(self.rng.integers(len(self.h.types)));new=self.region.wrap(self.h.positions[i]+self.rng.uniform(-self.max_translation,self.max_translation,3))
            if not self.region.contains(new):return False
            trial=self.h.propose([i],[self.h.types[i]],new[None,:]);ratio=-trial['delta_energy']/self.temperature
        if self._accept(ratio):
            self.h.commit(trial);self.counters[name][1]+=1;return True
        return False
    def run(self,steps,burnin=0,sample_every=10,record_positions=False):
        if steps<1 or burnin<0 or sample_every<1:raise ValueError('invalid run lengths')
        for _ in range(int(burnin)):self.step()
        counts=[];energy=[];charges=[];positions=[];saved_types=[];t0=time.perf_counter()
        for i in range(int(steps)):
            self.step()
            if (i+1)%sample_every==0:
                counts.append(self.counts.copy());energy.append(self.h.energy)
                if self.h.electrodes:charges.append(self.h.electrode_state['q'].copy())
                if record_positions:positions.append(self.h.positions.copy());saved_types.append(self.h.types.copy())
        elapsed=time.perf_counter()-t0
        self.check_charge()
        return {'counts':np.asarray(counts),'energy_kelvin':np.asarray(energy),'electrode_charges':np.asarray(charges),
                'positions':positions,'types':saved_types,'wall_seconds':elapsed,'attempts':int(steps),
                'moves_per_second':steps/elapsed,'counters':{k:v.copy() for k,v in self.counters.items()},'profile':dict(self.h.profile)}
    def _model_key(self):
        return content_key({'h':self.h.fingerprint(),'T':self.temperature,'channels':[c.__dict__ for c in self.channels],
                           'exchange_probability':self.exchange_probability,'max_translation':self.max_translation,
                           'allow_single_ion':self.allow_single_ion,'bias':None if self.bias is None else {k:(str(v) if isinstance(v,float) and not np.isfinite(v) else v) for k,v in self.bias.__dict__.items()},
                           'proposals':{str(s):p.to_dict() for s,p in self.proposals.items()}},
                          {'activities':self.log_activities,'lower':self.region.lower,'upper':self.region.upper})
    def save_checkpoint(self,path):
        payload={'schema':'poremc-ion-state-v1','model':self._model_key(),'types':self.h.types.tolist(),'positions':self.h.positions.tolist(),
                 'rng':self.rng.bit_generator.state,'steps':self.steps,'counters':self.counters}
        Path(path).write_text(json.dumps(payload,indent=2),encoding='utf8')
    def restore_checkpoint(self,path):
        p=json.loads(Path(path).read_text(encoding='utf8'))
        if p.get('schema')!='poremc-ion-state-v1' or p['model']!=self._model_key():raise ValueError('checkpoint Hamiltonian/ensemble mismatch')
        self.h.set_state(p['types'],p['positions']);self.check_charge()
        if not np.isfinite(self._bias(self.counts)):raise ValueError('checkpoint outside window')
        self.rng.bit_generator.state=p['rng'];self.steps=int(p['steps']);self.counters=p['counters']
