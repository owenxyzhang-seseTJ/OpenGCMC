"""Equilibrium landscapes, frozen bias and log-space multistate reweighting.

No pseudocounts: unsampled bins remain infinite and explicitly masked.
MBAR Kish ESS is overlap, not a time-correlation-adjusted sample size.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True)
class HarmonicCountBias:
    coefficients: tuple
    center: float
    stiffness: float
    lower: float=-np.inf
    upper: float=np.inf
    def __post_init__(self):
        if self.stiffness<0 or not np.isfinite(self.stiffness) or not np.isfinite(self.center):raise ValueError('invalid bias')
        if self.lower>self.upper or not np.isfinite(self.coefficients).all():raise ValueError('invalid window')
    def __call__(self,counts):
        x=float(np.asarray(counts)@np.asarray(self.coefficients))
        return .5*self.stiffness*(x-self.center)**2 if self.lower<=x<=self.upper else np.inf
    def values(self,counts):
        x=np.asarray(counts)@np.asarray(self.coefficients)
        return np.where((x>=self.lower)&(x<=self.upper),.5*self.stiffness*(x-self.center)**2,np.inf)


def landscape(samples,edges,temperature=1.,log_weights=None,density=False):
    x=np.asarray(samples,float)
    if x.ndim==1:x=x[:,None]
    if x.ndim!=2 or len(x)==0 or not np.isfinite(x).all():raise ValueError('invalid samples')
    if len(edges)!=x.shape[1] or temperature<=0 or not np.isfinite(temperature):raise ValueError('invalid edges or T')
    es=[np.asarray(e,float) for e in edges]
    if any(e.ndim!=1 or len(e)<2 or not np.isfinite(e).all() or np.any(np.diff(e)<=0) for e in es):raise ValueError('edges must increase')
    # Prevent silent loss and renormalization of out-of-range samples.
    if any(np.any(x[:,i]<e[0]) or np.any(x[:,i]>e[-1]) for i,e in enumerate(es)):raise ValueError('histogram does not cover all samples')
    lw=np.zeros(len(x)) if log_weights is None else np.asarray(log_weights,float)
    if lw.shape!=(len(x),) or np.isnan(lw).any() or np.isposinf(lw).any() or not np.isfinite(lw).any():raise ValueError('invalid weights')
    w=np.exp(lw-logsumexp(lw));mass=np.histogramdd(x,bins=es,weights=w)[0]
    raw=np.histogramdd(x,bins=es)[0].astype(int);prob=mass.copy()
    if density:
        widths=np.ones(tuple(len(e)-1 for e in es))
        for i,e in enumerate(es):
            shape=[1]*len(es);shape[i]=len(e)-1;widths*=np.diff(e).reshape(shape)
        prob/=widths
    free=np.full(prob.shape,np.inf);mask=prob>0;free[mask]=-temperature*np.log(prob[mask]);free[mask]-=free[mask].min()
    return {'probability_mass':mass,'density' if density else 'probability':prob,
            'free_energy_kelvin':free,'sampled':mask,'counts':raw,
            'kish_ess':float(1/(w@w)),'edges':es}


class MBAR:
    """FP64 MBAR fixed-point solver with an explicit overlap diagnostic.

    u_kn is the complete reduced potential on a shared measure. At fixed T,
    shared U/T cancels, so u_kn=-lambda_k.N+b_k is sufficient.
    Correlated input is allowed for point estimates; use block_bootstrap for errors.
    """
    def __init__(self,u_kn,n_k,tolerance=1e-10,max_iterations=20000,min_gap=1e-8,compress=True):
        u=np.asarray(u_kn,float);n=np.asarray(n_k,int)
        if u.ndim!=2 or n.shape!=(u.shape[0],) or np.any(n<=0) or n.sum()!=u.shape[1]:raise ValueError('u_kn / n_k mismatch')
        if np.isnan(u).any() or np.isneginf(u).any() or np.any(~np.isfinite(u).any(axis=0)):raise ValueError('invalid reduced potentials')
        # Exact repeated-column compression, without quantization. This is
        # especially effective for count-biased windows and lattice models.
        if compress:
            unique, inverse, multiplicity = np.unique(u.T, axis=0, return_inverse=True, return_counts=True)
            work=unique.T; logmultiplicity=np.log(multiplicity)
        else:
            work=u; inverse=np.arange(u.shape[1]); logmultiplicity=np.zeros(u.shape[1])
        self.unique_reduced_potentials=work.shape[1]
        f=np.zeros(len(n));logn=np.log(n)
        for iteration in range(int(max_iterations)):
            den=logsumexp((logn+f)[:,None]-work,axis=0)
            new=-logsumexp(-work-den+logmultiplicity,axis=1);new-=new[0]
            if np.max(np.abs(new-f))<tolerance:f=new;break
            f=.5*f+.5*new
        else:raise RuntimeError('MBAR did not converge; check window overlap')
        self.u=u;self.n=n;self.f=f;self.iterations=iteration+1
        self.log_denominator=logsumexp((logn+f)[:,None]-work,axis=0)[inverse]
        self.W=np.exp(f[:,None]-u-self.log_denominator).T
        self.overlap=(self.W.T@self.W)*n[None,:]
        eig=np.linalg.eigvalsh((np.sqrt(n)[:,None]*(self.W.T@self.W))*np.sqrt(n)[None,:])
        self.spectral_gap=float(1-eig[-2]) if len(n)>1 else 1.
        if self.spectral_gap<min_gap:raise ValueError('disconnected or unresolved MBAR windows')
    def log_weights(self,u_target):
        target=np.asarray(u_target,float)
        if target.shape!=(self.u.shape[1],) or np.isnan(target).any() or np.isneginf(target).any():raise ValueError('invalid target')
        lw=-target-self.log_denominator
        if not np.isfinite(lw).any():raise ValueError('target has no support')
        return lw-logsumexp(lw)
    def expectation(self,values,u_target):
        w=np.exp(self.log_weights(u_target));x=np.asarray(values,float)
        if len(x)!=len(w) or not np.isfinite(x).all():raise ValueError('invalid observable')
        return {'mean':np.tensordot(w,x,axes=(0,0)), 'kish_ess':float(1/(w@w)),
                'weight_fraction':float(1/(w@w)/len(w))}


def block_bootstrap_mbar(u_kn,n_k,u_target,observable,block_lengths,replicates=100,seed=28):
    """Independent circular block resampling within each window; re-solves MBAR."""
    u=np.asarray(u_kn,float);nk=np.asarray(n_k,int);blocks=np.asarray(block_lengths,int)
    if blocks.shape!=nk.shape or np.any(blocks<1) or np.any(blocks>nk):raise ValueError('invalid block lengths')
    rng=np.random.default_rng(seed);offset=np.r_[0,np.cumsum(nk)];out=[]
    for _ in range(int(replicates)):
        ids=[]
        for j,n in enumerate(nk):
            b=blocks[j];starts=rng.integers(0,n,size=int(np.ceil(n/b)))
            ids.extend((offset[j]+((starts[:,None]+np.arange(b))%n).ravel()[:n]).tolist())
        ids=np.asarray(ids);fit=MBAR(u[:,ids],nk)
        out.append(fit.expectation(np.asarray(observable)[ids],np.asarray(u_target)[ids])['mean'])
    if not out:raise ValueError('at least one bootstrap replicate is required')
    a=np.asarray(out)
    return {'replicates':a,'ci95':np.percentile(a,[2.5,97.5],axis=0)}


def interfacial_landscape(count_coordinate,electrode_charges,voltage_pattern,
                          electrode_response,temperature,count_edges,charge_edges,log_weights=None):
    """Full P(n,Q): integrate the conditional Gaussian electrode fluctuations.

    BO-only histograms would omit these fluctuations. Returns captured mass
    without silently renormalizing tails outside the requested charge grid.
    """
    from scipy.special import ndtr
    n=np.asarray(count_coordinate,float).reshape(-1);q=np.asarray(electrode_charges,float);v=np.asarray(voltage_pattern,float)
    ne=np.asarray(count_edges,float);qe=np.asarray(charge_edges,float)
    if q.ndim!=2 or len(q)!=len(n) or len(n)==0 or v.shape!=(q.shape[1],) or temperature<=0:raise ValueError('invalid interfacial samples')
    if any(len(e)<2 or np.any(np.diff(e)<=0) or not np.isfinite(e).all() for e in [ne,qe]):raise ValueError('invalid edges')
    if n.min()<ne[0] or n.max()>ne[-1]:raise ValueError('count grid excludes samples')
    lw=np.zeros(len(n)) if log_weights is None else np.asarray(log_weights,float)
    if lw.shape!=(len(n),) or np.isnan(lw).any() or np.isposinf(lw).any() or not np.isfinite(lw).any():raise ValueError('invalid weights')
    w=np.exp(lw-logsumexp(lw));mu=q@v;variance=float(temperature*(v@electrode_response.P@v))
    if variance<=0:raise ValueError('conjugate charge must have positive conditional variance')
    mass=np.zeros((len(ne)-1,len(qe)-1));index=np.minimum(np.searchsorted(ne,n,side='right')-1,len(ne)-2)
    # Chunking avoids allocating the full (trajectory length x Q bins) array.
    for start in range(0,len(n),4096):
        stop=min(start+4096,len(n));cdf=ndtr((qe[None,:]-mu[start:stop,None])/np.sqrt(variance))
        np.add.at(mass,index[start:stop],w[start:stop,None]*np.diff(cdf,axis=1))
    density=mass/np.diff(qe)[None,:];free=np.full(mass.shape,np.inf);mask=mass>0
    free[mask]=-temperature*np.log(density[mask]);free[mask]-=free[mask].min()
    return {'probability_mass':mass,'charge_density':density,'free_energy_kelvin':free,
            'captured_mass':float(mass.sum()),'conditional_charge_variance_e2':variance,
            'count_edges':ne,'charge_edges':qe,'sampled_count_bins':np.bincount(index,minlength=len(ne)-1)>0}
