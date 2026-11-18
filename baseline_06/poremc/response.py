"""Chemical-potential response on a fixed-temperature equilibrium ensemble.

lambda_i=log(z_i [A^-3] * 1 A^3). d<N>/d lambda = population covariance.
Time correlation changes the uncertainty, NOT the thermodynamic covariance.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp
from .statistics import integrated_autocorrelation_time


def _matrix(x):
    x=np.asarray(x,dtype=float)
    if x.ndim==1:x=x[:,None]
    if x.ndim!=2 or len(x)==0 or not np.isfinite(x).all():
        raise ValueError('finite nonempty (samples, observables) array required')
    return x

@dataclass
class MomentAccumulator:
    """Mergeable FP64 centered moments; O(d^2) storage, no trajectory needed."""
    dimension: int
    def __post_init__(self):
        if self.dimension<1:raise ValueError('dimension must be positive')
        self.n=0;self.mean=np.zeros(self.dimension);self.M2=np.zeros((self.dimension,self.dimension))
    def update(self,x):
        x=_matrix(x)
        if x.shape[1]!=self.dimension:raise ValueError('dimension mismatch')
        other=MomentAccumulator(self.dimension);other.n=len(x);other.mean=x.mean(axis=0)
        d=x-other.mean;other.M2=d.T@d
        return self.merge(other)
    def merge(self,other):
        if other.dimension!=self.dimension:raise ValueError('dimension mismatch')
        if not other.n:return self
        if not self.n:
            self.n=other.n;self.mean=other.mean.copy();self.M2=other.M2.copy();return self
        n=self.n+other.n;d=other.mean-self.mean
        self.M2+=other.M2+np.outer(d,d)*(self.n*other.n/n)
        self.mean+=d*(other.n/n);self.n=n;return self
    @property
    def covariance(self):
        if self.n==0:raise ValueError('no observations')
        return (self.M2+self.M2.T)/(2*self.n)


def weighted_response(counts,log_weights=None):
    x=_matrix(counts)
    logs=np.zeros(len(x)) if log_weights is None else np.asarray(log_weights,float)
    if logs.shape!=(len(x),) or np.isnan(logs).any() or np.isposinf(logs).any() or not np.isfinite(logs).any():
        raise ValueError('invalid log weights')
    w=np.exp(logs-logsumexp(logs));mean=w@x;d=x-mean
    covariance=(d*w[:,None]).T@d
    return {'mean':mean,'response':(covariance+covariance.T)*.5,
            'kish_ess':float(1/(w@w)),'weights':w}


def response_summary(counts,temperature=None,block_length=None,bootstrap=200,seed=17):
    """Stationary single-chain response, moving-block bootstrap and diagnostics.

    Production observations only; burn-in/adaptive bias frames must be excluded.
    Bootstrap uncertainty is for the covariance itself, using correlated products.
    """
    x=_matrix(counts);n,d=x.shape;r=weighted_response(x)
    centered=x-r['mean'];products=np.stack([centered[:,i]*centered[:,j] for i in range(d) for j in range(i,d)],axis=1)
    taus=np.array([integrated_autocorrelation_time(a) for a in np.concatenate([x,products],axis=1).T])
    finite=taus[np.isfinite(taus)];tmax=float(finite.max()) if len(finite) else 1.
    if block_length is None:block_length=min(n,max(1,int(np.ceil(5*tmax))))
    block_length=int(block_length)
    if block_length<1 or block_length>n:raise ValueError('block_length must be in [1,n]')
    blocks=n//block_length;warnings=[]
    if blocks<8:warnings.append('fewer than eight blocks: covariance uncertainty is poorly resolved')
    if block_length<2*tmax:warnings.append('block length shorter than twice the largest estimated correlation time')
    rng=np.random.default_rng(seed);replicas=[]
    if bootstrap<0:raise ValueError('bootstrap must be nonnegative')
    # Circular moving-block bootstrap preserves local serial dependence.
    for _ in range(int(bootstrap)):
        starts=rng.integers(0,n,size=int(np.ceil(n/block_length)))
        ids=((starts[:,None]+np.arange(block_length))%n).ravel()[:n]
        replicas.append(weighted_response(x[ids])['response'])
    ci=np.percentile(replicas,[2.5,97.5],axis=0) if replicas else np.full((2,d,d),np.nan)
    out={'samples':n,'mean':r['mean'],'response':r['response'],'response_ci95':ci,
         'eigenvalues':np.linalg.eigvalsh(r['response']), 'block_length':block_length,
         'full_blocks':blocks,'tau_observables_and_products':taus,'warnings':warnings}
    if temperature is not None:
        if not np.isfinite(temperature) or temperature<=0:raise ValueError('temperature must be positive')
        # derivative w.r.t. mu/k_B, measured in K
        out['dmean_dmu_kelvin']=r['response']/temperature
    return out


class SaltCoordinates:
    """Independent neutral stoichiometric basis N=N0+S n, S=(species,salts).

    theta=S^T lambda; choose lambda=L^T theta, L=(S^T S)^-1 S^T.
    d<N>/d theta=Cov(N,n); d<n>/d theta=Cov(n,n).
    """
    def __init__(self,stoichiometry,valences,reference_counts=None):
        s=np.asarray(stoichiometry,float);q=np.asarray(valences,float)
        if s.ndim==1:s=s[:,None]
        if s.ndim!=2 or q.shape!=(s.shape[0],) or not np.isfinite(s).all() or not np.isfinite(q).all():raise ValueError('invalid basis')
        if np.linalg.matrix_rank(s)!=s.shape[1]:raise ValueError('salt basis must have independent columns')
        if not np.allclose(q@s,0,atol=1e-12):raise ValueError('salt columns must be electroneutral')
        self.S=s;self.L=np.linalg.pinv(s);self.q=q
        self.reference=np.zeros(len(q)) if reference_counts is None else np.asarray(reference_counts,float)
        if self.reference.shape!=q.shape or not np.isfinite(self.reference).all():raise ValueError('invalid reference')
    def extents(self,counts):
        x=_matrix(counts);y=x-self.reference;n=y@self.L.T
        if not np.allclose(n@self.S.T,y,atol=1e-9):raise ValueError('trajectory leaves the specified salt sector')
        return n
    def response(self,counts):
        x=_matrix(counts);n=self.extents(x);r=weighted_response(x)['response']
        return {'salt_mean':n.mean(axis=0),'salt_response':self.L@r@self.L.T,
                'species_to_salt_response':r@self.L.T,'charge_null_residual':float(np.max(np.abs(self.q@r)))}
    def log_activities(self,theta):
        return self.L.T@np.asarray(theta,float)


def chemical_reweight(counts,delta_log_activities,log_weights=None):
    x=_matrix(counts);delta=np.asarray(delta_log_activities,float)
    if delta.shape!=(x.shape[1],) or not np.isfinite(delta).all():raise ValueError('invalid chemical-potential shift')
    lw=x@delta
    if log_weights is not None:lw=lw+np.asarray(log_weights,float)
    result=weighted_response(x,lw)
    result['weight_fraction']=result['kish_ess']/len(x)
    result['warning']='low importance-weight overlap' if result['weight_fraction']<.1 else None
    return result


def electrochemical_response(counts,electrode_charges,voltage_pattern,electrode_response,temperature):
    """Joint Hessian for (log activities, e*DeltaV/kBT), including Gaussian term.

    Q is the conjugate electrode charge in units of e. Chemical--voltage cross
    terms need no conditional correction; Q--Q requires T v^T P v.
    """
    from .electrode import VOLT_TO_K
    n=_matrix(counts);q=_matrix(electrode_charges);v=np.asarray(voltage_pattern,float)
    if len(n)!=len(q) or v.shape!=(q.shape[1],) or temperature<=0:raise ValueError('invalid interfacial response inputs')
    conjugate=q@v;x=np.column_stack([n,conjugate]);r=weighted_response(x)
    hess=r['response'].copy();hess[-1,-1]+=temperature*(v@electrode_response.P@v)
    return {'mean':r['mean'],'generalized_response':hess,
            'dN_dvoltage_per_V':VOLT_TO_K/temperature*hess[:-1,-1],
            'dQ_dlogactivity_e':hess[-1,:-1],
            'dQ_dvoltage_e_per_V':VOLT_TO_K/temperature*hess[-1,-1],
            'eigenvalues':np.linalg.eigvalsh(hess)}


def electrochemical_reweight(counts,electrode_charges,delta_log_activities,delta_voltage,
                             voltage_pattern,electrode_response,temperature,log_weights=None):
    """Exact fixed-A chemical/voltage reweighting within sampled support.

    q means BO conditional means; target q is updated analytically. This is not
    independent evidence of support beyond the sampled states; Kish ESS is given.
    """
    from .electrode import VOLT_TO_K
    n=_matrix(counts);q=_matrix(electrode_charges);v=np.asarray(voltage_pattern,float)
    dl=np.asarray(delta_log_activities,float)
    if len(n)!=len(q) or dl.shape!=(n.shape[1],) or v.shape!=(q.shape[1],) or temperature<=0:raise ValueError('invalid coupled reweighting')
    d=VOLT_TO_K*float(delta_voltage);P=electrode_response.P
    increment=n@dl+d*(q@v)/temperature+.5*d*d*(v@P@v)/temperature
    old=np.zeros(len(n)) if log_weights is None else np.asarray(log_weights,float)
    lw=old+increment;weights=np.exp(lw-logsumexp(lw));qtarget=q+d*(P@v)
    return {'mean_counts':weights@n,'mean_electrode_charges':weights@qtarget,
            'weights':weights,'log_weights':lw-logsumexp(lw),
            'log_partition_ratio':float(logsumexp(lw)-logsumexp(old)),
            'kish_ess':float(1/(weights@weights)),'weight_fraction':float(1/(weights@weights)/len(n))}
