"""Fixed-geometry Gaussian electrodes at constant potential.

Hamiltonian in kelvin: U0 + 1/2 q.A.q + b.q - (e/kB)*psi.q.
A is cached once; q satisfies sum(q)=-Q_fixed-Q_ions. Ions may be charged
provided the electrodes compensate. 3D periodic Ewald, optional slab dipole
correction; it is NOT an exact 2D Green function. Gaussian widths are standard
deviations [A]. Constant dielectric is a primitive-model parameter.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.special import erfc
from .electrostatics import Ewald
from .cell import cutoff_image
from .cache import ArrayCache, content_key
from .units import COULOMB_KA,KB
ELEMENTARY_CHARGE=1.602176634e-19
VOLT_TO_K=ELEMENTARY_CHARGE/KB


def solve_projected_charge(apply,field,potentials,total_charge,*,eigenvalue_lower_bound,
                           tolerance=1e-10,max_iterations=None,precondition=None):
    """Solve fixed-potential charges using only products with the electrode matrix.

    The caller supplies a positive spectral lower bound of A on sum(q)=0.
    Given that bound, a feasible iterate lies at most ||r_perp||²/(2*m)
    above the minimum. Roundoff in A and the dot products is outside the bound.
    """
    from inspect import signature
    from scipy.sparse.linalg import LinearOperator, cg

    b=np.asarray(field,float)
    v=np.asarray(potentials,float)
    q_total=float(total_charge)
    lower=float(eigenvalue_lower_bound)
    if (b.ndim!=1 or b.size<2 or v.shape!=b.shape or not np.isfinite(b).all()
            or not np.isfinite(v).all() or not np.isfinite([q_total,lower,tolerance]).all()
            or lower<=0 or tolerance<=0):
        raise ValueError('finite electrode fields and a positive eigenvalue lower bound are required')

    def project(x):
        return x-np.mean(x)

    def product(x):
        result=np.asarray(apply(x),dtype=float)
        if result.shape!=b.shape or not np.isfinite(result).all():
            raise ValueError('invalid electrode matrix-vector product')
        return result

    def projected_product(x):
        return project(product(project(x)))+np.mean(x)

    fixed=np.full(b.shape,q_total/b.size)
    linear=b-VOLT_TO_K*v
    rhs=-project(product(fixed)+linear)
    op=LinearOperator((b.size,b.size),matvec=projected_product,dtype=float)
    pc=None
    if precondition is not None:
        pc=LinearOperator(op.shape,matvec=lambda x:project(precondition(project(x)))+np.mean(x),dtype=float)
    relative={'rtol' if 'rtol' in signature(cg).parameters else 'tol':tolerance}
    free,info=cg(op,rhs,atol=0.,maxiter=max_iterations,M=pc,**relative)
    if info<0 or not np.isfinite(free).all():
        raise FloatingPointError('projected electrode solver failed')
    q=fixed+project(free)
    aq=product(q)
    gradient=aq+linear
    residual=project(gradient)
    energy=float(0.5*q@aq+linear@q)
    bound=float(residual@residual/(2*lower))
    return {'q':q,'free_energy':energy,'free_energy_lower_bound':energy-bound,
            'energy_error_bound':bound,'potential_residual_V':float(np.max(np.abs(residual))/VOLT_TO_K),
            'lagrange':float(-np.mean(gradient)),'converged':info==0}


class ElectrodeResponse:
    """Analytical Gaussian integration / constrained minimization for fixed A.

    O(M^3) factorization once; O(M^2) delta response. The integrated determinant
    is a constant for fixed geometry, so minima give exact marginal ionic weights.
    Conditional covariance T*P must be restored for electrode charge statistics.
    """
    def __init__(self,matrix,cache_dir=None):
        a=np.asarray(matrix,float)
        if a.ndim!=2 or a.shape[0]!=a.shape[1] or len(a)<2 or not np.isfinite(a).all() or not np.allclose(a,a.T,rtol=1e-12,atol=1e-9):raise ValueError('A must be finite symmetric square')
        self.A=(a+a.T)*.5;self.m=len(a);self.ones=np.ones(self.m)
        key=content_key({'kind':'electrode-response-v1'}, {'A':self.A})
        cache=ArrayCache(cache_dir) if cache_dir else None;data=cache.get(key) if cache else None
        self.cache_hit=data is not None;self.cache_key=key
        if data is None:
            cf=cho_factor(self.A,check_finite=True);S=cho_solve(cf,np.eye(self.m))
            h=S@self.ones;den=float(h.sum())
            if den<=0:raise ValueError('non-positive constrained capacitance')
            P=S-np.outer(h,h)/den;P=(P+P.T)*.5
            data={'P':P,'t':h/den,'den':np.array(den)}
            if cache:cache.put(key,data)
        self.P=np.array(data['P']);self.t=np.array(data['t']);self.den=float(data['den'])
        if np.max(np.abs(self.ones@self.P))>1e-8:raise ValueError('electrode projection failed')
        for x in (self.A,self.P,self.t):x.flags.writeable=False
    def solve(self,field,potentials,total_charge=0.):
        b=np.asarray(field,float);v=np.asarray(potentials,float)
        if b.shape!=(self.m,) or v.shape!=(self.m,) or not np.isfinite(b).all() or not np.isfinite(v).all() or not np.isfinite(total_charge):raise ValueError('invalid electrode state')
        c=VOLT_TO_K*v-b;Q=float(total_charge)
        q=self.P@c+self.t*Q;lagrange=float(self.t@c-Q/self.den)
        free=-.5*c@self.P@c-Q*(self.t@c)+.5*Q*Q/self.den
        residual=self.A@q+b-VOLT_TO_K*v+lagrange*self.ones
        return {'q':q,'field':b.copy(),'Q':Q,'lagrange':lagrange,'free_energy':float(free),
                'potential_residual_V':float(np.max(np.abs(residual))/VOLT_TO_K)}
    def delta(self,state,delta_field,delta_total_charge=0.):
        db=np.asarray(delta_field,float);dq=float(delta_total_charge)
        if db.shape!=(self.m,) or not np.isfinite(db).all() or not np.isfinite(dq):raise ValueError('invalid delta')
        shift=self.P@db
        df=(state['q']@db-.5*db@shift-state['lagrange']*dq+dq*(self.t@db)+.5*dq*dq/self.den)
        new={'q':state['q']-shift+self.t*dq,'field':state['field']+db,'Q':state['Q']+dq,
             'lagrange':state['lagrange']-self.t@db-dq/self.den,
             'free_energy':state['free_energy']+float(df)}
        return float(df),new
    def batch_deltas(self,state,delta_fields,delta_charges=None):
        db=np.asarray(delta_fields,float)
        if db.ndim!=2 or db.shape[1]!=self.m:raise ValueError('batch must be (B,M)')
        dq=np.zeros(len(db)) if delta_charges is None else np.asarray(delta_charges,float)
        if dq.shape!=(len(db),) or not np.isfinite(db).all() or not np.isfinite(dq).all():raise ValueError('invalid batch')
        return db@state['q']-.5*np.sum((db@self.P)*db,axis=1)-state['lagrange']*dq+dq*(db@self.t)+.5*dq*dq/self.den
    def capacitance(self,charge_trace,voltage_pattern,temperature,area_A2=None):
        q=np.asarray(charge_trace,float);v=np.asarray(voltage_pattern,float)
        if q.ndim!=2 or q.shape[1]!=self.m or v.shape!=(self.m,) or len(q)==0 or temperature<=0:raise ValueError('invalid capacitance data')
        total=q@v;bo=float(np.var(total));conditional=float(temperature*(v@self.P@v))
        f=ELEMENTARY_CHARGE**2/KB*(bo+conditional)/temperature
        out={'mean_conjugate_charge_e':float(total.mean()),'BO_variance_e2':bo,
             'conditional_variance_e2':conditional,'capacitance_F':f,
             'dcharge_dvoltage_e_per_V':f/ELEMENTARY_CHARGE}
        if area_A2 is not None:
            if area_A2<=0:raise ValueError('area must be positive')
            out['capacitance_uF_cm2']=f/(area_A2*1e-20)*100
        return out
    def charge_samples(self,state,temperature,size=1,seed=5):
        if temperature<=0 or size<1:raise ValueError('invalid T or size')
        vals,vecs=np.linalg.eigh(self.P);keep=vals>max(vals.max()*1e-12,1e-18)
        rng=np.random.default_rng(seed)
        return state['q']+rng.normal(size=(size,keep.sum()))@(vecs[:,keep]*np.sqrt(temperature*vals[keep])).T
    def to_device(self):
        """Optional persistent CuPy response; imported only on explicit use."""
        import cupy as cp
        return DeviceElectrodeResponse(self,cp)


class DeviceElectrodeResponse:
    def __init__(self,response,cp):
        self.cp=cp;self.P=cp.asarray(response.P);self.t=cp.asarray(response.t);self.den=response.den
    def batch_deltas(self,state,delta_fields,delta_charges=None):
        cp=self.cp;db=cp.asarray(delta_fields,dtype=cp.float64)
        if db.ndim!=2 or db.shape[1]!=len(self.t):raise ValueError('invalid device batch')
        dq=cp.zeros(len(db)) if delta_charges is None else cp.asarray(delta_charges,dtype=cp.float64)
        return db@cp.asarray(state['q'])-.5*cp.sum((db@self.P)*db,axis=1)-state['lagrange']*dq+dq*(db@self.t)+.5*dq*dq/self.den


class GaussianElectrodes:
    """Gaussian-convolved periodic Ewald kernel plus optional Yeh--Berkowitz term.

    Real part [erfc(alpha*r)-erfc(gamma*r)]/r, gamma=1/(sqrt(2)*width).
    The zero-mode constant is -pi*(alpha^-2-gamma^-2)/V. A uses width*sqrt(2).
    No hidden regularizer: hardness (K/e^2) is an explicit model term.
    """
    def __init__(self,positions,potentials,cell,cutoff,alpha=.3,kmax=6,width=1.,dielectric=1.,
                 slab=False,hardness=0.,cache_dir=None):
        self.positions=np.asarray(positions,float).reshape(-1,3).copy();self.potentials=np.asarray(potentials,float).copy()
        if len(self.positions)<2 or self.potentials.shape!=(len(self.positions),) or not np.isfinite(self.positions).all() or not np.isfinite(self.potentials).all():raise ValueError('invalid electrodes')
        if width<=0 or dielectric<=0 or hardness<0 or not np.isfinite([width,dielectric,hardness]).all():raise ValueError('invalid electrode parameters')
        self.ewald=Ewald(cell,cutoff,alpha,kmax,background=True);self.cell=self.ewald.cell
        self.width=float(width);self.dielectric=float(dielectric);self.slab=bool(slab);self.hardness=float(hardness)
        self.K=COULOMB_KA/dielectric;self.volume=self.ewald.volume
        if slab and (not np.allclose(self.cell,np.diag(np.diag(self.cell))) or self.cell[2,2]<=0):raise ValueError('slab correction currently requires orthorhombic z-normal geometry')
        self.phase=np.exp(1j*(self.positions@self.ewald.k.T))
        A=self._green(self.positions,width*np.sqrt(2))
        # Rows are probes, columns are electrode sites; symmetry is enforced only
        # against floating-point reduction noise, not a missing physical term.
        A=(A+A.T)*.5+np.eye(len(A))*hardness
        self.response=ElectrodeResponse(A,cache_dir=cache_dir)
    def _green(self,probe,width):
        probe=np.asarray(probe,float).reshape(-1,3);e=self.ewald;gamma=1/(np.sqrt(2)*width)
        d=cutoff_image(probe[:,None,:]-self.positions[None,:,:],self.cell);r=np.linalg.norm(d,axis=2)
        safe=np.where(r>1e-10,r,1.);real=(erfc(e.alpha*safe)-erfc(gamma*safe))/safe
        real=np.where(r<=1e-10,2*(gamma-e.alpha)/np.sqrt(np.pi),real)
        real=np.where(r<e.cutoff,real,0.)
        phase=np.exp(1j*(probe@e.k.T))
        rec=np.real((phase*(2*e.coefficient)[None,:])@self.phase.conj().T)
        g=real+rec-np.pi*(1/e.alpha**2-1/gamma**2)/e.volume
        if self.slab:g+=4*np.pi/e.volume*probe[:,2,None]*self.positions[None,:,2]
        return self.K*g
    def fields(self,positions,charges):
        p=np.asarray(positions,float).reshape(-1,3);q=np.asarray(charges,float)
        if q.shape!=(len(p),) or not np.isfinite(p).all() or not np.isfinite(q).all():raise ValueError('invalid field inputs')
        if not len(p):return np.empty((0,len(self.positions)))
        return self._green(p,self.width)*q[:,None]
    def solve(self,positions,charges):
        q=np.asarray(charges,float)
        return self.response.solve(self.fields(positions,q).sum(axis=0),self.potentials,-float(q.sum()))
