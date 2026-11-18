"""Buffered supported-electrolyte diffusion + Faradaic boundary currents.

A one-dimensional finite-volume O + mH+ + ne- <-> R model. Solvent/ions are
coarse-grained here; D and k0 are physical inputs, never extracted from MC time.
No electromigration, convection, unbuffered pH evolution or pore tortuosity is
silently inferred. Surface concentrations use an eliminated half-cell resistance.
SI units throughout this module: m, s, mol/m^3, V, A.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags, bmat
from .kinetics import KineticEdge, paired_log_rates
from .electrode import VOLT_TO_K, ELEMENTARY_CHARGE
from .units import NA
FARADAY = ELEMENTARY_CHARGE * NA


@dataclass(frozen=True)
class DiffusionRedox:
    length_m: float = 1e-4
    cells: int = 200
    diffusivity_oxidized_m2_s: float = 1e-9
    diffusivity_reduced_m2_s: float = 1e-9
    bulk_oxidized_mol_m3: float = 1.
    bulk_reduced_mol_m3: float = 0.
    k0_m_s: float = 1e-5
    formal_potential_V: float = 0.
    temperature_K: float = 298.15
    electrons: int = 1
    protons: int = 0
    pH: float = 7.
    reference_pH: float = 7.
    transfer_coefficient: float = .5
    area_m2: float = 1e-6
    double_layer_capacitance_F_m2: float = 0.
    closed_bulk: bool = False

    def __post_init__(self):
        positive = [self.length_m, self.diffusivity_oxidized_m2_s,
                    self.diffusivity_reduced_m2_s, self.k0_m_s,
                    self.temperature_K, self.area_m2]
        if not np.isfinite(positive).all() or min(positive) <= 0 or self.cells < 3 or int(self.cells)!=self.cells:
            raise ValueError('positive finite physical parameters and at least three cells required')
        if min(self.bulk_oxidized_mol_m3, self.bulk_reduced_mol_m3, self.double_layer_capacitance_F_m2) < 0:
            raise ValueError('negative concentration or capacitance')
        if not np.isfinite([self.bulk_oxidized_mol_m3,self.bulk_reduced_mol_m3,
              self.double_layer_capacitance_F_m2,self.formal_potential_V,self.pH,self.reference_pH]).all():
            raise ValueError('nonfinite reservoir input')
        if self.electrons < 1 or self.electrons != int(self.electrons) or self.protons != int(self.protons):
            raise ValueError('integral positive electron and integral proton stoichiometry required')
        if not 0 < self.transfer_coefficient < 1:
            raise ValueError('transfer coefficient must lie in (0,1)')

    @property
    def dx(self):
        return self.length_m / self.cells

    def boundary_coefficients(self, potential_V):
        d = (self.electrons * VOLT_TO_K * (potential_V-self.formal_potential_V)/self.temperature_K
             + self.protons*math.log(10)*(self.pH-self.reference_pH))
        a = math.log(self.k0_m_s)-self.transfer_coefficient*d
        b = math.log(self.k0_m_s)+(1-self.transfer_coefficient)*d
        # Numerically eliminate J=(kf*cO0-kr*cR0)/(1+kf*h/DO+kr*h/DR).
        scale = max(0., a, b)
        fa, fb = math.exp(a-scale), math.exp(b-scale)
        den = math.exp(-scale) + fa*self.dx/(2*self.diffusivity_oxidized_m2_s) + fb*self.dx/(2*self.diffusivity_reduced_m2_s)
        return fa/den, fb/den

    def _diffusion_matrix(self, D):
        n, dx = self.cells, self.dx
        diagonal = np.full(n, -2.)
        diagonal[0] = -1.
        diagonal[-1] = -1. if self.closed_bulk else -3.
        return diags([np.ones(n-1),diagonal,np.ones(n-1)],[-1,0,1],format='csc') * (D/dx**2)

    def operator(self, potential_V):
        n, dx = self.cells, self.dx
        f, r = self.boundary_coefficients(potential_V)
        A = self._diffusion_matrix(self.diffusivity_oxidized_m2_s).tolil()
        B = self._diffusion_matrix(self.diffusivity_reduced_m2_s).tolil()
        cross_OR = diags([np.r_[r/dx,np.zeros(n-1)]],[0],format='csc')
        cross_RO = diags([np.r_[f/dx,np.zeros(n-1)]],[0],format='csc')
        A[0,0] -= f/dx
        B[0,0] -= r/dx
        matrix = bmat([[A,cross_OR],[cross_RO,B]],format='csc')
        source = np.zeros(2*n)
        if not self.closed_bulk:
            source[n-1] = 2*self.diffusivity_oxidized_m2_s*self.bulk_oxidized_mol_m3/dx**2
            source[-1] = 2*self.diffusivity_reduced_m2_s*self.bulk_reduced_mol_m3/dx**2
        return matrix, source

    def steady_flux(self, potential_V):
        if self.closed_bulk:
            raise ValueError('closed cell has no sustained reservoir flux')
        d=(self.electrons*VOLT_TO_K*(potential_V-self.formal_potential_V)/self.temperature_K
            +self.protons*math.log(10)*(self.pH-self.reference_pH))
        a=math.log(self.k0_m_s)-self.transfer_coefficient*d
        b=math.log(self.k0_m_s)+(1-self.transfer_coefficient)*d
        scale=max(0.,a,b)
        f,r=math.exp(a-scale),math.exp(b-scale)
        den=math.exp(-scale)+f*self.length_m/self.diffusivity_oxidized_m2_s+r*self.length_m/self.diffusivity_reduced_m2_s
        return (f*self.bulk_oxidized_mol_m3-r*self.bulk_reduced_mol_m3)/den

    def run(self, times_s, potentials_V, *, output_times_s=None, samples=401,
            initial_oxidized=None, initial_reduced=None, rtol=2e-7, atol=1e-10):
        """Continuous piecewise-linear E(t). Constant E supports chronoamperometry.

        A voltage STEP is supplied as a new run with a changed initial potential;
        its ideal capacitive impulse C*ΔE is separate from the finite t>0 trace.
        """
        times, potentials = np.asarray(times_s,float),np.asarray(potentials_V,float)
        if times.ndim!=1 or len(times)<2 or potentials.shape!=times.shape or np.any(np.diff(times)<=0) or not np.isfinite(np.r_[times,potentials]).all():
            raise ValueError('strictly increasing waveform time points required')
        tout=np.linspace(times[0],times[-1],samples) if output_times_s is None else np.asarray(output_times_s,float)
        if tout.ndim!=1 or len(tout)<2 or np.any(np.diff(tout)<=0) or tout[0]<times[0] or tout[-1]>times[-1]:
            raise ValueError('output time points must increase within protocol')
        n=self.cells
        o=np.full(n,self.bulk_oxidized_mol_m3) if initial_oxidized is None else np.broadcast_to(np.asarray(initial_oxidized,float),(n,)).copy()
        r=np.full(n,self.bulk_reduced_mol_m3) if initial_reduced is None else np.broadcast_to(np.asarray(initial_reduced,float),(n,)).copy()
        y=np.r_[o,r]
        if y.min()<0 or not np.isfinite(y).all():raise ValueError('invalid initial concentration')
        profiles=np.empty((len(tout),2*n));evaluations=0
        for j in range(len(times)-1):
            a,b=times[j:j+2];slope=(potentials[j+1]-potentials[j])/(b-a)
            cache={}
            def operator(t):
                E=potentials[j]+slope*(t-a)
                if not cache or cache['E']!=E:
                    cache['E']=E;cache['op']=self.operator(E)
                return cache['op']
            def rhs(t,c):
                M,s=operator(t);return M@c+s
            def jac(t,c):return operator(t)[0]
            sol=solve_ivp(rhs,(a,b),y,method='BDF',jac=jac,dense_output=True,
                          rtol=rtol,atol=atol,max_step=(b-a)/20)
            if not sol.success:raise RuntimeError(sol.message)
            ids=np.flatnonzero((tout>=a)&(tout<=b));profiles[ids]=sol.sol(tout[ids]).T
            y=sol.y[:,-1];evaluations+=sol.nfev
        E=np.interp(tout,times,potentials);flux=[];slopes=[]
        for t,e,c in zip(tout,E,profiles):
            f,r=self.boundary_coefficients(e);flux.append(f*c[0]-r*c[n])
            j=min(np.searchsorted(times,t,side='right')-1,len(times)-2)
            slopes.append((potentials[j+1]-potentials[j])/(times[j+1]-times[j]))
        flux=np.asarray(flux)
        IF=-self.electrons*FARADAY*self.area_m2*flux
        IC=self.double_layer_capacitance_F_m2*self.area_m2*np.asarray(slopes)
        # Surface values reconstructed from the same half-cell balance.
        Os=profiles[:,0]-flux*self.dx/(2*self.diffusivity_oxidized_m2_s)
        Rs=profiles[:,n]+flux*self.dx/(2*self.diffusivity_reduced_m2_s)
        return {'time_s':tout,'potential_V':E,'x_m':(np.arange(n)+.5)*self.dx,
                'oxidized_mol_m3':profiles[:,:n],'reduced_mol_m3':profiles[:,n:],
                'surface_oxidized_mol_m3':Os,'surface_reduced_mol_m3':Rs,
                'molar_flux_mol_m2_s':flux,'faradaic_current_A':IF,
                'capacitive_current_A':IC,'total_current_A':IF+IC,
                'total_inventory_mol':profiles.sum(axis=1)*self.dx*self.area_m2,
                'minimum_concentration_mol_m3':float(min(profiles.min(),Os.min(),Rs.min())),
                'rhs_evaluations':evaluations,
                'model':'1D diffusion; buffered pH; supporting electrolyte; no migration/convection'}
