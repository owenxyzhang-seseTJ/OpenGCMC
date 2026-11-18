#!/usr/bin/env python3
"""Run the OpenGCMC thermodynamics benchmarks. BLAS should use one thread."""
from pathlib import Path
import argparse,json,time,platform,sys,os
import numpy as np
from scipy.special import logsumexp
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from poremc.response import *
from poremc.free_energy import *
from poremc.electrolyte import *
from poremc.electrode import *
from poremc.thermo_cli import jsonable,build
from poremc.statistics import summarize
OUT=ROOT/'benchmarks/runs/thermodynamics';OUT.mkdir(parents=True, exist_ok=True)


def write(name,data):
    (OUT/(name+'.json')).write_text(json.dumps(jsonable(data),indent=2),encoding='utf8')
    print(name, json.dumps(jsonable(data),ensure_ascii=False)[:1000],flush=True)


def response_demo():
    rng=np.random.default_rng(810);p=np.array([.3,.4,.3]);sites=12
    x=rng.multinomial(sites,p,size=160000)[:,1:]
    r=response_summary(x,300,block_length=100,bootstrap=100)
    exact=sites*(np.diag(p[1:])-np.outer(p[1:],p[1:]))
    np.savez_compressed(OUT/'response_competition.npz',counts=x,response=r['response'],exact=exact,ci95=r['response_ci95'])
    write('response_competition',{'model':'12 independent sites, each empty/A/B','probabilities':p,'response':r['response'],'exact':exact,'max_absolute_error':np.max(abs(r['response']-exact)), 'ci95':r['response_ci95']})


def salts():
    rows=[]
    for nu,q,zv in [((1,1),(1,-1),(4.,4.)),((1,2),(2,-1),(2.,4.))]:
        for seed in [31,47]:
            species=[Ion('cation',q[0],0,0),Ion('anion',q[1],0,0)];cell=np.eye(3)*24
            h=IonicHamiltonian(species,cell,8,coulomb='none');sim=ElectrolyteGCMC(h,300,np.log(np.array(zv)/np.linalg.det(cell)),[ExchangeChannel('salt',nu,insert_probability=.7)],seed=seed,exchange_probability=1.)
            t=sim.run(60000,burnin=2000,sample_every=2);ref=ideal_salt_distribution(nu,np.log(zv));name=f'salt_{nu[1]}_{seed}'
            r=response_summary(t['counts'],block_length=100,bootstrap=40)
            np.savez_compressed(OUT/(name+'.npz'),counts=t['counts'],ref_counts=ref['counts'],ref_probability=ref['probability'])
            rows.append({'case':name,'stoichiometry':nu,'seed':seed,'mean':r['mean'],'exact_mean':ref['mean'],'covariance':r['response'],'exact_covariance':ref['covariance'],'statistics':[summarize(t['counts'][:,j],t['wall_seconds']) for j in range(2)],'wall_seconds':t['wall_seconds'],'moves_per_second':t['moves_per_second'],'max_charge_error':float(np.max(abs(t['counts']@q)))})
            h.close()
    write('ideal_salts',rows)


def umbrella():
    rng=np.random.default_rng(924);ns=np.arange(41);F=8*(((ns-20)/12)**2-1)**2
    centers=np.arange(4,41,6);stiffness=.12;bias=.5*stiffness*(ns[None,:]-centers[:,None])**2
    saved=[];nks=[];t0=time.perf_counter()
    for i,center in enumerate(centers):
        state=int(center);trace=[];u=F+bias[i]
        for step in range(85000):
            candidate=state+(1 if rng.random()<.5 else -1)
            if 0<=candidate<len(ns) and np.log(max(rng.random(),1e-300))<min(0,u[state]-u[candidate]):state=candidate
            if step>=5000 and step%5==0:trace.append(state)
        saved.extend(trace);nks.append(len(trace))
    x=np.asarray(saved);u=bias[:,x];fit=MBAR(u,nks);lw=fit.log_weights(np.zeros(len(x)))
    r=landscape(x,[np.arange(-.5,41.5)],log_weights=lw)
    frec=r['free_energy_kelvin'];ref=F-F.min();mask=(ref<12)&np.isfinite(frec)
    # Align zero on populated minima, no fit in the barrier region.
    err=frec[mask]-ref[mask];err-=np.mean(err)
    bootstrap=block_bootstrap_mbar(u,nks,np.zeros(len(x)),x,np.full(len(nks),40),replicates=20,seed=40)
    np.savez_compressed(OUT/'umbrella_double_well.npz',cv=x,u_kn=u,n_k=nks,u_target=np.zeros(len(x)),edges=np.arange(-.5,41.5),centers=centers,exact_free=ref,reconstructed_free=frec,overlap=fit.overlap,probability=r['probability_mass'])
    write('umbrella_double_well',{'model':'discrete occupancy free energy, 8 kBT barrier; seven frozen windows','steps_per_window':85000,'saved_per_window':nks,'spectral_gap':fit.spectral_gap,'iterations':fit.iterations,'rms_F_error_kBT_F_under_12':np.sqrt(np.mean(err**2)),'max_F_error_kBT_F_under_12':np.max(abs(err)), 'mean':fit.expectation(x,np.zeros(len(x))),'bootstrap_mean_ci95':bootstrap['ci95'],'wall_seconds':time.perf_counter()-t0})


def molecular_windows():
    from poremc.engine import GCMC,ZeroPotential
    from poremc.topology import Molecule
    m=Molecule('ideal',[0],[0]);biases=[HarmonicCountBias((1,),c,.35) for c in [1.,4.,7.,10.]]
    traces=[]
    for j,b in enumerate(biases):
        sim=GCMC([m],ZeroPotential(),np.eye(3)*24,300,[3/24**3],trials=1,move_weights={'insert':.5,'delete':.5},count_bias=b,seed=28+j)
        traces.append(np.asarray(sim.run(12000,burnin=1000,sample_every=3,record_energy=False)['counts'])[:,0])
    x=np.concatenate(traces);u=np.array([b.values(x[:,None]) for b in biases]);fit=MBAR(u,list(map(len,traces)))
    r=fit.expectation(x,np.zeros(len(x)))
    write('molecular_windows',{'model':'original molecular CBMC engine + frozen count bias, ideal gas mean 3','reweighted_mean':r,'spectral_gap':fit.spectral_gap})
    np.savez_compressed(OUT/'molecular_windows.npz',cv=x,u_kn=u,n_k=list(map(len,traces)),u_target=np.zeros(len(x)),edges=np.arange(-.5,float(x.max())+1.5))


def interface(voltage):
    rows=[]
    for seed in [61,83]:
        cfg=json.loads((ROOT/'examples/thermodynamics/interface_constant_potential.json').read_text());cfg['seed']=seed
        cfg['electrodes']['voltage_difference']=voltage;cfg['forcefield']['kmax']=[5,5,10]
        sim=build(cfg);t=sim.run(6000,burnin=1500,sample_every=5,record_positions=True)
        e=sim.h.electrodes;v=np.repeat([-.5,.5],9)
        cap=e.response.capacitance(t['electrode_charges'],v,300,area_A2=24**2)
        joint=electrochemical_response(t['counts'],t['electrode_charges'],v,e.response,300)
        bs=response_summary(np.column_stack([t['counts'],t['electrode_charges']@v]),block_length=100,bootstrap=50)
        offsets=np.r_[0,np.cumsum([len(x) for x in t['positions']])];positions=np.concatenate(t['positions']);types=np.concatenate(t['types'])
        code=f'interface_{voltage:+.2f}_{seed}';np.savez_compressed(OUT/(code+'.npz'),counts=t['counts'],charges=t['electrode_charges'],energy=t['energy_kelvin'],positions=positions,types=types,offsets=offsets)
        if voltage==0 and seed==61:
            sd=np.sqrt(300*(v@e.response.P@v));Q=t['electrode_charges']@v;qe=np.linspace(Q.min()-5*sd,Q.max()+5*sd,81)
            # N_total is integer even when individual ion exchanges are enabled.
            N=t['counts'].sum(axis=1);ne=np.arange(-.5,float(N.max())+1.5)
            land=interfacial_landscape(N,t['electrode_charges'],v,e.response,300,ne,qe)
            np.savez_compressed(OUT/'interface_landscape.npz',**land)
            fields=[]
            for d in np.linspace(-.1,.1,21):
                r=electrochemical_reweight(t['counts'],t['electrode_charges'],[0,0],d,v,e.response,300)
                fields.append([d,*r['mean_counts'],r['mean_electrode_charges']@v,r['kish_ess'],r['weight_fraction']])
            np.savez_compressed(OUT/'interface_reweight.npz',data=fields)
        full=sim.h.full_energy();direct=e.solve(np.concatenate([sim.h.host_positions,sim.h.positions]),np.r_[sim.h.host_charge,sim.h.valences[sim.h.types]])
        row={'voltage_V':voltage,'seed':seed,'mean_counts':t['counts'].mean(axis=0),'charge_summary':summarize(t['electrode_charges']@v,t['wall_seconds']), 'capacitance':cap,'joint_response':joint,'covariance_ci95':bs['response_ci95'], 'production_attempts':6000,'burnin_attempts':1500,'wall_seconds':t['wall_seconds'],'moves_per_second':t['moves_per_second'],'full_energy_error_K':sim.h.energy-full,'potential_residual_V':direct['potential_residual_V'], 'max_global_charge_error_e':float(np.max(abs(t['charges'].sum(axis=1)+t['counts']@sim.h.valences))) if 'charges' in t else float(np.max(abs(t['electrode_charges'].sum(axis=1)+t['counts']@sim.h.valences))), 'profile':t['profile'],'counters':t['counters'],'fingerprint':sim.h.fingerprint()}
        rows.append(row);sim.h.close()
    write(f'interface_{voltage:+.2f}',rows)


def convergence():
    cell=np.diag([24.,24.,48.]);xyz=np.array([[x,y,z] for z in [12.,36.] for x in [4,12,20] for y in [4,12,20]])
    pos=np.array([[5,6,22],[11,12,26]]);charge=np.array([1.,-1.]);v=np.repeat([-.5,.5],9);rows=[]
    for k in [3,4,5,6,8,10]:
        e=GaussianElectrodes(xyz,.1*v,cell,8,kmax=[k,k,2*k],dielectric=78.4,slab=True)
        s=e.solve(pos,charge);cap=e.response.capacitance(s['q'][None,:],v,300,area_A2=576)
        rows.append({'kmax':[k,k,2*k],'electrode_free_energy_K':s['free_energy'],'capacitance_empty_uF_cm2':cap['capacitance_uF_cm2'],'residual_V':s['potential_residual_V']})
    write('electrostatic_convergence',rows)


def performance():
    rng=np.random.default_rng(333);rows=[]
    for side in [4,6,8]:
        cell=np.diag([48.,48.,96.]);xyz=np.array([[48*(i+.5)/side,48*(j+.5)/side,z] for z in [24,72] for i in range(side) for j in range(side)])
        m=len(xyz);v=np.r_[-np.ones(m//2)/2,np.ones(m//2)/2]
        start=time.perf_counter();e=GaussianElectrodes(xyz,.1*v,cell,12,alpha=.25,kmax=[4,4,8],dielectric=78.4,slab=True);setup=time.perf_counter()-start
        r=e.response;b=rng.normal(0,50,m);state=r.solve(b,.1*v,.3);db=rng.normal(0,2,(128,m));dQ=rng.normal(0,.2,128)
        K=np.block([[r.A,np.ones((m,1))],[np.ones((1,m)),np.zeros((1,1))]])
        def fresh():
            out=[]
            for d,q0 in zip(db,dQ):
                c=VOLT_TO_K*.1*v-b-d;q=np.linalg.solve(K,np.r_[c,.3+q0])[:-1]
                out.append(.5*q@r.A@q-c@q-state['free_energy'])
            return np.array(out)
        def cached():return r.batch_deltas(state,db,dQ)
        a=fresh();z=cached();err=float(np.max(abs(a-z)));ts=[]
        for repeat in range(5):
            order=[('fresh',fresh),('cached',cached)] if repeat%2==0 else [('cached',cached),('fresh',fresh)]
            row={}
            for name,fun in order:
                start=time.perf_counter()
                for _ in range(3):fun()
                row[name]=(time.perf_counter()-start)/3
            ts.append(row)
        rows.append({'electrode_sites':m,'batch':128,'setup_seconds':setup,'seconds':ts,'median_paired_speedup':np.median([x['fresh']/x['cached'] for x in ts]),'max_energy_error_K':err})
    write('electrode_performance',rows)
    # Incremental energy vs independent full-energy path, same Hamiltonian.
    rows=[]
    for units in [4,12,24]:
        cell=np.eye(3)*40;ions=[Ion('C',1,2,20),Ion('A',-1,2,20)];h=IonicHamiltonian(ions,cell,9,kmax=4)
        grid=np.array([[4+8*i,4+8*j,4+8*k] for i in range(4) for j in range(4) for k in range(4)])
        t=np.tile([0,1],units);p=grid[:len(t)].astype(float);h.set_state(t,p)
        positions=[p[0]+rng.uniform(-.5,.5,3) for _ in range(30)];deltas=[];full=[]
        start=time.perf_counter()
        for x in positions:deltas.append(h.propose([0],[0],x[None,:])['delta_energy'])
        td=time.perf_counter()-start;start=time.perf_counter()
        for x in positions:
            p2=p.copy();p2[0]=x;full.append(h.full_energy(t,p2)-h.energy)
        tf=time.perf_counter()-start
        rows.append({'ions':len(t),'trials':30,'incremental_seconds':td,'full_recompute_seconds':tf,'speedup':tf/td,'max_error_K':float(np.max(abs(np.array(deltas)-full))),'profile':h.profile});h.close()
    write('incremental_performance',rows)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('section',choices=['response','salts','umbrella','molecular','interface','convergence','performance']);p.add_argument('--voltage',type=float,default=0.);a=p.parse_args()
    if a.section=='interface':interface(a.voltage)
    else:{'response':response_demo,'salts':salts,'umbrella':umbrella,'molecular':molecular_windows,'convergence':convergence,'performance':performance}[a.section]()
