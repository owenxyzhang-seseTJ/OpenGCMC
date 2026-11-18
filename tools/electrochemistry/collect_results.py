"""Reproducible 0.4 validation curves, physical currents and cache microbenchmarks."""
from pathlib import Path
import sys,os,json,time,math,platform,subprocess,xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
from poremc.reactive_cli import build,build_network
from poremc.hydration import MolecularHamiltonian,Molecule
from poremc.reactive import acid_site,redox_site,ReactionEdge,Reservoirs,prepare_reactions,ReactiveGCMC
from poremc.reaction_cache import LocalChargeResponse
from poremc.kinetics import KineticEdge
from poremc.calibration import network_from_reaction_windows
from poremc.electrode import ELEMENTARY_CHARGE,VOLT_TO_K
from poremc.thermo_cli import jsonable
from poremc.transport import FARADAY
OUT=ROOT/'benchmarks/runs/electrochemistry';OUT.mkdir(parents=True, exist_ok=True)
def save(name,x): (OUT/name).write_text(json.dumps(jsonable(x),indent=2),encoding='utf8')
def blocks(x,n=20):
    x=np.asarray(x);m=len(x)//n;b=x[:m*n].reshape(n,m).mean(axis=1)
    return float(x.mean()),float(b.std(ddof=1)/np.sqrt(n))


def statistical_curves():
    rows=[];raw={}
    for kind,values in [('pH',np.linspace(5,9,9)),('potential',np.linspace(-.12,.12,9))]:
        for j,value in enumerate(values):
            T=300.;t=acid_site('acid',pKa=7.) if kind=='pH' else redox_site('redox')
            ts,edges=prepare_reactions([t],[ReactionEdge(0,0,1)])
            h=MolecularHamiltonian(ts,np.eye(3)*24,cutoff=8,coulomb='none')
            h.set_molecules([Molecule(0,0,[[12,12,12]],mobile=False,exchangeable=False)])
            sim=ReactiveGCMC(h,Reservoirs(T,value if kind=='pH' else 7.,value if kind=='potential' else 0.),
                             reactions=edges,move_weights=(.2,0,.8),seed=610+j)
            out=sim.run(6000,burnin=500,sample_every=1)
            y=out['state_counts'][:,1];mean,se=blocks(y)
            exact=1/(1+10**(value-7)) if kind=='pH' else 1/(1+np.exp(VOLT_TO_K*value/T))
            rows.append(dict(kind=kind,x=float(value),mean=mean,block_sem=se,exact=float(exact),absolute_error=abs(mean-exact)))
            raw[f'{kind}_{j}']=y;h.close()
    save('equilibrium_curves.json',rows);np.savez_compressed(OUT/'equilibrium_curves_raw.npz',**raw)
    print('statistical curves complete',flush=True)


def cache_benchmark():
    cfg=json.loads((ROOT/'examples/electrochemistry/hydrated_interface.json').read_text())
    cfg['reactions'][1]['stages']=32
    sim=build(cfg);h=sim.h;uid=63;fullpath=np.asarray(sim.reactions[1].bridge)
    rows=[]
    for count in (2,9,17,33):
        states=fullpath[np.linspace(0,len(fullpath)-1,count).round().astype(int)]
        def direct():
            i=h.molecule_index(uid);m=h.molecules[i]
            values=[]
            for s in states:
                n=m.clone();n.state=int(s);values.append(h.propose([i],[n])['delta_energy'])
            return np.array(values)
        def cached():return LocalChargeResponse(h,uid).deltas(states)
        a,b=direct(),cached();ds=[];cs=[]
        for repeat in range(9):
            order=[('direct',direct,ds),('cached',cached,cs)]
            if repeat%2:order.reverse()
            for _,fn,times in order:
                start=time.perf_counter()
                for _ in range(4):fn()
                times.append((time.perf_counter()-start)/4)
        rows.append({'states':count,'direct_seconds':ds,'cached_seconds':cs,
            'direct_median_ms':1000*np.median(ds),'cached_median_ms':1000*np.median(cs),
            'paired_speedup_median':np.median(np.array(ds)/np.array(cs)),
            'maximum_absolute_delta_error_K':np.max(np.abs(a-b)),
            'cache_construction_included':True,'scope':'same coordinates; identical LJ; one local reactive charge; 60 SPC/E waters; 18 Gaussian electrode sites'})
    sim.h.close();save('charge_response_benchmark.json',rows)
    print('charge cache complete',flush=True)


def kinetic_curves():
    cfg=json.loads((ROOT/'examples/electrochemistry/two_contact_current.json').read_text());network=build_network(cfg)
    rows=[];raw=[]
    for dv in np.linspace(-.12,.12,9):
        v=[-dv/2,dv/2];exact=network.steady_state(v)
        samples=[];probs=[]
        for seed in (41,67,89):
            r=network.ssa([{'duration_s':200.,'potentials_V':v}],seed=seed)
            samples.append(r['faradaic_current_A'].tolist());probs.append(r['time_weighted_probability'].tolist())
            raw.append({'voltage_difference_V':float(dv),'seed':seed,'net_electrons':r['net_electrons_to_species'].tolist(),
                        'duration_s':r['duration_s'],'probability':r['time_weighted_probability'].tolist(),'events':len(r['events'])})
        samples=np.array(samples)
        rows.append({'voltage_difference_V':float(dv),'analytic_current_A':exact['faradaic_current_A'].tolist(),
                     'ssa_current_A':samples.tolist(),'ssa_mean_A':samples.mean(axis=0).tolist(),
                     'ssa_sem_A':(samples.std(axis=0,ddof=1)/np.sqrt(len(samples))).tolist(),
                     'probability':probs})
    save('kinetic_iv.json',rows);save('kinetic_iv_raw.json',raw)
    # Actual hydration-window outputs are mapped to two physical endpoints.
    cal=json.loads((OUT/'hydrated_redox_calibration/calibration.json').read_text())
    c=json.loads((ROOT/'benchmarks/runs/electrochemistry/hydrated_calibrated_pipeline.json').read_text());sim=build(c)
    net,meta=network_from_reaction_windows(sim.h.templates[4],cal,[0,1],
        [KineticEdge(0,1,electrode=i,k0_per_s=20.) for i in (0,1)],temperature=298.15,
        reference_label='same-fixture short seven-window pilot; not a material calibration',
        electrode_groups=[0]*9+[1]*9,reference_potentials_V=[-.05,.05])
    result=net.steady_state([-.05,.05]);sim.h.close()
    save('hydration_to_current_bridge.json',dict(metadata=meta,steady=result,
       rate_note='20 /s supplied as an illustrative independent kinetic input, not inferred from MC'))
    print('kinetics complete',flush=True)


def aggregate():
    summary={}
    tree=ET.parse(OUT/'all_tests.xml');suites=tree.getroot().findall('testsuite')
    summary['tests']={key:sum(int(s.attrib.get(key,0)) for s in suites) for key in ('tests','failures','errors','skipped')}
    rows=json.loads((OUT/'equilibrium_curves.json').read_text())
    summary['equilibrium']={k:{'max_absolute_error':max(r['absolute_error'] for r in rows if r['kind']==k)} for k in ('pH','potential')}
    xyz=np.load(OUT/'hydrated_calibrated_pipeline/coordinates.npz');p=xyz['positions'];recs=xyz['records']
    cell=np.diag([12.8,12.8,48.]);zs=[];cn=[];dists=[];charge_rows=[]
    for frame in np.unique(recs[:,0]):
        group=recs[recs[:,0]==frame];waters=group[group[:,2]==0];oxygen=p[waters[:,4]]
        zs.extend(oxygen[:,2])
        for ion in group[np.isin(group[:,2],[1,2])]:
            d=oxygen-p[ion[4]];d[:,:2]-=np.round(d[:,:2]/12.8)*12.8
            rr=np.linalg.norm(d,axis=1);cn.append([int(ion[2]),int(np.sum(rr<3.5))]);dists.extend([[int(ion[2]),v] for v in rr])
    np.savez_compressed(OUT/'hydration_structure.npz',oxygen_z_A=zs,ion_coordination=np.array(cn),ion_oxygen_distances_A=np.array(dists))
    for name in ('hydrated_interface','hydrated_calibrated_pipeline','dummy_proton_titration','pcet_four_state'):
        summary[name]=json.loads((OUT/name/'summary.json').read_text())
    chron=np.load(OUT/'diffusion_chronoamperometry/transport.npz');t=chron['time_s']
    exact=np.full_like(t,np.nan);positive=t>0;exact[positive]=-FARADAY*1e-6*1.*np.sqrt(1e-9/(np.pi*t[positive]))
    use=(t>=.005)&(t<=.5);err=np.abs(chron['faradaic_current_A'][use]/exact[use]-1)
    summary['cottrell']={'comparison_time_s':[.005,.5],'max_relative_error':float(err.max()),'rms_relative_error':float(np.sqrt(np.mean(err**2)))}
    np.savez_compressed(OUT/'cottrell_comparison.npz',time_s=t,current_A=chron['faradaic_current_A'],cottrell_A=exact,comparison_mask=use)
    cv=np.load(OUT/'diffusion_voltammetry/transport.npz')
    summary['diffusion_cv']={'max_absolute_total_current_A':float(np.max(np.abs(cv['total_current_A']))),
              'minimum_concentration_mol_m3':float(min(cv['oxidized_mol_m3'].min(),cv['reduced_mol_m3'].min()))}
    pcet=np.load(OUT/'pcet_four_state/trajectory.npz');c=json.loads((OUT/'pcet_four_state/config.json').read_text());sim=build(c)
    b=np.array([sim.reservoirs.state_bias(s) for s in sim.h.templates[0].states[:4]]);exact=np.exp(-b+min(b));exact/=exact.sum();sim.h.close()
    mean=pcet['state_counts'][:,:4].mean(axis=0)
    summary['pcet']={'exact':exact,'sampled':mean,'max_error':np.max(np.abs(mean-exact))}
    raw=json.loads((OUT/'kinetic_iv.json').read_text());nz=[r for r in raw if abs(r['voltage_difference_V'])>.01]
    summary['kinetic_iv']={'max_relative_error_three_seed_mean':max(abs(r['ssa_mean_A'][0]/r['analytic_current_A'][0]-1) for r in nz)}
    # Every state change conserves charge in the actual hydrated trajectory.
    tr=np.load(OUT/'hydrated_calibrated_pipeline/trajectory.npz');sc=tr['state_counts'];
    cfg=json.loads((OUT/'hydrated_calibrated_pipeline/config.json').read_text());sim=build(cfg)
    charges=np.array([s.charge for template in sim.h.templates for s in template.states]);sim.h.close()
    balance=sc@charges+tr['electrode_charges'].sum(axis=1)
    summary['hydrated_trajectory_max_charge_error_e']=float(np.max(np.abs(balance)))
    summary['hardware']={'platform':platform.platform(),'python':sys.version,'numpy':np.__version__,
        'cpu':next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),'unknown'),
        'compiler':subprocess.check_output(['g++','--version'],text=True).splitlines()[0],
        'threads':{'OMP':os.environ.get('OMP_NUM_THREADS'),'OPENBLAS':os.environ.get('OPENBLAS_NUM_THREADS')},
        'gpu_execution':False}
    save('summary.json',summary);print(json.dumps(jsonable(summary),indent=2),flush=True)


if __name__=='__main__':
    task=sys.argv[1] if len(sys.argv)>1 else 'all'
    if task in ('all','statistics'):statistical_curves()
    if task in ('all','cache'):cache_benchmark()
    if task in ('all','kinetics'):kinetic_curves()
    if task in ('all','aggregate'):aggregate()
