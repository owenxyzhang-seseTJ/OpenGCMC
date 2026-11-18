"""JSON runner for primitive electrolyte and constant-potential interface GCMC."""
from pathlib import Path
import argparse,json
import numpy as np
from .electrolyte import Ion,ExchangeChannel,FractionalRegion,IonicHamiltonian,ElectrolyteGCMC
from .electrode import GaussianElectrodes
from .free_energy import HarmonicCountBias,MBAR,landscape
from .response import response_summary,electrochemical_response


def jsonable(obj):
    if isinstance(obj,np.ndarray):return jsonable(obj.tolist())
    if isinstance(obj,np.generic):return jsonable(obj.item())
    if isinstance(obj,float) and not np.isfinite(obj):return None
    if isinstance(obj,dict):return {str(k):jsonable(v) for k,v in obj.items()}
    if isinstance(obj,(tuple,list)):return [jsonable(v) for v in obj]
    return obj


def build(config):
    c=dict(config);cell=np.asarray(c['cell'],float)
    if cell.shape==(3,):cell=np.diag(cell)
    species=[Ion(**d) for d in c['species']];params=dict(c.get('forcefield',{}));params.setdefault('cutoff',8.);params.setdefault('alpha',.3);params.setdefault('kmax',5);params.setdefault('dielectric',78.4)
    electrodes=None
    if c.get('electrodes'):
        e=dict(c['electrodes'])
        if 'plane_grid' in e:
            g=e.pop('plane_grid');nx,ny=g['shape'];z=g['z']
            e['positions']=[[cell[0,0]*(i+.5)/nx,cell[1,1]*(j+.5)/ny,z0] for z0 in z for i in range(nx) for j in range(ny)]
            voltage=e.pop('voltage_difference');e['potentials']=[-voltage/2]*(nx*ny)+[voltage/2]*(nx*ny)
        electrodes=GaussianElectrodes(cell=cell,**{k:params[k] for k in ('cutoff','alpha','kmax','dielectric')},slab=params.get('slab',False),**e)
    h=IonicHamiltonian(species,cell,electrodes=electrodes,**params)
    if c.get('initial'):h.set_state(c['initial']['types'],c['initial']['positions'])
    region=FractionalRegion(cell,**c.get('region',{}))
    bias=HarmonicCountBias(**c['bias']) if c.get('bias') else None
    opts=dict(c.get('moves',{}));sim=ElectrolyteGCMC(h,c['temperature'],c['log_activities'],[ExchangeChannel(**d) for d in c['channels']],seed=c.get('seed',123),region=region,bias=bias,**opts)
    return sim


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    run=sub.add_parser('run');run.add_argument('config');run.add_argument('-o','--output',default='run_thermo');run.add_argument('--restart')
    response=sub.add_parser('response');response.add_argument('trajectory');response.add_argument('-o','--output',default='response.json');response.add_argument('--blocks',type=int,default=100)
    mbar=sub.add_parser('mbar');mbar.add_argument('windows',help='NPZ with u_kn,n_k,u_target,cv,edges (1D)');mbar.add_argument('-o','--output',default='landscape.npz')
    args=parser.parse_args(argv)
    if args.command=='run':
        cfg=json.loads(Path(args.config).read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
        sim=build(cfg)
        try:
            if args.restart:sim.restore_checkpoint(args.restart)
            result=sim.run(**cfg.get('run',{'steps':10000,'burnin':1000,'sample_every':10}))
            np.savez_compressed(out/'trajectory.npz',counts=result.pop('counts'),energy_kelvin=result.pop('energy_kelvin'),electrode_charges=result.pop('electrode_charges'))
            if result['positions']:
                offsets=np.r_[0,np.cumsum([len(x) for x in result['positions']])]
                np.savez_compressed(out/'positions.npz',offsets=offsets,positions=np.concatenate(result['positions']),types=np.concatenate(result['types']))
            result.pop('positions');result.pop('types');result['model_fingerprint']=sim.h.fingerprint()
            result['final_energy_full_recompute_error_K']=sim.h.energy-sim.h.full_energy()
            sim.save_checkpoint(out/'checkpoint.json');(out/'config.json').write_text(json.dumps(cfg,indent=2))
            with np.load(out/'trajectory.npz') as tr:
                summary=response_summary(tr['counts'],sim.temperature,bootstrap=50)
                if sim.h.electrodes:
                    pattern=np.asarray(cfg.get('voltage_pattern',[-.5]*(len(sim.h.electrodes.positions)//2)+[.5]*(len(sim.h.electrodes.positions)//2)))
                    summary['electrochemical']=electrochemical_response(tr['counts'],tr['electrode_charges'],pattern,sim.h.electrodes.response,sim.temperature)
                    summary['capacitance']=sim.h.electrodes.response.capacitance(tr['electrode_charges'],pattern,sim.temperature)
            (out/'response.json').write_text(json.dumps(jsonable(summary),indent=2))
            (out/'summary.json').write_text(json.dumps(jsonable(result),indent=2));print(json.dumps(jsonable(result),indent=2))
        finally:sim.h.close()
    elif args.command=='response':
        with np.load(args.trajectory) as f:r=response_summary(f['counts'],bootstrap=args.blocks)
        Path(args.output).write_text(json.dumps(jsonable(r),indent=2))
    else:
        with np.load(args.windows) as f:
            fit=MBAR(f['u_kn'],f['n_k']);r=landscape(f['cv'],[f['edges']],log_weights=fit.log_weights(f['u_target']))
            np.savez_compressed(args.output,free_energy_kBT=r['free_energy_kelvin'],probability=r['probability_mass'],counts=r['counts'],sampled=r['sampled'],overlap=fit.overlap,f_k=fit.f)
    return 0

if __name__=='__main__':raise SystemExit(main())
