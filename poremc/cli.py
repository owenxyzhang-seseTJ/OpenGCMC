"""Command-line entry points. JSON quantities carry units in their field names."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import math
import multiprocessing
from pathlib import Path
import platform
import sys
import time
import numpy as np
from . import __version__
from .topology import Molecule,activity_from_fugacity_pa,independent_torsion_log_partition,independent_bond_angle_log_partition
from .engine import GCMC, ZeroPotential
from .potentials import ClassicalPotential
from .io import read_cif, Framework, write_xyz
from .statistics import split_rhat
from .units import uptake_mmol_g


def clean_json(x):
    if isinstance(x,dict): return {str(k):clean_json(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [clean_json(v) for v in x]
    if isinstance(x,np.ndarray): return clean_json(x.tolist())
    if isinstance(x,(np.integer,)): return int(x)
    if isinstance(x,(float,np.floating)): return float(x) if math.isfinite(x) else None
    return x


def write_json(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(clean_json(data),indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temp.replace(path)


def construct_case(config,base_directory='.',*,checkpoint=None):
    config=copy.deepcopy(config); base=Path(base_directory)
    permitted={'schema','description','temperature_K','cell_A','species','host','potential','reservoir',
               'sampling','surrogate','root_bias','initial_particles','metadata'}
    unknown=set(config)-permitted
    if unknown: raise ValueError(f'unknown input fields: {sorted(unknown)}')
    if config.get('schema','poremc-v1')!='poremc-v1': raise ValueError('unsupported input schema')
    temperature=float(config['temperature_K']); species=[]
    for definition in config['species']:
        definition=copy.deepcopy(definition)
        if 'file' in definition:
            if len(definition)!=1: raise ValueError('species file cannot have inline overrides')
            definition=json.loads((base/definition['file']).read_text())
        normalization=definition.pop('internal_partition',None)
        m=Molecule.from_dict(definition)
        if normalization is not None:
            if normalization!='independent_bond_angle_torsions': raise ValueError('unknown internal partition method')
            # Analytic factorization is only valid if every internal nonbonded
            # pair is excluded and torsion degrees of freedom are independent.
            ex={tuple(sorted(x)) for x in m.exclusions}
            allpairs={(i,j) for i in range(m.n_atoms) for j in range(i+1,m.n_atoms)}
            if not m.is_rigid and ex!=allpairs:
                raise ValueError('factorized internal partition requires all intramolecular nonbonded pairs explicitly excluded')
            terminal=[max(t['indices']) for t in m.torsions]
            if len(terminal)!=len(set(terminal)):
                raise ValueError('multiple terms on one torsional coordinate need a combined partition integral')
            # Restrict automatic torsion factorization to the exact Z-matrix
            # coordinate being sampled; arbitrary quadruples can be coupled.
            for t in m.torsions:
                i=max(t['indices'])
                expected=[int(m.dihedral_refs[i]),int(m.angle_refs[i]),int(m.parents[i]),i]
                if t['indices']!=expected and t['indices']!=expected[::-1]:
                    raise ValueError('automatic torsion partition requires an independent Z-matrix dihedral')
            m.log_q_internal=independent_bond_angle_log_partition(m,temperature)+sum(independent_torsion_log_partition(t['k'],temperature) for t in m.torsions)
            m.q_internal_temperature=temperature
        species.append(m)
    if len({m.name for m in species})!=len(species): raise ValueError('species names must be unique')
    pconfig=dict(config.get('potential',{'kind':'classical','cutoff_A':9.0}))
    kind=pconfig.pop('kind','classical'); cutoff=pconfig.pop('cutoff_A',9.0)
    allowed_potential={'backend','algorithm','coulomb','alpha','kmax','shifted','threads','allow_charge_background','scale14_lj','scale14_coulomb'}
    if set(pconfig)-allowed_potential: raise ValueError(f'unknown potential options: {sorted(set(pconfig)-allowed_potential)}')
    cell=np.asarray(config.get('cell_A',np.eye(3)*30),float)
    if cell.shape==(3,): cell=np.diag(cell)
    host=Framework(cell,np.empty((0,3)),np.empty(0),np.empty(0),np.empty(0),[])
    hconfig=config.get('host')
    if hconfig:
        hconfig=dict(hconfig)
        if 'cif' in hconfig:
            ff=hconfig.get('force_field')
            if isinstance(ff,str): ff=json.loads((base/ff).read_text())
            if not isinstance(ff,dict): raise ValueError('CIF host needs an explicit force_field map')
            host=read_cif(base/hconfig['cif'],ff,hconfig.get('replicas',[1,1,1]),cutoff,
                          hconfig.get('charge_source','cif'))
            if 'cell_A' in config and not np.allclose(cell,host.cell): raise ValueError('explicit cell disagrees with CIF supercell')
            cell=host.cell
        else:
            allowed={'positions_A','sigma_A','epsilon_K','charge_e','mass_amu'}
            if set(hconfig)-allowed: raise ValueError('unknown inline host fields')
            positions=np.asarray(hconfig['positions_A'],float).reshape(-1,3); n=len(positions)
            def vector(key,default=0.): return np.broadcast_to(np.asarray(hconfig.get(key,default),float),(n,)).copy()
            host=Framework(cell,positions,vector('sigma_A'),vector('epsilon_K'),vector('charge_e'),['X']*n,hconfig.get('mass_amu'))
    if kind=='zero':
        if len(host.positions): raise ValueError('zero potential cannot silently ignore a host')
        if any(m.torsions or np.any(m.bond_force_constants) or np.any(m.angle_force_constants) for m in species):
            raise ValueError('zero potential would ignore specified internal interactions; use classical epsilon=0 instead')
        exact=ZeroPotential()
    elif kind=='classical':
        exact=ClassicalPotential(species,cell,cutoff,host.positions,host.sigma,host.epsilon,host.charge,**pconfig)
    else: raise ValueError('CLI supports zero/classical; ML total energy callbacks are supplied through the Python API')
    reservoir=config['reservoir']
    keys=set(reservoir)
    if 'activities_A3_inverse' in reservoir:
        if keys!={'activities_A3_inverse'}: raise ValueError('activities cannot be mixed with pressure/fugacity input')
        activities=reservoir['activities_A3_inverse']
    else:
        if 'fugacities_Pa' in reservoir:
            if keys!={'fugacities_Pa'}: raise ValueError('fugacities cannot be mixed with pressure input')
            fugacity=np.asarray(reservoir['fugacities_Pa'],float)
        else:
            if keys-{'pressure_Pa','mole_fractions','fugacity_coefficients'}: raise ValueError('unknown reservoir fields')
            fractions=np.asarray(reservoir['mole_fractions'],float)
            if np.any(fractions<0) or fractions.sum()>1+1e-12: raise ValueError('mole fractions must be nonnegative and sum <=1')
            coefficients=np.asarray(reservoir.get('fugacity_coefficients',np.ones(len(species))),float)
            if np.any(coefficients<=0): raise ValueError('fugacity coefficients must be positive')
            fugacity=float(reservoir['pressure_Pa'])*fractions*coefficients
        if np.shape(fugacity)!=(len(species),): raise ValueError('one fugacity per species is required')
        activities=[activity_from_fugacity_pa(f,temperature,m) for f,m in zip(fugacity,species)]
    proposal=exact; surrogate_info=None
    if config.get('surrogate'):
        from .accelerators import GridHostSurrogate,LearnedHostSurrogate
        options=dict(config['surrogate']); skind=options.pop('kind')
        if skind=='grid': proposal=GridHostSurrogate(exact,**options)
        elif skind=='learned_fourier': proposal=LearnedHostSurrogate(exact,**options)
        else: raise ValueError('surrogate kind must be grid or learned_fourier')
        surrogate_info=proposal.metadata
    root_proposal=None
    if config.get('root_bias'):
        from .proposals import build_host_proposal
        options=dict(config['root_bias'])
        if options.pop('kind','host_alias')!='host_alias':raise ValueError('unknown root bias')
        if not isinstance(exact,ClassicalPotential):raise ValueError('host alias requires a classical target')
        if options.get('cache_dir') is not None:options['cache_dir']=str(base/options['cache_dir'])
        root_proposal=build_host_proposal(exact,species[0],temperature,**options)
    sample=dict(config.get('sampling',{}))
    driver=sample.pop('driver','python')
    if driver not in ('python','cpp'): raise ValueError('sampling.driver must be python or cpp')
    run_args={k:sample.pop(k) for k in ('steps','burnin','sample_every','adapt','record_energy') if k in sample}
    run_args={'steps':10000,'burnin':1000,'sample_every':10,**run_args}
    allowed_sampling={'seed','trials','move_weights','species_probabilities','max_translation','max_rotation','max_particles'}
    bias_spec=sample.pop('count_bias',None)
    bias=None
    if bias_spec is not None:
        if driver!='python':
            raise ValueError('count_bias requires the Python GCMC driver')
        from .free_energy import HarmonicCountBias
        bias=HarmonicCountBias(**bias_spec)
        if len(bias.coefficients)!=len(species):
            raise ValueError('count bias requires one coefficient per species')
    if driver=='cpp':
        allowed_sampling|={'reciprocal_method','pme_mesh','reciprocal_backend','cuda_library',
                           'sampling_mode','host_grid_mesh','host_grid_cap','host_grid_cache',
                           'grid_root_bias','ewald_half_spectrum','ewald_k_cutoff','host_grid_steep',
                           'guest_update','pme_refresh','partial_regrowth','host_field_cache',
                           'short_range_backend','short_range_min_batch'}
        if sample.get('cuda_library') is not None:sample['cuda_library']=str((base/sample['cuda_library']).resolve())
        if sample.get('host_grid_cache') is not None:sample['host_grid_cache']=str((base/sample['host_grid_cache']).resolve())
        if sample.get('host_field_cache') is not None:sample['host_field_cache']=str((base/sample['host_field_cache']).resolve())
    if set(sample)-allowed_sampling: raise ValueError(f'unknown sampling options: {sorted(set(sample)-allowed_sampling)}')
    if checkpoint is not None and driver=='cpp':
        from .fast import FastGCMC
        simulation=FastGCMC.from_checkpoint(checkpoint,proposal)
    elif checkpoint is not None:
        simulation=GCMC.from_checkpoint(checkpoint,proposal,exact if proposal is not exact else None)
    elif driver=='cpp':
        from .fast import FastGCMC
        simulation=FastGCMC(species,proposal,cell,temperature,activities,target_potential=exact if proposal is not exact else None,root_proposal=root_proposal,**sample)
    else:
        simulation=GCMC(species,proposal,cell,temperature,activities,
                        target_potential=exact if proposal is not exact else None,
                        root_proposal=root_proposal,count_bias=bias,**sample)
    if checkpoint is None:
        for particle in config.get('initial_particles',[]): simulation.add_particle(particle['species'],particle['positions_A'])
    physics={'cell':cell.tolist(),'species':[m.to_dict() for m in species],
             'host':{k:v for k,v in host.to_dict().items() if k not in ('source','labels')},
             'potential':config.get('potential'),'temperature_K':temperature,'activities':list(activities),
             'surrogate':config.get('surrogate'),'root_bias':config.get('root_bias'),
             'count_bias':bias_spec,'execution_driver':driver}
    if driver=='cpp':
        method=sample.get('reciprocal_method','direct')
        mesh=tuple(map(int,np.broadcast_to(sample.get('pme_mesh',32),(3,)))) if method=='pme' else None
        physics['long_range']={'method':method,'mesh':mesh,'k_cutoff':float(sample.get('ewald_k_cutoff',0.))}
    from .cache import content_key
    fingerprint=content_key(physics,{})
    if checkpoint is not None and checkpoint.get('physics_id')!=fingerprint:
        if hasattr(simulation,'close'):simulation.close()
        raise ValueError('restart physics fingerprint differs from the input model and reservoir')
    return simulation,run_args,host,fingerprint,surrogate_info


def run_configuration(config,output_directory,base_directory='.',restart=None):
    started=time.perf_counter()
    checkpoint=json.loads(Path(restart).read_text()) if restart else None
    sim,run_args,host,fingerprint,surrogate=construct_case(config,base_directory,checkpoint=checkpoint)
    if restart:
        run_args['burnin']=0; run_args['adapt']=False
    result=sim.run(**run_args)
    result['metadata']={'poremc_version':__version__,'python':sys.version.split()[0],'platform':platform.platform(),
                        'driver':config.get('sampling',{}).get('driver','python'),'temperature_K':sim.temperature,'volume_A3':float(np.linalg.det(sim.cell)),'activities_A3_inverse':sim.activities.tolist(),
                        'physics_id':fingerprint,'input_description':config.get('description'),
                        'surrogate':surrogate,'total_wall_seconds':time.perf_counter()-started,
                        'energy_unit':'U/kB in Kelvin','length_unit':'Angstrom','step_definition':'one attempted move'}
    if host.mass_amu:
        result['uptake_mmol_g']={m.name:uptake_mmol_g(result['species_statistics'][m.name]['mean'],host.mass_amu)
                                for m in sim.species if m.name in result['species_statistics']}
    out=Path(output_directory); out.mkdir(parents=True,exist_ok=True)
    write_json(out/'result.json',result); write_json(out/'input.json',config)
    checkpoint=sim.checkpoint(); checkpoint['physics_id']=fingerprint
    write_json(out/'checkpoint.json',checkpoint)
    write_xyz(out/'final.xyz',sim.entries(),sim.species)
    return result


def _batch_item(item):
    return run_configuration(item['config'],item['output'],item['base'])


def main(argv=None):
    parser=argparse.ArgumentParser(description='OpenGCMC: cache-first molecular GCMC with exact energy evaluation')
    sub=parser.add_subparsers(dest='command',required=True)
    run=sub.add_parser('run'); run.add_argument('input'); run.add_argument('-o','--output',default='poremc-output'); run.add_argument('--restart')
    validate=sub.add_parser('validate'); validate.add_argument('input')
    batch=sub.add_parser('replicas'); batch.add_argument('input'); batch.add_argument('-o','--output',default='poremc-replicas'); batch.add_argument('-n','--replicas',type=int,default=4); batch.add_argument('--workers',type=int,default=2)
    inspect=sub.add_parser('capabilities')
    convert=sub.add_parser('import-raspa3'); convert.add_argument('input'); convert.add_argument('-o','--output',required=True); convert.add_argument('--steps',type=int,required=True); convert.add_argument('--burnin',type=int,default=1000)
    args=parser.parse_args(argv)
    try:
        if args.command=='capabilities':
            print(json.dumps({'version':__version__,'core':['rigid and tree flexible GCMC','bounded harmonic bond/angle plus torsion',
                'multicomponent CBMC','two-stage target correction','native cell-list LJ/DSF/Ewald',
                'native host-grid CBMC with exact residual correction',
                'CPU harmonic reciprocal contraction','direct Ewald and PME',
                'CIF explicit symmetry','exact checkpoint','CPU replica parallelism','native multicomponent sampler','incremental guest cells and deferred PME refresh'],
                'python_api':['NPT bulk','Gibbs bulk','replica exchange','TMMC controller','TorchScript total-energy adapter'],
                'thermodynamics':['chemical-potential response matrices','occupancy and charge free-energy landscapes',
                    'stoichiometric electrolyte GCMC','Gaussian constant-potential electrodes'],
                'electrochemistry':['explicit SPC/E hydration','fixed-topology protonation, redox and PCET sampling',
                    'staged NCMC and multistate free-energy calibration','reaction networks and finite-volume diffusion currents'],
                'entry_points':{'molecular':'opengcmc','thermodynamics':'opengcmc-thermo','electrochemistry':'opengcmc-reactive'},
                'native_reciprocal_backends':['cpu','auto','cuda','cuda-tensor','cuda-resident','cuda-pme'],
                'gpu_status':'Direct Ewald CUDA candidate batches and optional FP64 tensor contraction for exact batches and selected delayed structures; delayed corrections support CPU or opted-in tensor CUDA',
                'not_implemented':['ring-closure CBMC','flexible framework','CFCMC','molecular dynamics',
                    'arbitrary reactive topology changes','general bulk polarization/QEq',
                    'general RASPA/gRASPA input compatibility']} ,indent=2)); return 0
        if args.command=='import-raspa3':
            from .raspa_adapter import import_raspa3
            converted=import_raspa3(args.input,args.steps,args.burnin)
            write_json(args.output,converted); print(f'Converted supported rigid subset to {args.output}'); return 0
        path=Path(args.input).resolve(); config=json.loads(path.read_text())
        if args.command=='validate':
            sim,runargs,host,fp,sm=construct_case(config,path.parent)
            print(json.dumps({'valid':True,'species':[m.name for m in sim.species],'activities_A3_inverse':sim.activities.tolist(),
                              'host_atoms':len(host.positions),'run':runargs,'physics_id':fp},indent=2)); return 0
        if args.command=='run':
            result=run_configuration(config,args.output,path.parent,args.restart)
            print(json.dumps(clean_json({'output':str(Path(args.output).resolve()),'statistics':result['species_statistics'],
                             'moves_per_second':result['attempted_moves_per_second'],'final_counts':result['final_counts']}),indent=2)); return 0
        if args.replicas<2 or args.workers<1: raise ValueError('need at least two replicas and one worker')
        seed=config.get('sampling',{}).get('seed',2026)
        seeds=np.random.SeedSequence(seed).spawn(args.replicas)
        jobs=[]
        for i,s in enumerate(seeds):
            c=copy.deepcopy(config); c.setdefault('sampling',{})['seed']=int(s.generate_state(1)[0])
            jobs.append({'config':c,'output':str(Path(args.output)/f'replica_{i:03d}'),'base':str(path.parent)})
        start=time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            results=list(pool.map(_batch_item,jobs))
        names=list(results[0]['species_statistics'])
        rhats={name:split_rhat([np.array(r['counts'])[:,i] for r in results]) for i,name in enumerate(names)}
        summary={'replicas':args.replicas,'workers':args.workers,'wall_seconds':time.perf_counter()-start,
                 'classical_split_rhat':rhats,'replica_means':[{s:v['mean'] for s,v in r['species_statistics'].items()} for r in results]}
        write_json(Path(args.output)/'replica_summary.json',summary); print(json.dumps(clean_json(summary),indent=2)); return 0
    except (ValueError,KeyError,RuntimeError,FileNotFoundError,FloatingPointError) as exc:
        print(f'OpenGCMC: {exc}',file=sys.stderr); return 2

if __name__=='__main__': raise SystemExit(main())
