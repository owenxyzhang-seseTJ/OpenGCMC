"""Run RASPA3 CBMC with the same force field and preloaded coordinates."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

from benchmark_precision import SEEDS, read_model, summarize, trans_count, write_json
from poremc.cli import construct_case


SITE_MASS={'H':1.00794,'C':12.0107,'N':14.0067,'O':15.9994,'Si':28.0855,
           'Ni':58.6934,'Zn':65.38,'Cu':63.546,'Zr':91.224}


def write_cif(path, cell, pos, names, elements):
    lengths=np.linalg.norm(cell,axis=1)
    unit=cell/lengths[:,None]
    angle=np.degrees(np.arccos(np.clip([unit[1]@unit[2],unit[0]@unit[2],unit[0]@unit[1]],-1,1)))
    frac=np.mod(pos@np.linalg.inv(cell),1)
    with path.open('w') as f:
        f.write('data_framework\n_symmetry_space_group_name_H-M "P 1"\n_symmetry_Int_Tables_number 1\n')
        for key,value in zip(('length_a','length_b','length_c','angle_alpha','angle_beta','angle_gamma'),
                             [*lengths,*angle]):
            f.write(f'_cell_{key} {value:.14g}\n')
        f.write('loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n'
                '_atom_site_fract_y\n_atom_site_fract_z\n_atom_site_occupancy\n')
        for i,(name,element,xyz) in enumerate(zip(names,elements,frac)):
            # RASPA3 resolves a CIF label by stripping its trailing digits.
            f.write(f'{name}{i} {element} '+' '.join(f'{x:.15g}' for x in xyz)+' 1\n')


def framework_symbol(sigma, epsilon, fallback):
    # Element codes affect output masses. LJ and charge come from the model.
    # Identify the two embedded UFF/DREIDING framework inputs by atom type.
    types=[(2.571,22.14,'H'),(2.846,7.65,'H'),(3.431,52.84,'C'),(3.473,47.86,'C'),
           (3.261,34.72,'N'),(3.118,30.20,'O'),(3.033,48.17,'O'),
           (2.524,7.55,'Ni'),(2.461,62.9,'Zn')]
    near=[el for s,e,el in types if abs(sigma-s)<.015 and abs(epsilon-e)<1.5]
    return near[0] if near else fallback


def make_model(path,out):
    out.mkdir(parents=True,exist_ok=False)
    model=read_model(path)
    native=dict(model)
    native['sampling']={**model['sampling'],'sampling_mode':'exact','grid_root_bias':0.}
    with construct_case(native,path.parent)[0] as sim:
        species=sim.species
        p=sim.potential
        if p.shifted=='force' or model['sampling'].get('ewald_k_cutoff'):
            raise ValueError('RASPA comparison requires the shared rectangular Ewald set and common LJ truncation')
        atoms=[]
        interactions=[]
        def site(name,el,integer,sigma,eps,charge,mass):
            atoms.append({'name':name,'framework':integer,'print_to_output':not integer,
                          'element':el,'print_as':el,'mass':mass,'charge':float(charge)})
            interactions.append({'name':name,'type':'lennard-jones',
                                 'parameters':[float(eps),float(sigma)]})
        codes={}
        atom_names=[]
        elements=[]
        tags=model['metadata'].get('framework_site_types',[None]*len(p.host_positions))
        if 'cif' in model['host']:
            fields=model['host']['force_field']
            tags_by_pair={(float(spec['sigma']),float(spec['epsilon'])):name for name,spec in fields.items()}
            tags=[tags_by_pair.get((s,e)) for s,e in zip(p.host_sigma,p.host_epsilon)]
        for tag,s,e,q in zip(tags,p.host_sigma,p.host_epsilon,p.host_charge):
            el=tag if tag in SITE_MASS else framework_symbol(s,e,'C')
            key=(el,s,e,q)
            if key not in codes:
                m=len(codes)
                name='H'+chr(65+m//26)+chr(65+m%26)
                codes[key]=name
                site(name,el,True,s,e,q,SITE_MASS[el])
            atom_names.append(codes[key])
            elements.append(el)
        write_cif(out/'framework.cif',sim.cell,p.host_positions,atom_names,elements)
        first_bead_names=[]
        guest_site_names=[]
        for sid,m in enumerate(species):
            names=[f'{chr(65+sid)}{k:03}' for k in range(m.n_atoms)]
            first_bead_names.append(names[0])
            guest_site_names.append(names)
            for name,s,e,q in zip(names,m.sigma,m.epsilon,m.charge):
                site(name,'C',False,s,e,q,m.molar_mass/m.n_atoms)
            xyz=m.rigid_positions if m.rigid_positions is not None else np.zeros((m.n_atoms,3))
            spec={'CriticalTemperature':400.,'CriticalPressure':5e6,'AcentricFactor':.20,
                  'Type':'rigid' if m.is_rigid else 'flexible',
                  'pseudoAtoms':[[name,coord.tolist()] for name,coord in zip(names,xyz)]}
            if not m.is_rigid:
                if m.n_atoms!=4 or m.name!='n-butane':
                    raise ValueError('the flexible translator expects the TraPPE-UA butane input')
                spec['Connectivity']=[[0,1],[1,2],[2,3]]
                spec['Bonds']=[[[names[i],names[i+1]],'FIXED',[1.54]] for i in range(3)]
                bend=model['metadata'].get('bend_K_rad2')
                spec['Bends']=[[names[i:i+3],'HARMONIC',[bend,114.]] if bend else
                               [names[i:i+3],'FIXED',[114.]] for i in range(2)]
                spec['Torsions']=[[names,'TRAPPE',[0.,355.03,-68.19,791.32]]]
            write_json(out/f'{m.name}.json',spec)
        force_field={'PseudoAtoms':atoms,'SelfInteractions':interactions,
                     'MixingRule':'Lorentz-Berthelot','CutOff':p.cutoff,
                     'TruncationMethod':'shifted' if p.shifted else 'truncated','TailCorrections':False}
        if p.ewald:
            force_field['EwaldParameters']=[p.alpha,*p.ewald.kmax.tolist()]
        write_json(out/'force_field.json',force_field)
        q_reference=[math.exp(m.log_q_internal or 0.) for m in species]
        if model['metadata'].get('raspa_q_reference') is not None:
            q_reference[0]=model['metadata']['raspa_q_reference']
        return {'names':[m.name for m in species],'first_beads':first_bead_names,'site_names':guest_site_names,
                'temperature_K':sim.temperature,'charges':p.coulomb=='ewald','weights':sim.move_weights,
                'q_reference':q_reference}


def make_inputs(model_path,directory,seed,start,max_cycles,warmup,stride,mode):
    info=make_model(model_path,directory)
    model=read_model(model_path)
    reservoir=model['reservoir']
    if 'fugacities_Pa' in reservoir:
        fugacities=np.asarray(reservoir['fugacities_Pa'])
    else:
        fugacities=reservoir['pressure_Pa']*np.asarray(reservoir['mole_fractions'])*reservoir.get('fugacity_coefficients',1.)
    p=float(fugacities.sum())
    counts=[0]*len(info['names'])
    coords=defaultdict(list)
    for sid,positions in start:
        counts[sid]+=1
        coords[info['names'][sid]].extend(positions)
    weights=info['weights']
    components=[]
    for i,name in enumerate(info['names']):
        component={'Name':name,'MolFraction':float(fugacities[i]/p),'FugacityCoefficient':1.,
                   'IdealGasRosenbluthWeight':info['q_reference'][i],
                   'TranslationProbability':weights['translate'],'RotationProbability':weights['rotate'],
                   'ReinsertionProbability':weights['regrow'],'CreateNumberOfMolecules':0}
        swap='SwapProbability' if mode=='cbmc' else 'CFCMC_CBMC_SwapProbability'
        component[swap]=weights['insert']+weights['delete']
        components.append(component)
    system={'Type':'Framework','Name':'framework','NumberOfUnitCells':[1,1,1],
            'ChargeMethod':'Ewald' if info['charges'] else 'None',
            'ExternalTemperature':info['temperature_K'],'ExternalPressure':p,
            'OutputPDBMovie':True,'SampleMovieEvery':stride,'RestrictMoviePositionsToBox':True}
    if start:
        write_json(directory/'start.json',coords)
        system['RestartFileName']='start.json'
    if mode!='cbmc':
        system['ComputeNumberOfMoleculesEvolution']=True
    init=warmup if mode=='cbmc' else 5000
    equil=warmup-init
    write_json(directory/'simulation.json',
               {'SimulationType':'MonteCarlo','RandomSeed':seed,'NumberOfCycles':max_cycles+int(mode=='cfcmc'),
                'NumberOfInitializationCycles':init,'NumberOfEquilibrationCycles':equil,
                'PrintEvery':max(100,warmup//2),'WriteBinaryRestartEvery':10*max_cycles,
                'Systems':[system],'Components':components})
    return info,counts


def observe_movie_line(line, counts, trans, info, flex, frame):
    names=info['first_beads']
    if line.startswith('ATOM  '):
        frame.append((line[12:16].strip(),line))
    elif line.startswith('ENDMDL'):
        keys=[k for k,_ in frame]
        count=[keys.count(n) for n in names]
        if len(frame)!=sum(len(s)*n for s,n in zip(info['site_names'],count)):
            raise ValueError('RASPA PDB movie/site count mismatch')
        counts.append(count)
        if flex:
            xyz=[[float(row[30:38]),float(row[38:46]),float(row[46:54])] for _,row in frame]
            trans.append(trans_count(np.asarray(xyz).reshape(-1,4,3)))
        frame.clear()


def raspa_chain(job):
    path, output, binary, seed, start, mode, budgets, warmup, stride, origin=job
    raw=output.with_suffix('')/str(seed)
    info,initial=make_inputs(path,raw,seed,start,budgets[-1],warmup,stride,mode)
    flex=read_model(path)['metadata'].get('observe_trans',False)
    counts,trans,frame,stamps=[],[],[],[]
    seen=0
    movie=raw/'movies'/'movie.s0.pdb'
    with (raw/'stdout.log').open('w') as log:
        start_time=time.perf_counter()
        process=subprocess.Popen([str(binary)],cwd=raw,stdout=log,stderr=subprocess.STDOUT)
        try:
            while True:
                if movie.exists():
                    if movie.stat().st_size!=seen:
                        with movie.open() as lines:
                            lines.seek(seen)
                            # Files are appended by the simulator; read only complete lines.
                            for line in iter(lines.readline,''):
                                if not line.endswith('\n'):
                                    break
                                seen+=len(line.encode())
                                observe_movie_line(line,counts,trans,info,flex,frame)
                while len(stamps)<len(budgets) and len(counts)>=budgets[len(stamps)]//stride:
                    stamps.append(time.perf_counter()-origin)
                status=process.poll()
                if status is not None:
                    if status:
                        raise RuntimeError(f'RASPA3 exited {status}: {raw}/stdout.log')
                    if len(stamps)==len(budgets):
                        break
                    raise RuntimeError(f'RASPA3 movie incomplete: have {len(counts)}, need {budgets[-1]//stride}')
                time.sleep(.10)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
    filename=f'{output.stem}_{seed}.npz'
    final_json=next((raw/'output').glob('output_*.json'),None)
    if mode!='cbmc':
        evolution=raw/'number_of_molecules_evolution'/'number_of_molecules_evolution.s0.txt'
        data=np.loadtxt(evolution,dtype=int,ndmin=2)
        counts=data[warmup:warmup+budgets[-1]:stride]
        if len(counts)!=budgets[-1]//stride:
            raise ValueError('RASPA3 integer-molecule evolution did not reach the requested production length')
    keep=budgets[-1]//stride
    np.savez_compressed(output.parent/filename,counts=np.asarray(counts[:keep]),trans_count=np.asarray(trans[:keep]))
    return {'seed':seed,'start':'preloaded' if start else 'empty','initial_counts':initial,
            'elapsed_at_prefix_s':stamps,'subprocess_wall_s':time.perf_counter()-start_time,
            'raspa_output':str(final_json.relative_to(output.parent)) if final_json else None,'data':filename}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model',type=Path)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=('cbmc','cfcmc'),default='cbmc')
    p.add_argument('--budgets',type=int,nargs='+',default=[1000,2500,5000,10000])
    p.add_argument('--warmup',type=int,default=1000)
    p.add_argument('--stride',type=int,default=5)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    model=read_model(a.model)
    cycle_cap=model['metadata'].get('raspa_cycle_cap')
    if cycle_cap and a.budgets[-1]>cycle_cap:
        a.budgets=[b for b in a.budgets if b<cycle_cap]+[cycle_cap]
        print(f'{a.model.stem}: production is capped at {cycle_cap} cycles',flush=True)
    if any(x%a.stride or x<4*a.stride for x in a.budgets) or a.budgets!=sorted(set(a.budgets)):
        p.error('use increasing budgets aligned to the observation period')
    if a.mode=='cfcmc' and (a.warmup<5000 or (a.warmup+a.budgets[-1])%5000):
        p.error('CFCMC needs initialization and a final integer-count output boundary')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    starts=read_model(a.model.with_name(a.model.stem+'_starts.json'))
    origin=time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        chains=list(pool.map(raspa_chain,[(a.model,a.output,a.binary.resolve(),seed,start,a.mode,
                                          a.budgets,a.warmup,a.stride,origin)
                                         for seed,start in zip(SEEDS,starts)]))
    if a.mode=='cbmc':
        result=summarize(chains,model['metadata'],[b//a.stride for b in a.budgets],
                         a.budgets,a.output,a.model.with_name(a.model.stem+'_nist.npz'))
    else:
        # This stream contains integer loadings across all lambda values.
        # The raw expanded-ensemble chain cannot use the CBMC precision rule.
        result={'expanded_ensemble':True,'qualified_wall_s':None,'production_budgets':a.budgets}
    result.update(case=a.model.stem,engine='RASPA3',mode=a.mode,platform=platform.platform(),
                  warmup_cycles=a.warmup,sample_every=a.stride,workers=a.workers,chains=chains,
                  production_cycle_cap=cycle_cap)
    write_json(a.output,result)
    print(a.model.stem,a.mode,'qualified_wall_s',result['qualified_wall_s'],flush=True)


if __name__=='__main__':
    main()
