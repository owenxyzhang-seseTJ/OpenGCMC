"""Recompute RASPA3 CBMC final-state energies and check flexible geometry."""

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark_precision import read_model, setup, write_json


def geometry(x):
    bonds=np.diff(np.asarray(x),axis=1)
    lengths=np.linalg.norm(bonds,axis=-1)
    u=bonds/lengths[...,None]
    angles=np.degrees(np.arccos(np.clip(-np.sum(u[:,:-1]*u[:,1:],axis=-1),-1,1)))
    return {'bonds_A':np.quantile(lengths,[0.,.01,.5,.99,1.],axis=0),
            'bend_degrees':np.quantile(angles,[0.,.01,.5,.99,1.],axis=0)}


def check_case(case, model_dir, result_dir):
    path=model_dir/f'{case}.json'
    model=read_model(path)
    out=[]
    for stat in sorted((result_dir/f'{case}-raspa-cbmc').glob('*/output/output_*.json')):
        data=read_model(stat)
        frame=read_model(stat.with_name(stat.name.replace('output_','restart_',1)))
        seed=int(stat.parent.parent.name)
        with setup(model,path,seed,'exact',Path('.'),32,'incremental') as sim:
            poses=[]
            for sid,mol in enumerate(sim.species):
                xyz=np.asarray(frame.get(mol.name,[]),dtype=float).reshape(-1,mol.n_atoms,3)
                poses.append(xyz)
                for pose in xyz:
                    sim.add_particle(sid,pose)
            e=sim.total_energy()
            r=data['output']['recomputedEnergies']['Total potential energy [K]']
            row={'seed':seed,'counts':sim.counts.tolist(),'RASPA3_version':data['version'],
                 'RASPA3_Ewald':data['initialization']['forceField']['Ewald'],
                 'RASPA3_temperature_K':data['initialization']['initialConditions']['temperature'],
                 'OpenGCMC_energy_K':e,'RASPA3_energy_K':r,'absolute_difference_K':abs(e-r)}
            if model['metadata'].get('observe_trans') and len(poses[0]):
                row['geometry']=geometry(poses[0])
        out.append(row)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs',type=Path)
    p.add_argument('results',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('cases',nargs='+')
    a=p.parse_args()
    write_json(a.output,{case:check_case(case,a.inputs,a.results) for case in a.cases})
