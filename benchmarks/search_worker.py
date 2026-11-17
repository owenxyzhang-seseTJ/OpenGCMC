"""Analytic search-space experiment: noninteracting particles in a voxel well.

Not a fitted MOF: the exact particle-number law is Poisson(lambda=6).
Both samplers use the same Hamiltonian; only the normalized root proposal differs.
"""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from poremc.engine import GCMC
from poremc.topology import Molecule
from poremc.proposals import VoxelProposal
from poremc.cli import clean_json
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=37);p.add_argument('--steps',type=int,default=50000);a=p.parse_args()
T=300.;cell=np.eye(3)*40;shape=(4,4,4);field=np.full(shape,18*T);field[0,0,0]=0
class VoxelPotential:
    def molecule_energies(self,coordinates,molecule,environment):
        f=np.mod(np.asarray(coordinates)[:,0,:]/40,1);idx=np.minimum((f*4).astype(int),3)
        return field[idx[:,0],idx[:,1],idx[:,2]]
    def molecule_energy(self,coordinates,molecule,environment):return float(self.molecule_energies(np.asarray(coordinates)[None],molecule,environment)[0])
pot=VoxelPotential();m=Molecule('probe',[0],[0]);mean_weight=float(np.exp(-field/T).mean());z=6/(np.linalg.det(cell)*mean_weight)
records={}
for name in ['uniform','alias']:
    bias=VoxelProposal.from_energies(cell,field,T,.01) if name=='alias' else None
    sim=GCMC([m],pot,cell,T,[z],seed=a.seed,trials=1,move_weights={'insert':.5,'delete':.5},root_proposal=bias)
    # Start both from the same valid state; burn-in still run and excluded.
    for i in range(6):sim.add_particle(0,np.array([[1+i,1,1.]]))
    result=sim.run(a.steps,burnin=5000,sample_every=5,record_energy=False)
    records[name]=result
records.update(seed=a.seed,expected_mean=6.,expected_variance=6.,well_fraction=1/64,barrier_over_kBT=18.,uniform_fraction=.01,activity_A3_inverse=z)
records['ess_per_second_gain']=records['alias']['species_statistics']['probe']['effective_samples_per_second']/records['uniform']['species_statistics']['probe']['effective_samples_per_second']
path=ROOT/'benchmarks'/f'search_{a.seed}.json';path.write_text(json.dumps(clean_json(records),indent=2))
print(json.dumps({'seed':a.seed,'gain':records['ess_per_second_gain'],'uniform':records['uniform']['species_statistics'],'alias':records['alias']['species_statistics']},indent=2))
