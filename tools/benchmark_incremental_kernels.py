"""Paired CPU timings for accepted guest updates and deferred PME transforms."""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics

import numpy as np

from poremc.fast import FastGCMC
from poremc.potentials import ClassicalPotential
from poremc.scheduling import _publish
from poremc.topology import Molecule


def _guest_comparison(repeats):
    m=Molecule.from_dict({'name':'LJ','sigma':[3.],'epsilon':[80.]})
    cell=np.eye(3)*60.
    potential=ClassicalPotential([m],cell,8.,algorithm='cell')
    xyz=np.asarray(list(np.ndindex(8,8,8)),float)*7.5+2.
    output=[]
    for count in (8,96,384):
        rows=[]
        for i in range(repeats):
            rng=np.random.default_rng(891+i)
            positions=xyz[rng.choice(len(xyz),count,replace=False)]
            runs={}
            for mode in (('rebuild','incremental') if i%2==0 else ('incremental','rebuild')):
                with FastGCMC([m],potential,cell,298.,[0.],seed=819+i,move_weights={'translate':1},
                              guest_update=mode,trials=1) as sim:
                    for x in positions:
                        sim.add_particle(0,x[None,:])
                    result=sim.run(10000,burnin=1000,sample_every=100,record_energy=False)
                    runs[mode]={'seconds':result['production_seconds'],
                                'final_energy_K':sim.total_energy(),
                                'accepts':sim.counters['translate']['accepted'],
                                'guest_builds':sim.profile['guest_environment_builds']}
            rows.append(runs)
        output.append({'molecules':count,'rows':rows,
                       'speedup_median':statistics.median(r['rebuild']['seconds']/r['incremental']['seconds'] for r in rows),
                       'maximum_final_energy_difference_K':max(abs(r['rebuild']['final_energy_K']-
                                                                  r['incremental']['final_energy_K']) for r in rows)})
    return output


def _mesh_comparison(repeats):
    m=Molecule.from_dict({'name':'charged dimer','sigma':[2.4,2.4],'epsilon':[75.,75.],
                          'charge':[.25,-.25],'rigid_positions':[[0,0,0],[1.25,0,0]]})
    cell=np.eye(3)*34.
    potential=ClassicalPotential([m],cell,9.,coulomb='ewald',kmax=6)
    rows=[]
    for i in range(repeats):
        runs={}
        for refresh in ((1,8) if i%2==0 else (8,1)):
            with FastGCMC([m],potential,cell,298.,[6./34**3],seed=22+i,trials=2,
                          reciprocal_method='pme',pme_mesh=32,pme_refresh=refresh) as sim:
                result=sim.run(1000,burnin=100,sample_every=5)
                runs[refresh]={'seconds':result['production_seconds'],
                               'fft_transforms':sim.long_range_profile['fft_transforms'],
                               'energy':result['energies_kelvin'],'counts':result['counts']}
        a,b=(runs[j] for j in (1,8))
        error=float(np.max(np.abs(np.array(a.pop('energy'))-b.pop('energy'))))
        rows.append({'immediate':a,'deferred':b,'energy_max_difference_K':error,
                     'count_traces_equal':a.pop('counts')==b.pop('counts')})
    return {'rows':rows,
            'speedup_median':statistics.median(r['immediate']['seconds']/r['deferred']['seconds'] for r in rows)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    _publish(a.output,{'replicas':a.repeats,'guest_cells':_guest_comparison(a.repeats),
                       'deferred_pme':_mesh_comparison(a.repeats)})


if __name__=='__main__':
    main()
