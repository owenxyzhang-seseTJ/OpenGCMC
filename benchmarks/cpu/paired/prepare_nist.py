"""Convert NIST FEASST particles and reweight independent TMMC replicas."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import softmax

from poremc.units import KB, NA


KJ_MOL_TO_K = 1000/(KB*NA)
CASES = {
    'nist_co2_zif8': ('CO2_REF_ISOTHERMS/CO2_ZIF8', 'data.ZIF8_Snurr_rep222',
                     'co2_zif8.300K', 'co2_zif8.300K.r[0-9].csv',
                     21846.91, [50000.], 'force', [14]),
    'nist_n2_co2_zif8': ('N2_CO2_REF_ISOTHERMS/ZIF8', 'data.ZIF8_Snurr_rep222',
                         'binary_n2_co2_ZIF8.300K', 'n2_co2_zif8.300K.lnpi.r[0-9].csv',
                         21846.91, [50000.,50000.], False, [1,13]),
    'holdout_co2_uio66': ('CO2_REF_ISOTHERMS/CO2_UiO66', 'data.UiO66_Snurr_rep222',
                         'co2_uio66.350K', 'co2_uio66.350K.r[0-9].csv',
                         53249.82, [60000.], 'force', [12]),
}


def particles(path):
    tables = {s:[] for s in ('Site Labels','Site Properties','Sites')}
    section = None
    for text in path.read_text().splitlines():
        line = text.split('#',1)[0].strip()
        if line in tables:
            section = line
        elif line:
            fields = line.split()
            if section and fields[0].isdigit():
                tables[section].append(fields)
            else:
                section = None
    types = {int(row[0]):row[1] for row in tables['Site Labels']}
    properties = {}
    for row in tables['Site Properties']:
        spec = dict(zip(row[1::2], row[2::2]))
        if float(spec['cutoff']) != 12.0:
            raise ValueError('unexpected FEASST cutoff')
        properties[int(row[0])] = (float(spec['sigma']), float(spec['epsilon'])*KJ_MOL_TO_K,
                                   float(spec['charge']))
    atoms = np.array([[*map(float,row[2:]), *properties[int(row[1])]]
                      for row in tables['Sites']])
    labels = [types[int(row[1])] for row in tables['Sites']]
    assert atoms.ndim==2 and atoms.shape[1]==6
    return atoms, labels


def prepare(source, output, case):
    folder, host_file, meta, replicas, mass, fugacities, shift, preload = CASES[case]
    folder = source/folder
    meta_path = folder/(meta+'.metadata.json')
    metadata = json.loads(meta_path.read_text())
    temperature = metadata['temperature']
    dims = np.asarray(metadata['cell'])
    alpha = float(metadata['ewald_parameters']['alpha'])
    kmax = list(map(int,metadata['ewald_parameters']['kvectors']))
    if not np.all(dims == dims[0]) or min(kmax) != max(kmax):
        raise ValueError('these NIST recipes use cubic cells and equal reciprocal bounds')
    # FEASST excludes the equality shell. In a cubic cell k^2 is integral in
    # units of (2 pi/L)^2. The midpoint implements the strict sphere robustly.
    radial = 2*math.pi/dims[0]*math.sqrt(kmax[0]**2-.5)
    atoms, labels = particles(folder/host_file)
    co2, _ = particles(source/'CO2_REF_ISOTHERMS/CO2_Bulk/data.CO2_Trappe')
    guests = [dict(name='CO2',sigma=co2[:,3].tolist(),epsilon=co2[:,4].tolist(),
                   charge=co2[:,5].tolist(),rigid_positions=co2[:,:3].tolist(),molar_mass=44.0095)]
    if len(fugacities)==2:
        guests.insert(0,dict(name='N2',sigma=[0.,3.31,3.31],epsilon=[0.,36.,36.],
                             charge=[.964,-.482,-.482],rigid_positions=[[0.,0.,0.],[0.,-.55,0.],[0.,.55,0.]],
                             molar_mass=28.0134))
    z = np.asarray(fugacities)*1e-30/(KB*temperature)
    output.mkdir(parents=True,exist_ok=True)
    ref = np.atleast_1d(metadata['lnz'])
    dimension = len(guests)
    tables = []
    weights = []
    paths = sorted(folder.glob(replicas))
    assert len(paths)>=3
    for path in paths:
        data = np.loadtxt(path,delimiter=',',skiprows=1)
        counts = data[:,:dimension].astype(int)
        prob = softmax(data[:,dimension] + counts@(np.log(z)-ref))
        tables.append(counts)
        weights.append(prob)
    occupied_states = np.unique(np.concatenate(tables,axis=0),axis=0)
    indices = {tuple(counts):i for i,counts in enumerate(occupied_states)}
    ref_replicas = np.zeros((len(tables),len(indices)))
    for row, (counts,prob) in zip(ref_replicas,zip(tables,weights)):
        row[[indices[tuple(n)] for n in counts]] = prob
    np.savez_compressed(output/f'{case}_nist.npz', counts=occupied_states, probabilities=ref_replicas,
                        activities_A3_inverse=z)
    model = dict(schema='poremc-v1',temperature_K=temperature,cell_A=dims.tolist(),
                 species=guests,
                 host=dict(positions_A=atoms[:,:3].tolist(),sigma_A=atoms[:,3].tolist(),
                           epsilon_K=atoms[:,4].tolist(),charge_e=atoms[:,5].tolist(),mass_amu=mass),
                 potential=dict(kind='classical',cutoff_A=12.,coulomb='ewald',alpha=alpha,kmax=kmax,
                                shifted=shift,algorithm='cached',backend='native'),
                 reservoir=dict(fugacities_Pa=fugacities),
                 sampling=dict(driver='cpp',reciprocal_method='direct',reciprocal_backend='cpu',
                               ewald_k_cutoff=radial,trials=10,host_grid_mesh=32,
                               move_weights=dict(insert=.25,delete=.25,translate=.2,rotate=.1,regrow=.2)),
                 metadata=dict(nist_metadata=str(meta_path.relative_to(source)),nist_chemical_potentials=ref.tolist(),
                               nist_replicas=[f.name for f in paths],framework_site_types=labels,
                               preload_counts=preload,
                               precision_goals={'count_0':.25} if dimension==1 else
                               {'count_0':.15,'count_1':.40,'joint_probability':.025}))
    if dimension==2:
        model['metadata']['joint_threshold']=[2,13]
    (output/f'{case}.json').write_text(json.dumps(model,indent=2)+'\n')
    means = ref_replicas@occupied_states
    print(case,'nist means per supercell:',means.tolist(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('nist_repository',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('cases',nargs='+',choices=list(CASES))
    args=parser.parse_args()
    for case in args.cases:
        prepare(args.nist_repository,args.output,case)
