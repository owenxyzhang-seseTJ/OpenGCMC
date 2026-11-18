"""Deliberately explicit migration of the RASPA3 rigid-LJ subset.

Input syntax is not a simulation specification by itself. Unsupported ensemble,
interactions, or molecular flexibility must not be silently discarded. This
adapter embeds the expanded framework, so the new JSON is self-contained.
"""
import json
import math
from pathlib import Path
from .io import read_cif


def _check_fields(data,allowed,context):
    unknown=set(data)-set(allowed)
    if unknown: raise ValueError(f'{context}: unsupported fields {sorted(unknown)}')


def import_raspa3(path,steps,burnin=1000):
    path=Path(path).resolve(); base=path.parent; inp=json.loads(path.read_text())
    _check_fields(inp,['SimulationType','NumberOfProductionCycles','NumberOfInitializationCycles','PrintEvery',
                      'Systems','Components','ForceField','NumberOfEquilibrationCycles'],'simulation')
    if inp.get('SimulationType','MonteCarlo')!='MonteCarlo': raise ValueError('only MonteCarlo GCMC input can be converted')
    systems=inp.get('Systems',[])
    if len(systems)!=1: raise ValueError('convert exactly one RASPA3 system at a time')
    system=systems[0]
    _check_fields(system,['Type','Name','NumberOfUnitCells','ChargeMethod','ExternalTemperature','ExternalPressure',
                          'UseChargesFrom','HeliumVoidFraction','BoxLengths','BoxAngles'],'system')
    fp=base/'force_field.json'
    ff=json.loads(fp.read_text())
    _check_fields(ff,['MixingRule','TruncationMethod','TailCorrections','CutOffVDW','CutOffCoulomb','PseudoAtoms',
                      'SelfInteractions','BinaryInteractions','ChargeMethod','EwaldPrecision'],'force field')
    if ff.get('MixingRule','Lorentz-Berthelot')!='Lorentz-Berthelot': raise ValueError('only Lorentz-Berthelot mixing is supported')
    if ff.get('TailCorrections',False): raise ValueError('tail correction must be specified through a compatible implemented model; conversion refuses to drop it')
    if ff.get('BinaryInteractions'): raise ValueError('custom unlike-pair overrides are not supported by this converter')
    cutoff=float(ff.get('CutOffVDW',12))
    if float(ff.get('CutOffCoulomb',cutoff))!=cutoff: raise ValueError('different LJ and charge cutoffs need manual migration')
    truncate=ff.get('TruncationMethod','shifted').lower()
    if truncate not in ('shifted','truncated'): raise ValueError('unsupported truncation method')
    mapping={x['name']:{'mass':x.get('mass'),'charge':x.get('charge',0)} for x in ff['PseudoAtoms']}
    for interaction in ff['SelfInteractions']:
        if interaction['type'].lower()!='lennard-jones': raise ValueError('only LJ self-interactions can be converted')
        epsilon,sigma=interaction['parameters']
        mapping[interaction['name']].update({'sigma':sigma,'epsilon':epsilon})
    if any('sigma' not in p or 'epsilon' not in p for p in mapping.values()): raise ValueError('missing pseudoatom LJ parameters')
    for v in mapping.values():
        if v['mass'] is None: v.pop('mass')
    molecules=[]; fractions=[]; coefficients=[]; moves=[]; notes=[]
    components=inp['Components']
    for component in components:
        _check_fields(component,['Name','FugacityCoefficient','IdealGasRosenbluthWeight','TranslationProbability',
            'RotationProbability','ReinsertionProbability','SwapProbability','WidomProbability','CreateNumberOfMolecules',
            'MolFraction','MoleFraction'],'component')
        if component.get('CreateNumberOfMolecules',0)!=0: raise ValueError('initial coordinates must be supplied explicitly; converter starts empty')
        name=component['Name']; mol=json.loads((base/(name+'.json')).read_text())
        _check_fields(mol,['CriticalTemperature','CriticalPressure','AcentricFactor','pseudoAtoms','Type'],'molecule '+name)
        if mol.get('Type','rigid').lower()!='rigid': raise ValueError('flexible RASPA3 molecule topology requires explicit PoreMC migration')
        if not math.isclose(float(component.get('IdealGasRosenbluthWeight',1)),1):
            raise ValueError('non-unit rigid ideal Rosenbluth normalization requires explicit energy-reference migration')
        atomtypes=[a[0] for a in mol['pseudoAtoms']]; coords=[a[1] for a in mol['pseudoAtoms']]
        molecules.append({'name':name,'sigma':[mapping[a]['sigma'] for a in atomtypes],
                'epsilon':[mapping[a]['epsilon'] for a in atomtypes],'charge':[mapping[a]['charge'] for a in atomtypes],
                'rigid_positions':coords,'molar_mass':sum(mapping[a].get('mass',0) for a in atomtypes)})
        if len(components)>1 and not ('MolFraction' in component or 'MoleFraction' in component):
            raise ValueError('mixture components need explicit mole fractions')
        fractions.append(component.get('MolFraction',component.get('MoleFraction',1.0)))
        if 'FugacityCoefficient' not in component: raise ValueError('provide explicit fugacity coefficient; equation-of-state choice must not be guessed')
        coefficients.append(component['FugacityCoefficient'])
        swap=float(component.get('SwapProbability',0))
        moves.append({'insert':swap/2,'delete':swap/2,'translate':component.get('TranslationProbability',0),
                      'rotate':component.get('RotationProbability',0),'regrow':component.get('ReinsertionProbability',0)})
        if component.get('WidomProbability',0): notes.append(f'{name}: Widom observations are requested separately with the Python observable API; they are not MC state transitions')
    if any(m!=moves[0] for m in moves[1:]): raise ValueError('component-specific move schedules need manual migration')
    if moves[0]['insert']<=0: raise ValueError('GCMC converter requires a nonzero SwapProbability')
    charge=system.get('ChargeMethod',ff.get('ChargeMethod','Ewald')).lower()
    if charge not in ('ewald','none'): raise ValueError('unsupported charge method')
    converted={'schema':'poremc-v1','description':'Explicit migration of supported RASPA3 rigid LJ GCMC subset',
        'temperature_K':system['ExternalTemperature'],'species':molecules,
        'potential':{'kind':'classical','cutoff_A':cutoff,'coulomb':charge,'shifted':truncate=='shifted','backend':'native'},
        'reservoir':{'pressure_Pa':system['ExternalPressure'],'mole_fractions':fractions,'fugacity_coefficients':coefficients},
        'sampling':{'steps':int(steps),'burnin':int(burnin),'sample_every':10,'seed':2026,'trials':8,'move_weights':moves[0]},
        'metadata':{'source_raspa3_input':str(path),'conversion_notes':notes+[
            'Explicit attempted-move count replaces RASPA cycle count; cycles were not assumed to equal moves.',
            'Ewald alpha/kmax are PoreMC defaults; converge and match both codes before comparison.',
            'No RASPA restart, output, transport observables or undocumented fields were imported.']}}
    if system.get('Type')=='Framework':
        use=system.get('UseChargesFrom','PseudoAtoms')
        if use not in ('PseudoAtoms','CIF_File'): raise ValueError('charge equilibration is not available in this converter')
        fw=read_cif(base/(system['Name']+'.cif'),mapping,system.get('NumberOfUnitCells',[1,1,1]),cutoff,
                    'cif' if use=='CIF_File' else 'force_field')
        converted['cell_A']=fw.cell.tolist()
        converted['host']={'positions_A':fw.positions.tolist(),'sigma_A':fw.sigma.tolist(),
                           'epsilon_K':fw.epsilon.tolist(),'charge_e':fw.charge.tolist(),'mass_amu':fw.mass_amu}
    elif system.get('Type')=='Box':
        from .cell import cell_from_parameters
        converted['cell_A']=cell_from_parameters(*system['BoxLengths'],*system.get('BoxAngles',[90,90,90])).tolist()
    else: raise ValueError('unsupported RASPA system type')
    return converted
