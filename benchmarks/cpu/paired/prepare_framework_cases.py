"""Set common CPU benchmarks from the previously audited framework inputs."""

import argparse
import copy
import math
from pathlib import Path
from shutil import copyfile

from scipy.integrate import quad

from benchmark_precision import read_model, write_json


def prepare(previous, out):
    out.mkdir(parents=True,exist_ok=True)
    for level,fugacity,loaded,tolerance in (
        ('low',.02,10,.20), ('mid',.5,65,.65), ('high',40.,140,1.20)
    ):
        model=read_model(previous/'nitpt_pxylene_0p1Pa.json')
        model['description']=(
            f'Charged 12-site p-xylene in the Ni_TPT_TATBA 2x2x2 supercell, '
            f'with a reservoir fugacity of {fugacity:g} Pa.'
        )
        model['reservoir']={'fugacities_Pa':[fugacity]}
        model.setdefault('metadata',{}).update(preload_counts=[loaded],precision_goals={'count_0':tolerance})
        if level in ('mid','high'):
            model['metadata']['raspa_cycle_cap']={'mid':1600,'high':800}[level]
        model['sampling']['move_weights']={'insert':.25,'delete':.25,'translate':.2,'rotate':.1,'regrow':.2}
        write_json(out/f'xylene_{level}.json',model)
    model=read_model(previous/'flexible_butane_adsorption.json')
    model['metadata'].update(preload_counts=[16],precision_goals={'count_0':.35,'trans_fraction':.015},
                             observe_trans=True)
    model['sampling']['partial_regrowth']=False
    write_json(out/'ton_butane.json',model)
    model['sampling']['partial_regrowth']=True
    write_json(out/'ton_butane_suffix.json',model)
    copyfile(previous/'TON.cif',out/'TON.cif')

    # Match the standard TraPPE-UA bending term in both engines. The angular
    # interval omits only tails more than eight thermal widths from 114 degrees.
    model['sampling']['partial_regrowth']=False
    torsion_log_q=model['species'][0]['log_q_internal']
    low,theta0,high=map(math.radians,(80.,114.,148.))
    k=62500.
    temp=model['temperature_K']
    z=quad(lambda theta:math.sin(theta)*math.exp(-k*(theta-theta0)**2/(2*temp)),low,high)[0]
    q_angle=z/(math.cos(low)-math.cos(high))
    model['species'][0].update(angle_ranges=[[0.,0.],[0.,0.],[low,high],[low,high]],
                                angle_force_constants=[0.,0.,k,k],
                                log_q_internal=torsion_log_q+2*math.log(q_angle))
    model['metadata'].pop('fixed_angles_deg')
    model['metadata'].update(bend_K_rad2=k,bend_center_deg=114.,
                             angular_support_deg=[80.,148.],raspa_q_reference=math.exp(torsion_log_q),
                             conditional_angle_partition=q_angle)
    write_json(out/'ton_butane_harmonic.json',model)
    model['sampling']['partial_regrowth']=True
    write_json(out/'ton_butane_harmonic_suffix.json',model)

    dilute=copy.deepcopy(model)
    dilute['reservoir']={'fugacities_Pa':[2.]}
    dilute['metadata'].update(preload_counts=[6],
                              precision_goals={'count_0':.15,'trans_fraction':.025})
    write_json(out/'ton_butane_dilute_suffix.json',dilute)
    dilute['sampling']['partial_regrowth']=False
    write_json(out/'ton_butane_dilute.json',dilute)

    gas=copy.deepcopy(model)
    gas['metadata'].update(preload_counts=[5],precision_goals={'count_0':.12,'trans_fraction':.025})
    gas['cell_A']=[32.,32.,32.]
    gas['host']={'positions_A':[[0.,0.,0.]],'sigma_A':[0.],'epsilon_K':[0.],'mass_amu':12.0107}
    gas['species'][0]['epsilon']=[0.,0.,0.,0.]
    gas['reservoir']={'fugacities_Pa':[300000.]}
    gas['sampling']['partial_regrowth']=False
    write_json(out/'butane_harmonic_ideal.json',gas)

    model=read_model(previous/'irmof_mixture.json')
    model['metadata'].update(preload_counts=[7,8],precision_goals={'count_0':.20,'count_1':.20,
                             'joint_probability':.025},joint_threshold=[3,4])
    model['metadata'].pop('source_raspa3_input',None)
    model['sampling'].update(trials=10,move_weights={'insert':.25,'delete':.25,'translate':.2,'rotate':.1,'regrow':.2})
    write_json(out/'irmof_mixture.json',model)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('previous_inputs',type=Path)
    p.add_argument('output_inputs',type=Path)
    a=p.parse_args()
    prepare(a.previous_inputs,a.output_inputs)
