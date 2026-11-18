from pathlib import Path
import json,copy,math
ROOT=Path(__file__).resolve().parents[2]
def write(name, obj):
    (ROOT/'examples/electrochemistry'/name).write_text(json.dumps(obj,indent=2),encoding='utf8')
cell=[12.8,12.8,48.]
cfg={
 'description':'Integrated explicit-SPC/E, salt, constant-pH and redox verification model. X/Y/site parameters and activities are illustrative, not calibrated to a material.',
 'seed':20261117,'cell':cell,
 'templates':[{'kind':'spce'},{'kind':'ion','name':'X+','charge':1.,'sigma':2.8,'epsilon':50.},
              {'kind':'ion','name':'Y-','charge':-1.,'sigma':4.0,'epsilon':50.},
              {'kind':'acid','name':'surface_acid','pKa':7.,'sigma':3.5,'epsilon':50.,'reference_deprotonation_kelvin':0.},
              {'kind':'redox','name':'surface_redox','E0_V':0.,'sigma':3.5,'epsilon':50.,'reference_reduction_kelvin':0.}],
 'forcefield':{'cutoff':6.,'alpha':.35,'kmax':[4,4,14],'dielectric':1.,'coulomb':'ewald','shifted':False,'backend':'native','slab':True},
 'electrodes':{'plane_grid':{'shape':[3,3],'z_A':[16.,32.]},'plane_potentials_V':[-.05,.05],'width':1.},
 'region':{'lower':[0,0,18/48],'upper':[1,1,30/48]},
 'initial':[{'template':0,'grid':{'shape':[4,4,4],'lower_A':[0,0,18],'upper_A':[12.8,12.8,30],
                                 'skip_flat_indices':[21,42,25,38]}},
            {'template':1,'anchor_A':[4.8,4.8,22.5]},
            {'template':2,'anchor_A':[8.,8.,25.5]},
            {'template':3,'state':0,'anchor_A':[4.8,8.,22.5],'mobile':False,'exchangeable':False},
            {'template':4,'state':0,'anchor_A':[8.,4.8,25.5],'mobile':False,'exchangeable':False}],
 'reservoirs':{'temperature':298.15,'pH':7.,'electron_potential_V':-.05,'proton_reference_V':0.},
 'log_activities':[-14.,-12.,-12.,0.,0.],
 'exchanges':[{'name':'water','stoichiometry':[1,0,0,0,0],'trials':4,'weight':4.},
              {'name':'salt','stoichiometry':[0,1,1,0,0],'trials':1,'weight':1.}],
 'reactions':[{'name':'acid','template':3,'state_a':0,'state_b':1,'stages':6,'relaxations':2},
              {'name':'redox','template':4,'state_a':0,'state_b':1,'stages':6,'relaxations':2}],
 'moves':{'move_weights':[.8,.08,.12],'translation_A':.18,'rotation_rad':.25},
 'run':{'steps':2500,'burnin':750,'sample_every':10,'record_positions':True}}
write('hydrated_interface.json',cfg)
cal=copy.deepcopy(cfg)
cal['description']='Frozen-composition explicit-water redox free-energy windows for MBAR; use calibrated reference free energies before predicting physical E0/pKa.'
cal['moves']['move_weights']=[1,0,0]
cal['calibration']={'edge_index':1,'uid':63,'samples_per_window':35,'decorrelation_moves':6,'burnin_moves':100}
write('hydrated_redox_calibration.json',cal)
# Multisite constant-pH example with a fixed dummy proton: both endpoint
# geometries are identical. Charges/LJ are pedagogic, not a named force field.
acid={'kind':'custom','name':'carboxyl_demo','geometry':[[0,0,0],[1.25,0,0],[-.625,1.08253175,0],[2.2,0,0]],
      'states':[{'name':'A-','charges':[.6,-.8,-.8,0.],'sigma':[3.4,3.,3.,0.],'epsilon':[40.,80.,80.,0.]},
                {'name':'HA','charges':[.6,-.55,-.55,.5],'sigma':[3.4,3.,3.,0.],'epsilon':[40.,80.,80.,0.],
                 'bound_protons':1,'intrinsic_dimensionless':-math.log(10)*7}],
      'citation':'Illustrative dummy-proton site; replace with calibrated charges and reference free energy.'}
mult=copy.deepcopy(cfg)
mult['templates'][3]=acid
mult['initial'][3]['anchor_A']=[4.8,8.,22.5]
mult['run']={'steps':500,'burnin':100,'sample_every':10,'record_positions':True}
write('dummy_proton_titration.json',mult)
# Four-state PCET thermodynamic cycle; energies assigned to states, never
# inconsistent pKas independently around a closed loop.
pcet={
 'description':'Analytic four-state PCET cycle: state-energy construction ensures cycle closure. No explicit solvent in this minimal test.',
 'seed':31,'cell':[24,24,24],
 'templates':[{'kind':'custom','name':'PCET_site','geometry':[[0,0,0]],'states':[
  {'name':'Ox','charges':[1.],'sigma':[0.],'epsilon':[0.]},
  {'name':'OxH','charges':[2.],'sigma':[0.],'epsilon':[0.],'bound_protons':1,'intrinsic_dimensionless':-math.log(10)*7},
  {'name':'Red','charges':[0.],'sigma':[0.],'epsilon':[0.],'bound_electrons':1},
  {'name':'RedH','charges':[1.],'sigma':[0.],'epsilon':[0.],'bound_electrons':1,'bound_protons':1,'intrinsic_dimensionless':-math.log(10)*9}]}],
 'forcefield':{'cutoff':8.,'coulomb':'none'},
 'initial':[{'template':0,'state':0,'anchor_A':[12,12,12],'mobile':False,'exchangeable':False}],
 'reservoirs':{'temperature':300.,'pH':8.,'electron_potential_V':.04},
 'reactions':[{'template':0,'state_a':0,'state_b':1,'name':'Ox protonation'},
              {'template':0,'state_a':2,'state_b':3,'name':'Red protonation'},
              {'template':0,'state_a':0,'state_b':2,'name':'unprotonated ET'},
              {'template':0,'state_a':1,'state_b':3,'name':'protonated ET'},
              {'template':0,'state_a':0,'state_b':3,'name':'concerted PCET'}],
 'moves':{'move_weights':[.1,0,.9]},'run':{'steps':20000,'burnin':1000,'sample_every':2}}
write('pcet_four_state.json',pcet)
kin={
 'description':'Exactly solvable two-contact redox kinetic model. Rates are illustrative per-site values, not predictions for the hydration example.',
 'states':[{'name':'Ox'},{'name':'Red','bound_electrons':1}],
 'edges':[{'source':0,'target':1,'electrode':0,'name':'left_ET','model':'butler_volmer','k0_per_s':20.},
          {'source':0,'target':1,'electrode':1,'name':'right_ET','model':'butler_volmer','k0_per_s':20.}],
 'network':{'temperature':300.,'electrodes':2,'state_electrode_charge_e':[[0.,0.],[.5,.5]]},
 'task':'ssa','ssa':{'segments':[{'duration_s':200.,'potentials_V':[-.05,.05]}],'initial_state':0,'seed':101},
 'current_bins':[float(i) for i in range(201)]}
write('two_contact_current.json',kin)
cv=copy.deepcopy(kin);cv.pop('ssa');cv.pop('current_bins');cv['task']='ramp'
cv['network']['capacitance_F']=[[2e-20,-2e-20],[-2e-20,2e-20]]
cv['ramp']={'times_s':[0,.2,.4],'potentials_V':[[-.12,0],[.12,0],[-.12,0]],'samples':401,'site_count':1.}
write('voltammetry.json',cv)
print('Wrote six JSON examples')
