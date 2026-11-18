"""Use saved short-window offsets ONLY for the same-fixture pipeline example."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
cfg=json.loads((ROOT/'examples/electrochemistry/hydrated_interface.json').read_text())
acid=json.loads((ROOT/'benchmarks/runs/electrochemistry/hydrated_acid_calibration/calibration.json').read_text())
redox=json.loads((ROOT/'benchmarks/runs/electrochemistry/hydrated_redox_calibration/calibration.json').read_text())
cfg['templates'][3]['reference_deprotonation_kelvin']=-acid['free_energy_kelvin'][-1]
cfg['templates'][4]['reference_reduction_kelvin']=redox['free_energy_kelvin'][-1]
cfg['description']='Pipeline demonstration using same-fixture short alchemical reference windows. Offsets are not converged experimental pKa/E0 calibrations. Full solvent GCMC remains uncalibrated in bulk activity.'
cfg['run']['steps']=3000;cfg['run']['burnin']=750
(ROOT/'benchmarks/runs/electrochemistry/hydrated_calibrated_pipeline.json').write_text(json.dumps(cfg,indent=2))
print('Updated same-fixture demonstration offsets; not a material calibration.')
