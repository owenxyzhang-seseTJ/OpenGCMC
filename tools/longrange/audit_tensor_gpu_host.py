#!/usr/bin/env python3
"""Host-only audit of FP64 tensor mathematics and selected-move plug-in integration."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.longrange.validate_tensor_gpu import primitive_checks,fixture_engine
from tools.longrange.fixtures import workload_fixture


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('tensor_host_validation.json'))
    args=p.parse_args()
    with tempfile.TemporaryDirectory() as temp:
        plugin=Path(temp)/'cpu_tensor_oracle.so'
        subprocess.run(['c++','-std=c++17','-O2','-ffp-contract=off','-shared','-fPIC','-I',str(ROOT/'native'),
                        str(ROOT/'tests/tensor_host_plugin.cpp'),'-o',str(plugin)],check=True)
        primitive=primitive_checks(plugin)
        kmax=3;host_atoms=80
        _,_,initial=workload_fixture(host_atoms=host_atoms,kmax=kmax)
        with fixture_engine('cuda-tensor','delayed',plugin,541,kmax,host_atoms) as sim, \
             fixture_engine('cpu','delayed',plugin,541,kmax,host_atoms) as cpu:
            trial=np.array([initial[0]+[.04*i,-.11,.02] for i in range(8)])
            exclusion_error=float(np.max(abs(sim.trial_molecule_energies(trial,exclude=1)-cpu.trial_molecule_energies(trial,exclude=1))))
            sim.run(240,sample_every=60,record_energy=False)
            final_error=abs(sim.total_energy()-sim.potential.total_energy(sim.entries()))
            profile=sim.long_range_profile
            if final_error>2e-7 or exclusion_error>2e-7 or profile['gpu_selected_structures']==0:
                raise AssertionError((final_error,exclusion_error,profile))
        # Count allocations in the actual kernels for one B=10/A=12/k=13 half-spectrum call.
        # Shared accepted-state buffers and output energies are the same for both paths.
        half_modes=(27**3-1)//2
        tiles=13*27+13+1
        old_bytes=10*half_modes*3*8
        new_bytes=10*12*81*16+10*tiles*8
        evidence=dict(scope='CPU ABI test double sharing the CUDA lattice plan and arithmetic; no GPU kernels executed or timed.',
            gpu_executed=False,primitive=primitive,normal_entry=dict(excluded_probe_error_K=exclusion_error,
            final_rebuilt_energy_error_K=final_error,attempts=240,synthetic_host_atoms=host_atoms,kmax=kmax,
            test_double_batches=profile['gpu_batches'],test_double_selected_structures=profile['gpu_selected_structures']),
            workspace_model=dict(B=10,A=12,charged_sites=8,K=half_modes,mode_tiles=tiles,
            direct_energy_intermediate_bytes=old_bytes,tensor_energy_intermediate_bytes=new_bytes,
            intermediate_reduction_factor=old_bytes/new_bytes,axis_sincos_calls=3*10*12,
            direct_sincos_calls=10*8*half_modes,selected_return_bytes=2*8*half_modes,
            exclusions='Candidate coordinates, immutable metadata, base structure/weights, output scores and allocator overhead are outside the two intermediate counts; this is a storage/work model, not measured speed.'),
            real_device_runner='python tools/longrange/validate_tensor_gpu.py --input real.json --output tensor_device_results.json')
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(evidence,indent=2)+'\n')
        print(json.dumps(evidence,indent=2))

if __name__=='__main__':main()
