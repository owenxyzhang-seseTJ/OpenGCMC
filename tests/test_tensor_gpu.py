"""Exercise the real Ewald native selection/commit with a CPU-only test plug-in."""
import subprocess
from pathlib import Path
import numpy as np
import pytest
from tools.longrange.validate_tensor_gpu import primitive_checks,fixture_engine,ROOT


@pytest.fixture(scope='module')
def host_plugin(tmp_path_factory):
    plugin=tmp_path_factory.mktemp('tensor_host_oracle')/'host_oracle.so'
    subprocess.run(['c++','-std=c++17','-O2','-ffp-contract=off','-shared','-fPIC',
                    '-I',str(ROOT/'native'),str(ROOT/'tests/tensor_host_plugin.cpp'),'-o',str(plugin)],check=True)
    return plugin


def test_tensor_primitive_against_direct_exponentials(host_plugin):
    record=primitive_checks(host_plugin)
    assert 'CPU ABI test double' in record['device']
    assert len(record['cases'])==3


def test_tensor_native_selected_commit_and_exact_probe(host_plugin):
    from tools.longrange.fixtures import workload_fixture
    _,_,initial=workload_fixture(host_atoms=80,kmax=3)
    configs=dict(kmax=3,host_atoms=80)
    with fixture_engine('cuda-tensor','delayed',host_plugin,541,**configs) as sim, \
         fixture_engine('cpu','delayed',host_plugin,541,**configs) as cpu:
        trial=np.array([initial[0]+[.04*i,-.11,.02] for i in range(8)])
        np.testing.assert_allclose(sim.trial_molecule_energies(trial,exclude=1),cpu.trial_molecule_energies(trial,exclude=1),atol=1e-8,rtol=2e-12)
        assert sim.long_range_profile['gpu_batches']>0
        assert sim.long_range_profile['gpu_selected_structures']==0
        sim.run(240,burnin=0,sample_every=60,record_energy=False)
        assert sim.long_range_profile['gpu_selected_structures']>0
        assert sim.acceleration_profile['selected_correction_backend']=='cuda-tensor-sf/cpu-residual'
        assert 'CPU ABI test double' in sim.long_range_profile['device']
        assert abs(sim.total_energy()-sim.potential.total_energy(sim.entries()))<2e-7


def test_required_tensor_never_mislabels_cpu(tmp_path):
    with pytest.raises(RuntimeError,match='required CUDA reciprocal backend unavailable'):
        fixture_engine('cuda-tensor','delayed',tmp_path/'missing.so',19,kmax=2,host_atoms=6)
