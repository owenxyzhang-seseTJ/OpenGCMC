"""Native long-range API, CLI routing and execution-path regressions."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from poremc.cli import construct_case,run_configuration
from poremc.fast import FastGCMC
from poremc.potentials import ClassicalPotential
from poremc.topology import Molecule

def configuration(method='direct'):
    molecule=Molecule('dipole',[2.,2.],[10.,10.],charge=[.2,-.2],
                      rigid_positions=[[0.,0.,0.],[1.,0.,0.]])
    return {'schema':'poremc-v1','temperature_K':300.,'cell_A':[24.,24.,24.],
            'species':[molecule.to_dict()],
            'potential':{'kind':'classical','cutoff_A':7.,'coulomb':'ewald','kmax':3},
            'reservoir':{'activities_A3_inverse':[2./24**3]},
            'sampling':{'driver':'cpp','steps':100,'burnin':10,'sample_every':10,
                        'trials':4,'seed':71,'reciprocal_method':method,'pme_mesh':[16,16,16]}}

class NativeAPI(unittest.TestCase):
    def test_no_python_electrostatic_callback_inside_cpp_sampling(self):
        cell=np.eye(3)*24.
        molecule=Molecule('dipole',[2.,2.],[10.,10.],charge=[.2,-.2],
                          rigid_positions=[[0.,0.,0.],[1.,0.,0.]])
        potential=ClassicalPotential([molecule],cell,7.,coulomb='ewald',kmax=3,
                    host_positions=[[3.,4.,5.],[14.,15.,16.]],host_charge=[.1,-.1])
        self.assertIsNone(potential._host_snapshot)
        with patch.object(potential,'prepare_environment',side_effect=AssertionError('Python callback')), \
             patch.object(potential.ewald,'snapshot',side_effect=AssertionError('Python structure factor')), \
             patch.object(potential.ewald,'insertion_energies',side_effect=AssertionError('Python reciprocal')):
            with FastGCMC(molecule,potential,cell,300,[3./24**3],trials=4) as sim:
                sim.run(300,burnin=20,sample_every=30)
                self.assertEqual(sim.long_range_profile['host_builds'],1)
                self.assertGreater(sim.long_range_profile['candidate_batches'],0)
                self.assertTrue(np.isfinite(sim.total_energy()))
        self.assertIsNone(potential._host_snapshot)

    def test_cli_ewald_and_pme_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            for method in ('direct','pme'):
                config=configuration(method);out=Path(tmp)/method
                result=run_configuration(config,out)
                self.assertEqual(result['long_range']['method'],'ewald' if method=='direct' else 'pme')
                checkpoint=json.loads((out/'checkpoint.json').read_text())
                self.assertEqual(checkpoint['schema'],'poremc-fast-checkpoint-v3')
                continued=run_configuration(config,out/'continued',restart=out/'checkpoint.json')
                self.assertEqual(continued['metadata']['driver'],'cpp')
                self.assertTrue(np.isfinite(continued['energies_kelvin']).all())

    def test_auto_is_native_cpu_when_plugin_missing_and_required_cuda_errors(self):
        config=configuration();config['sampling'].update(reciprocal_backend='auto',cuda_library='/no/such/poremc_cuda.so')
        sim,*_=construct_case(config)
        with sim:
            self.assertIn('cpu',sim.long_range_profile['backend'])
            self.assertIn('CUDA unavailable',sim.long_range_profile['backend'])
            sim.run(10,sample_every=10)
        config['sampling']['reciprocal_backend']='cuda'
        with self.assertRaisesRegex(RuntimeError,'required CUDA'):
            construct_case(config)

    def test_pme_probe_reuses_fft_and_mesh_fingerprint_changes(self):
        config=configuration('pme')
        sim,*_=construct_case(config)
        with sim:
            sim.add_particle(0,[[6.,7.,8.],[7.,7.,8.]])
            before=sim.long_range_profile['fft_transforms']
            energy=sim.total_energy();rng=sim.checkpoint()['rng_state'];counters=sim.counters
            sim.trial_molecule_energies([[[9.,10.,11.],[10.,10.,11.]]],exclude=0)
            self.assertEqual(sim.long_range_profile['fft_transforms'],before)
            self.assertEqual(sim.total_energy(),energy)
            self.assertEqual(sim.checkpoint()['rng_state'],rng)
            self.assertEqual(sim.counters,counters)
            other=copy.deepcopy(config);other['sampling']['pme_mesh']=[32,32,32]
            changed,*_=construct_case(other)
            with changed:self.assertNotEqual(sim._fingerprint(),changed._fingerprint())

    def test_reference_host_backend_still_uses_complete_native_reciprocal(self):
        config=configuration()
        config['potential']['backend']='reference'
        sim,*_=construct_case(config)
        with sim:
            sim.add_particle(0,[[4.,5.,6.],[5.,5.,6.]])
            self.assertAlmostEqual(sim.total_energy(),sim.potential.total_energy(sim.entries()),delta=1e-8)

if __name__=='__main__':unittest.main()
