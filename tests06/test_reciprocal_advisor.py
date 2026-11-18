"""Small independent configuration checks for the discarded-mode inequality."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from poremc.electrostatics import Ewald
from poremc.units import COULOMB_KA

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("advise_reciprocal", ROOT/"tools06/advise_reciprocal.py")
advisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(advisor)


class ReciprocalAdvisor(unittest.TestCase):
    def test_random_complete_move_bounds_and_shell_optimum(self):
        rng = np.random.default_rng(8022)
        cell = np.array([[24.,0,0],[2.,26.,0],[-1.,3.,28.]])
        ewald = Ewald(cell, 8., .3, (4,4,4))
        host_q = rng.normal(0., .1, 7)
        host_xyz = rng.random((7,3))@cell
        sh = advisor.host_structure(ewald, host_xyz, host_q, block_atoms=3)
        expected = (host_q[:,None]*np.exp(1j*(host_xyz@ewald.k.T))).sum(axis=0)
        np.testing.assert_allclose(sh, expected, atol=2e-14, rtol=2e-14)
        q = np.array([.12,-.07,-.05]);qabs = np.abs(q).sum();nmax = 3
        advice = advisor.advise(ewald, sh, qabs, nmax, 20., np.abs(host_q).sum(), len(host_q))
        cutoff = advice["recommendation"]["sampling"]["ewald_k_cutoff"]
        self.assertGreater(cutoff, 0.)
        discarded = np.linalg.norm(ewald.k, axis=1)>cutoff
        self.assertTrue(discarded.any())
        coefficient = COULOMB_KA*ewald.coefficient[discarded]
        bins = advice["bounds"]["insertion_or_deletion_increment_K"]
        brep = advice["bounds"]["replacement_increment_K"]
        self.assertLessEqual(brep, 20.)
        # Direct complex exponentials independently evaluate only omitted terms.
        def sf(x, charges):
            return (charges[:,None]*np.exp(1j*(x@ewald.k[discarded].T))).sum(axis=0)
        for n in range(nmax+1):
            for repeat in range(8):
                bath_xyz = rng.random((n*len(q),3))@cell
                bath_q = np.tile(q,n)
                base = expected[discarded]+sf(bath_xyz,bath_q)
                new = sf(rng.random((len(q),3))@cell,q)
                old = sf(rng.random((len(q),3))@cell,q)
                de_new = np.sum(coefficient*(2*np.real(base.conj()*new)+np.abs(new)**2))
                de_old = np.sum(coefficient*(2*np.real(base.conj()*old)+np.abs(old)**2))
                self.assertLessEqual(abs(de_new), bins+1e-10)
                self.assertLessEqual(abs(de_new-de_old), brep+1e-10)
        exact = advisor.advise(ewald, sh, qabs, nmax, 0.)
        self.assertEqual(exact["recommendation"]["sampling"]["ewald_k_cutoff"], 0.)
        self.assertFalse(exact["changes_finite_hamiltonian"])
        self.assertEqual(exact["bounds"]["replacement_increment_K"], 0.)

    def test_json_geometry_loader_does_not_construct_or_modify_sampler(self):
        from poremc.topology import Molecule
        molecule = Molecule("dipole", [2.,2.], [10.,10.], charge=[.2,-.2],
                            rigid_positions=[[0,0,0],[1.1,0,0]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"input.json"
            cfg = {"schema":"poremc-v1", "cell_A":[25.,25.,25.],
                   "species":[molecule.to_dict()],
                   "potential":{"kind":"classical","coulomb":"ewald","cutoff_A":8.,"kmax":[3,3,3]},
                   "sampling":{"driver":"cpp","sampling_mode":"delayed","host_grid_mesh":128}}
            raw = json.dumps(cfg);path.write_text(raw)
            ewald, xyz, q, molecular_q, provenance = advisor.load_input(path)
            self.assertEqual(path.read_text(),raw)
            self.assertEqual(len(ewald.k),342)
            self.assertEqual(xyz.shape,(0,3))
            self.assertEqual(len(q),0)
            np.testing.assert_array_equal(molecular_q,[.2,-.2])
            self.assertEqual(provenance["guest_sites"],2)


if __name__ == "__main__":
    unittest.main()
