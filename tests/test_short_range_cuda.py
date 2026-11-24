"""Check native CUDA against the CPU pair path and its accepted guest state."""
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from poremc.fast import FastGCMC
from poremc.potentials import ClassicalPotential
from poremc.topology import Molecule


@pytest.fixture(scope="module")
def cuda_plugin():
    extension = ".dylib" if sys.platform == "darwin" else ".so"
    library = Path(__file__).resolve().parents[1] / "poremc" / f"libporemc_ewald_cuda{extension}"
    if not library.is_file() or not shutil.which("nvidia-smi"):
        pytest.skip("requires the built native plug-in and an NVIDIA device")
    if subprocess.run(["nvidia-smi", "-L"], capture_output=True, check=False).returncode:
        pytest.skip("no working NVIDIA device")
    return library


def fixture(coulomb, shifted):
    # Keep the cell skewed so three-axis wrapping and the full metric are used.
    cell = np.array([[28., 0., 0.], [3., 27., 0.], [-4., 2., 29.]])
    species = [
        Molecule("dipole", [2.49, 2.2], [15., 11.], charge=[.25, -.25],
                 rigid_positions=[[0., 0., 0.], [1.1, 0., 0.]]),
        Molecule("neutral", [2.9], [20.], charge=[0.]),
    ]
    rng = np.random.default_rng(2219)
    positions = rng.uniform(0., 1., (120, 3)) @ cell
    potential = ClassicalPotential(
        species, cell, 8., coulomb=coulomb, alpha=.25, shifted=shifted, algorithm="cached",
        kmax=3, host_positions=positions, host_sigma=np.full(len(positions), 2.3),
        host_epsilon=np.full(len(positions), 8.), host_charge=np.tile([.09, -.09], len(positions)//2),
    )
    body = np.array([[0., 0., 0.], [1.1, 0., 0.]])
    return species, potential, body


def test_requested_cuda_with_missing_library_fails(tmp_path):
    species, potential, _ = fixture("dsf", True)
    with pytest.raises(RuntimeError, match="CUDA|missing.so"):
        FastGCMC(species, potential, potential.cell, 400., [1e-4, 1e-4],
                 short_range_backend="cuda", cuda_library=tmp_path / "missing.so")


@pytest.mark.parametrize(("coulomb", "shifted", "guest_update", "mode"), [
    ("ewald", "force", "incremental", "delayed"),
    ("dsf", True, "rebuild", "exact"),
    ("none", False, "incremental", "exact"),
])
def test_device_pairs_across_guest_changes(cuda_plugin, coulomb, shifted, guest_update, mode):
    species, potential, body = fixture(coulomb, shifted)
    kwargs = {"seed": 42, "trials": 8, "guest_update": guest_update, "sampling_mode": mode,
              "host_grid_mesh": 8}
    args = [species, potential, potential.cell, 400., [2e-4, 1e-4]]
    starts = [(0, body+[3., 4., 8.]), (1, np.array([[8., 5., 9.]]))]
    trials = np.array([body+[3., 4., 8.], body+[9., 9., 9.], body+[12., 8., 7.],
                       body+[3., 4., 8.] + potential.cell[1]])
    with FastGCMC(*args, **kwargs) as cpu, FastGCMC(
        *args, **kwargs, short_range_backend="cuda", short_range_min_batch=4, cuda_library=cuda_plugin
    ) as gpu:
        for s, xyz in starts:
            for simulation in (cpu, gpu):
                simulation.add_particle(s, xyz)
        gpu.trial_molecule_energies(trials[:1], exclude=0, species=0)
        assert gpu.short_range_profile["framework_batches"] == 0
        expected = cpu.trial_molecule_energies(trials, exclude=0, species=0)
        np.testing.assert_allclose(gpu.trial_molecule_energies(trials, exclude=0, species=0),
                                   expected, rtol=5e-11, atol=5e-8)
        np.testing.assert_allclose(gpu.trial_molecule_energies(trials, species=0),
                                   cpu.trial_molecule_energies(trials, species=0), rtol=5e-11, atol=5e-8)
        assert np.isposinf(gpu.trial_molecule_energies(trials, species=0)[0])
        profile = gpu.short_range_profile
        assert profile["backend"] == "cuda" and profile["device"]
        assert profile["framework_batches"] == profile["guest_batches"] == 3
        assert profile["guest_uploads"] == 1
        if mode == "delayed":
            np.testing.assert_allclose(gpu.trial_surrogate_molecule_energies(trials, exclude=0),
                                       cpu.trial_surrogate_molecule_energies(trials, exclude=0),
                                       rtol=5e-11, atol=5e-8)
            assert gpu.short_range_profile["framework_batches"] == profile["framework_batches"]
            assert gpu.short_range_profile["guest_batches"] == profile["guest_batches"]+1

        production = gpu.run(150, sample_every=30, record_energy=False)
        assert sum(c["accepted"] for c in production["acceptance"].values()) > 0
        assert gpu.short_range_profile["guest_uploads"] > profile["guest_uploads"]
        with FastGCMC(*args, **kwargs) as new_cpu:
            for sid, xyz in gpu.entries():
                new_cpu.add_particle(sid, xyz)
            np.testing.assert_allclose(gpu.trial_molecule_energies(trials, species=0),
                                       new_cpu.trial_molecule_energies(trials, species=0),
                                       rtol=5e-11, atol=5e-8)
            np.testing.assert_allclose(gpu.total_energy(), new_cpu.total_energy(),
                                       rtol=5e-11, atol=5e-8)
        with FastGCMC.from_checkpoint(gpu.checkpoint(), potential) as restarted:
            assert restarted.short_range_profile["backend"] == "cuda"
            assert restarted.short_range_profile["min_batch"] == 4
            np.testing.assert_allclose(restarted.trial_molecule_energies(trials, species=0),
                                       gpu.trial_molecule_energies(trials, species=0),
                                       rtol=5e-11, atol=5e-8)
