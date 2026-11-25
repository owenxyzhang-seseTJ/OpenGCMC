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


def test_zero_charge_trials_and_batched_translations(cuda_plugin):
    species, potential, body = fixture("ewald", True)
    args = (species, potential, potential.cell, 400., [2e-4, 1e-4])
    moves = dict(insert=0., delete=0., translate=1., rotate=0., regrow=0.)
    options = dict(seed=77, trials=8, move_weights=moves, guest_update="incremental")
    starts = [(0, body+[3., 4., 8.]), (1, np.array([[8., 5., 9.]]))]
    neutral_trials = np.array([[[8., 5., 9.]], [[11., 6., 4.]]])
    charged_trials = np.array([body+[3., 4., 8.], body+[9., 9., 9.]])
    with FastGCMC(*args, **options) as cpu, FastGCMC(
        *args, **options, short_range_backend="cuda", short_range_min_batch=1,
        reciprocal_backend="cuda-tensor", cuda_library=cuda_plugin
    ) as gpu:
        for sid, xyz in starts:
            cpu.add_particle(sid, xyz)
            gpu.add_particle(sid, xyz)
        initial_reciprocal = gpu.long_range_profile["gpu_batches"]
        np.testing.assert_allclose(
            gpu.trial_molecule_energies(neutral_trials, exclude=1, species=1),
            cpu.trial_molecule_energies(neutral_trials, exclude=1, species=1),
            rtol=5e-11, atol=5e-8,
        )
        assert gpu.long_range_profile["gpu_batches"] == initial_reciprocal

        # Zero net charge is distinct from zero partial charges on every site.
        np.testing.assert_allclose(
            gpu.trial_molecule_energies(charged_trials, exclude=0, species=0),
            cpu.trial_molecule_energies(charged_trials, exclude=0, species=0),
            rtol=5e-11, atol=5e-8,
        )
        assert gpu.long_range_profile["gpu_batches"] == initial_reciprocal+1

        initial_short = gpu.short_range_profile["framework_batches"]
        steps = 80
        gpu.run(steps, sample_every=8, record_energy=False)
        cpu.run(steps, sample_every=8, record_energy=False)
        assert gpu.short_range_profile["framework_batches"]-initial_short == steps
        assert 0 < gpu.long_range_profile["gpu_batches"]-initial_reciprocal-1 < steps
        assert gpu.counters == cpu.counters
        for (cpu_sid, cpu_xyz), (gpu_sid, gpu_xyz) in zip(cpu.entries(), gpu.entries(), strict=True):
            assert cpu_sid == gpu_sid
            np.testing.assert_array_equal(cpu_xyz, gpu_xyz)
        np.testing.assert_allclose(gpu.total_energy(), cpu.total_energy(), rtol=5e-11, atol=5e-8)


@pytest.mark.parametrize(("steep", "cap"), [
    ("linear", 1e5), ("boltzmann", 1e5), ("boltzmann", 3e5), ("linear", 0.),
])
def test_cuda_reference_grid_matches_cpu(cuda_plugin, steep, cap):
    species, potential, body = fixture("ewald", True)
    args = (species, potential, potential.cell, 400., [2e-4, 1e-4])
    options = dict(sampling_mode="delayed", trials=8, host_grid_mesh=(8, 16, 8),
                   host_grid_cap=cap, host_grid_steep=steep, seed=49, guest_update="incremental")
    shifts = np.random.default_rng(380).uniform(0.0, 1.0, (48, 3)) @ potential.cell
    trials = body[None, :, :] + shifts[:, None, :]
    trials[0] = body + potential.cell[1]
    with FastGCMC(*args, **options) as cpu, FastGCMC(
        *args, **options, short_range_backend="cuda", short_range_min_batch=4, cuda_library=cuda_plugin,
    ) as gpu:
        for sim in (cpu, gpu):
            sim.add_particle(0, body+[3., 4., 8.])
            sim.add_particle(1, [[9., 6., 7.]])
        np.testing.assert_allclose(
            gpu.trial_surrogate_molecule_energies(trials, exclude=0, species=0),
            cpu.trial_surrogate_molecule_energies(trials, exclude=0, species=0),
            atol=5e-8, rtol=5e-11,
        )
        np.testing.assert_allclose(
            gpu.trial_surrogate_site_energies(trials[0, :1], trials[:, 1], exclude=0, species=0),
            cpu.trial_surrogate_site_energies(trials[0, :1], trials[:, 1], exclude=0, species=0),
            atol=5e-8, rtol=5e-11,
        )
        profile = gpu.short_range_profile
        assert profile["reference_grid_batches"] == (2 if cap else 0)
        assert profile["reference_grid_site_lookups"] == (3*len(trials) if cap else 0)
        a = cpu.run(300, sample_every=10)
        b = gpu.run(300, sample_every=10)
        assert a["counts"] == b["counts"] and cpu.counters == gpu.counters
        np.testing.assert_allclose(a["energies_kelvin"], b["energies_kelvin"], atol=5e-8, rtol=5e-11)


def test_cuda_guest_uploads_only_the_translated_site(cuda_plugin):
    all_species, fixture_potential, _ = fixture("none", False)
    species = [all_species[1]]
    cell = fixture_potential.cell
    potential = ClassicalPotential(species, cell, 8., coulomb="none")
    moves = dict(insert=0., delete=0., translate=1., rotate=0., regrow=0.)
    with FastGCMC(species, potential, cell, 400., [1e-4], seed=474, move_weights=moves,
                  short_range_backend="cuda", short_range_min_batch=1, cuda_library=cuda_plugin) as gpu:
        for xyz in ([[3., 3., 3.]], [[11., 11., 11.]], [[18., 18., 18.]]):
            gpu.add_particle(0, xyz)
        gpu.trial_molecule_energies([[[8., 11., 19.]]])
        before = gpu.short_range_profile["guest_upload_bytes"]
        gpu.run(60, record_energy=False)
        gpu.trial_molecule_energies([[[8., 11., 19.]]])
        accepted = gpu.counters["translate"]["accepted"]
        assert accepted > 0
        assert gpu.short_range_profile["guest_upload_bytes"]-before == 48*accepted


@pytest.mark.parametrize(("mode", "refresh"), [("exact", 1), ("exact", 5), ("delayed", 5), ("reference-mh", 5)])
def test_cuda_pme_preserves_mesh_energy_and_restart(cuda_plugin, mode, refresh):
    species, potential, body = fixture("ewald", True)
    args = (species, potential, potential.cell, 400., [2e-4, 1e-4])
    options = dict(sampling_mode=mode, reciprocal_method="pme", pme_mesh=(8, 16, 8),
                   pme_refresh=refresh, trials=6, host_grid_mesh=8, seed=173, guest_update="incremental")
    trials = body[None, :, :]+np.array([[[3., 4., 8.]], [[9., 9., 9.]], [[12., 8., 7.]],
                                        [[3., 4., 8.]]+potential.cell[1]])
    with FastGCMC(*args, **options) as cpu, FastGCMC(
        *args, **options, reciprocal_backend="cuda-pme", cuda_library=cuda_plugin,
    ) as gpu:
        for sim in (cpu, gpu):
            sim.add_particle(0, trials[0])
            sim.add_particle(1, [[8., 5., 9.]])
        if refresh>1:
            assert gpu.long_range_profile["pme_pending_nodes"]>0
        np.testing.assert_allclose(gpu.total_energy(), cpu.total_energy(), atol=5e-8, rtol=5e-11)
        np.testing.assert_allclose(gpu.trial_molecule_energies(trials, exclude=0, species=0),
                                   cpu.trial_molecule_energies(trials, exclude=0, species=0), atol=5e-8, rtol=5e-11)
        np.testing.assert_allclose(gpu.trial_site_energies(trials[0, :1], trials[:, 1], exclude=0, species=0),
                                   cpu.trial_site_energies(trials[0, :1], trials[:, 1], exclude=0, species=0),
                                   atol=5e-8, rtol=5e-11)
        assert gpu.long_range_profile["gpu_batches"]>=2
        assert gpu.long_range_profile["gpu_fft_transforms"]>=2
        checkpoint = gpu.checkpoint()
        a = cpu.run(180, sample_every=10)
        b = gpu.run(180, sample_every=10)
        assert a["counts"] == b["counts"] and cpu.counters == gpu.counters
        np.testing.assert_allclose(a["energies_kelvin"], b["energies_kelvin"], atol=5e-8, rtol=5e-11)
        with FastGCMC.from_checkpoint(checkpoint, potential) as resumed:
            c = resumed.run(180, sample_every=10)
        assert b["counts"] == c["counts"]
        np.testing.assert_array_equal(b["energies_kelvin"], c["energies_kelvin"])
