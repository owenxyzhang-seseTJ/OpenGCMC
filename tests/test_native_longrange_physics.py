"""Independent electrostatic checks for the native long-range sampler.

The oracle below does not use poremc.electrostatics or ClassicalPotential's
energy routines.  It rebuilds the complete periodic sum from coordinates,
then subtracts the empty framework and primary intramolecular exclusions.
"""
import itertools
import math
import ctypes as C

import numpy as np
import tempfile
import unittest
from pathlib import Path

from poremc.fast import FastGCMC
from poremc.potentials import ClassicalPotential
from poremc.topology import Molecule
from poremc.units import COULOMB_KA


def independent_mesh_reciprocal(positions, charges, cell, alpha, mesh):
    """Cubic charge assignment and a NumPy FFT, independently of native PME."""
    mesh = np.broadcast_to(np.asarray(mesh, int), (3,))
    rho = np.zeros(tuple(mesh))
    scaled = np.mod(np.asarray(positions) @ np.linalg.inv(cell), 1.) * mesh
    for coordinate, charge in zip(scaled, charges):
        index = np.floor(coordinate).astype(int)
        t = coordinate - index
        weights = np.array([(1-t)**3, 3*t**3-6*t*t+4,
                            -3*t**3+3*t*t+3*t+1, t**3]) / 6
        for ix, iy, iz in itertools.product(range(4), repeat=3):
            at = (index + np.array([ix, iy, iz]) - 1) % mesh
            rho[tuple(at)] += charge * weights[ix, 0] * weights[iy, 1] * weights[iz, 2]
    integers = np.stack(np.meshgrid(*(np.fft.fftfreq(n) * n for n in mesh), indexing="ij"), axis=-1)
    k = 2 * np.pi * integers @ np.linalg.inv(cell).T
    k2 = np.sum(k*k, axis=-1)
    cardinal = np.prod((2 + np.cos(2*np.pi*integers/mesh)) / 3, axis=-1)**2
    valid = (k2 != 0) & ~np.any(np.abs(integers) == mesh / 2, axis=-1)
    influence = np.zeros_like(k2)
    influence[valid] = 2*np.pi/np.linalg.det(cell) * np.exp(-k2[valid]/(4*alpha**2)) / k2[valid] / cardinal[valid]
    return COULOMB_KA * np.sum(influence * np.abs(np.fft.fftn(rho))**2)


def independent_periodic_energy(positions, charges, cell, cutoff, alpha, kmax, mesh=None):
    """Finite, unshifted, 3D tin-foil Ewald with neutralizing background."""
    x = np.asarray(positions, float).reshape(-1, 3)
    q = np.asarray(charges, float)
    inverse = np.linalg.inv(cell)
    volume = float(np.linalg.det(cell))
    real = 0.0
    for i in range(len(q)):
        for j in range(i):
            frac = (x[i] - x[j]) @ inverse
            r = float(np.linalg.norm((frac - np.rint(frac)) @ cell))
            if r < cutoff and q[i] * q[j] != 0:
                real += q[i] * q[j] * math.erfc(alpha * r) / r
    reciprocal = 0.0
    for indices in itertools.product(*(range(-n, n + 1) for n in kmax)):
        if indices == (0, 0, 0):
            continue
        k = 2 * math.pi * np.asarray(indices) @ inverse.T
        k2 = float(k @ k)
        phase = x @ k
        sr = float(q @ np.cos(phase))
        si = float(q @ np.sin(phase))
        reciprocal += 2 * math.pi / volume * math.exp(-k2 / (4 * alpha**2)) / k2 * (sr**2 + si**2)
    if mesh is not None:
        reciprocal = independent_mesh_reciprocal(x, q, cell, alpha, mesh) / COULOMB_KA
    self_energy = -alpha / math.sqrt(math.pi) * float(q @ q)
    background = -math.pi * float(q.sum())**2 / (2 * alpha**2 * volume)
    return COULOMB_KA * (real + reciprocal + self_energy + background)


def independent_guest_energy(potential, entries, mesh=None):
    """Electrostatic guest contribution for zero-LJ, zero-bond fixtures."""
    molecule = potential.species[0]
    charges = [potential.host_charge]
    positions = [potential.host_positions]
    for _, coordinates in entries:
        positions.append(np.asarray(coordinates))
        charges.append(molecule.charge[:len(coordinates)])
    kwargs = dict(cell=potential.cell, cutoff=potential.cutoff,
                  alpha=potential.alpha, kmax=tuple(potential.ewald.kmax), mesh=mesh)
    energy = independent_periodic_energy(np.concatenate(positions), np.concatenate(charges), **kwargs)
    energy -= independent_periodic_energy(potential.host_positions, potential.host_charge, **kwargs)
    graph = np.full((molecule.n_atoms, molecule.n_atoms), 100)
    np.fill_diagonal(graph, 0)
    for i, parent in enumerate(molecule.parents):
        if parent >= 0:
            graph[i, parent] = graph[parent, i] = 1
    for k in range(molecule.n_atoms):
        graph = np.minimum(graph, graph[:, k, None] + graph[None, k, :])
    exclusions = set(map(tuple, molecule.exclusions))
    for _, coordinates in entries:
        for i in range(len(coordinates)):
            for j in range(i):
                scale = (0.0 if molecule.is_rigid or (j, i) in exclusions else
                         potential.scale14_coulomb if graph[i, j] == 3 else 1.0)
                r = float(np.linalg.norm(coordinates[i] - coordinates[j]))
                energy -= (1 - scale) * COULOMB_KA * molecule.charge[i] * molecule.charge[j] / r
    return energy


def native_site_probe(sim, excluded, prefix, trials):
    """Read-only production-path stage probe; no RNG or acceptance mutation."""
    pointer = C.POINTER(C.c_double)
    function = sim._lib.poremc_fast_site_energies
    function.argtypes = [C.c_void_p, C.c_size_t, C.c_size_t, pointer, C.c_size_t, pointer, pointer]
    function.restype = C.c_int
    prefix = np.ascontiguousarray(prefix, dtype=np.float64).reshape(-1, 3)
    trials = np.ascontiguousarray(trials, dtype=np.float64).reshape(-1, 3)
    result = np.empty(len(trials))
    error = function(sim._handle, C.c_size_t(-1).value if excluded is None else excluded,
                     len(prefix), prefix.ctypes.data_as(pointer), len(trials),
                     trials.ctypes.data_as(pointer), result.ctypes.data_as(pointer))
    assert error == 0, sim._lib.poremc_fast_last_error().decode()
    return result


def native_molecule_probe(sim, excluded, trials):
    pointer = C.POINTER(C.c_double)
    function = sim._lib.poremc_fast_molecule_energies
    function.argtypes = [C.c_void_p, C.c_size_t, C.c_size_t, pointer, pointer]
    function.restype = C.c_int
    trials = np.ascontiguousarray(trials, dtype=np.float64)
    result = np.empty(len(trials))
    error = function(sim._handle, C.c_size_t(-1).value if excluded is None else excluded,
                     len(trials), trials.ctypes.data_as(pointer), result.ctypes.data_as(pointer))
    assert error == 0, sim._lib.poremc_fast_last_error().decode()
    return result


def electrostatic_fixture(kind="rigid", skew=False, charge_scale=1.0, background=False, backend="native"):
    cell = np.array([[31., 0., 0.], [4., 33., 0.], [-3., 5., 35.]]) if skew else np.diag([31., 33., 35.])
    if kind == "mono":
        local = np.zeros((1, 3))
        molecule = Molecule("ion", [0.], [0.], charge=[.25 * charge_scale])
        background = True
    elif kind == "rigid":
        local = np.array([[0., 0., 0.], [1.1, .2, 0.], [-.3, 1., .2]])
        molecule = Molecule("rigid", [0.] * 3, [0.] * 3,
                            charge=np.array([-.4, .2, .2]) * charge_scale,
                            rigid_positions=local)
    else:
        local = np.array([[0., 0., 0.], [1.2, 0., 0.], [1.2, 1.2, 0.], [2.4, 1.2, 0.]])
        molecule = Molecule("chain", [0.] * 4, [0.] * 4,
                            charge=np.array([.3, -.2, .1, -.2]) * charge_scale,
                            bond_lengths=[0., 1.2, 1.2, 1.2],
                            angles=[math.pi / 2] * 4)
    host = np.array([[.1, .2, .3], [.6, .5, .4], [.4, .7, .8], [.8, .1, .6]]) @ cell
    host_q = np.array([.3, -.2, -.25, .15]) * charge_scale
    if background:
        host_q[0] += .11 * charge_scale
    potential = ClassicalPotential([molecule], cell, 8., host, np.zeros(4), np.zeros(4), host_q,
                                   coulomb="ewald", alpha=.27, kmax=(3, 4, 5),
                                   algorithm="cached", scale14_coulomb=.37, backend=backend,
                                   allow_charge_background=background)
    return molecule, potential, local


def check_native_ewald_full_energy_against_independent_sum(kind, skew):
    molecule, potential, local = electrostatic_fixture(kind, skew)
    with FastGCMC([molecule], potential, potential.cell, 800., [2 / np.linalg.det(potential.cell)], trials=4) as sim:
        assert sim.total_energy() == 0.0
        for fractional in ([.95, .12, .7], [.31, .4, .8]):
            # The first copy extends beyond the primary cell but stays unwrapped.
            sim.add_particle(0, local + np.asarray(fractional) @ potential.cell)
            expected = independent_guest_energy(potential, sim.entries())
            np.testing.assert_allclose(sim.total_energy(), expected, atol=2e-8, rtol=3e-12)


def check_native_ewald_each_move_keeps_full_hamiltonian(kind, move):
    molecule, potential, local = electrostatic_fixture(kind, skew=True, charge_scale=.15)
    weights = {"insert": .5, "delete": .5} if move == "exchange" else {move: 1.}
    with FastGCMC([molecule], potential, potential.cell, 1000., [2 / np.linalg.det(potential.cell)],
                  trials=5, seed=9811, move_weights=weights, max_translation=2.) as sim:
        sim.add_particle(0, local + [7., 8., 9.])
        sim.add_particle(0, local + [21., 20., 19.])
        for _ in range(5):
            sim.run(60, sample_every=60, record_energy=True)
            expected = independent_guest_energy(potential, sim.entries())
            np.testing.assert_allclose(sim.total_energy(), expected, atol=2e-8, rtol=3e-12)
        requested = ["insert", "delete"] if move == "exchange" else [move]
        for name in requested:
            assert sim.counters[name]["accepted"] > 0


def check_native_ewald_primary_exclusions_do_not_remove_periodic_images():
    molecule, potential, local = electrostatic_fixture("rigid", charge_scale=1.)
    potential = ClassicalPotential([molecule], potential.cell, 8., coulomb="ewald", alpha=.27,
                                   kmax=(3, 4, 5), algorithm="cached")
    with FastGCMC([molecule], potential, potential.cell, 300., [1e-5], trials=3) as sim:
        sim.add_particle(0, local + [30.8, 10., 15.])
        expected = independent_guest_energy(potential, sim.entries())
        assert abs(expected) > 1e-2
        np.testing.assert_allclose(sim.total_energy(), expected, atol=2e-8, rtol=3e-12)


def check_native_ewald_one_four_scale_applies_to_bare_primary_pair():
    molecule, potential, local = electrostatic_fixture("flexible")
    values = []
    for scale in (0., .37, 1.):
        current = ClassicalPotential([molecule], potential.cell, 8., coulomb="ewald", alpha=.27,
                                     kmax=(3, 4, 5), algorithm="cached", scale14_coulomb=scale)
        with FastGCMC([molecule], current, current.cell, 300., [1e-5]) as sim:
            sim.add_particle(0, local + [9., 11., 13.])
            values.append(sim.total_energy())
    bare14 = COULOMB_KA * molecule.charge[0] * molecule.charge[3] / np.linalg.norm(local[0] - local[3])
    np.testing.assert_allclose(np.array(values) - values[0], np.array([0., .37, 1.]) * bare14,
                               atol=2e-8, rtol=3e-12)


def check_native_ewald_lattice_translation_preserves_total():
    molecule, potential, local = electrostatic_fixture("rigid", skew=True)
    energies = []
    shift = np.array([2, -3, 1]) @ potential.cell
    for displacement in (np.zeros(3), shift):
        with FastGCMC([molecule], potential, potential.cell, 500., [1e-5]) as sim:
            sim.add_particle(0, local + [3., 4., 5.] + displacement)
            sim.add_particle(0, local + [22., 19., 23.])
            energies.append(sim.total_energy())
    np.testing.assert_allclose(energies[0], energies[1], atol=2e-8, rtol=3e-12)


def check_native_ewald_checkpoint_restarts_physical_state(tmp_path):
    molecule, potential, local = electrostatic_fixture("flexible", skew=True, charge_scale=.2)
    with FastGCMC([molecule], potential, potential.cell, 900., [2 / np.linalg.det(potential.cell)],
                  seed=12912, trials=4) as sim:
        sim.add_particle(0, local + [7., 8., 9.])
        sim.run(90, sample_every=10)
        path = tmp_path / "native-ewald.json"
        sim.save_checkpoint(path)
        expected = sim.run(170, sample_every=10)
        expected_positions = [particle.positions.copy() for particle in sim.particles]
        with FastGCMC.from_checkpoint(path, potential) as restored:
            actual = restored.run(170, sample_every=10)
            assert actual["counts"] == expected["counts"]
            for a, b in zip(restored.particles, expected_positions):
                np.testing.assert_array_equal(a.positions, b)
            np.testing.assert_allclose(actual["energies_kelvin"], expected["energies_kelvin"], atol=2e-8, rtol=3e-12)
            np.testing.assert_allclose(restored.total_energy(), independent_guest_energy(potential, restored.entries()),
                                       atol=2e-8, rtol=3e-12)


def check_native_ewald_restart_rejects_changed_reciprocal_hamiltonian(tmp_path):
    molecule, potential, local = electrostatic_fixture("rigid")
    with FastGCMC([molecule], potential, potential.cell, 300., [1e-5]) as sim:
        sim.add_particle(0, local + [5., 6., 7.])
        path = tmp_path / "native-ewald.json"
        sim.save_checkpoint(path)
    changed = ClassicalPotential([molecule], potential.cell, potential.cutoff,
                                 potential.host_positions, potential.host_sigma,
                                 potential.host_epsilon, potential.host_charge,
                                 coulomb="ewald", alpha=potential.alpha, kmax=(4, 4, 5),
                                 scale14_coulomb=potential.scale14_coulomb, algorithm="cached")
    with unittest.TestCase().assertRaisesRegex(ValueError, "fingerprint|Hamiltonian"):
        FastGCMC.from_checkpoint(path, changed)


class NativeLongRangePhysicsTests(unittest.TestCase):
    def test_full_energy_against_independent_sum(self):
        for kind in ("mono", "rigid", "flexible"):
            for skew in (False, True):
                with self.subTest(kind=kind, skew=skew):
                    check_native_ewald_full_energy_against_independent_sum(kind, skew)

    def test_each_move_keeps_full_hamiltonian(self):
        for kind in ("rigid", "flexible"):
            for move in ("translate", "rotate", "regrow", "exchange"):
                with self.subTest(kind=kind, move=move):
                    check_native_ewald_each_move_keeps_full_hamiltonian(kind, move)

    def test_primary_exclusions_preserve_images(self):
        check_native_ewald_primary_exclusions_do_not_remove_periodic_images()

    def test_one_four_scales_bare_primary_pair(self):
        check_native_ewald_one_four_scale_applies_to_bare_primary_pair()

    def test_lattice_translation(self):
        check_native_ewald_lattice_translation_preserves_total()

    def test_checkpoint_restarts_physical_state(self):
        with tempfile.TemporaryDirectory() as directory:
            check_native_ewald_checkpoint_restarts_physical_state(Path(directory))

    def test_checkpoint_rejects_changed_reciprocal_hamiltonian(self):
        with tempfile.TemporaryDirectory() as directory:
            check_native_ewald_restart_rejects_changed_reciprocal_hamiltonian(Path(directory))

    def test_charged_cbmc_prefixes_match_complete_energy_differences(self):
        molecule, potential, local = electrostatic_fixture("flexible", skew=True)
        for method in ("direct", "pme"):
            mesh = (16, 32, 32) if method == "pme" else None
            with self.subTest(method=method), FastGCMC([molecule], potential, potential.cell, 700., [1e-4],
                  reciprocal_method=method, pme_mesh=(16, 32, 32)) as sim:
                sim.add_particle(0, local + [5., 8., 11.])
                sim.add_particle(0, local + [19., 23., 24.])
                saved = sim.total_energy()
                for excluded in (None, 0):
                    entries = [entry for i, entry in enumerate(sim.entries()) if i != excluded]
                    for stage in range(molecule.n_atoms):
                        prefix = local[:stage] + [9., 14., 17.]
                        trials = local[stage] + np.array([[9., 14., 17.], [9.4, 14.1, 17.2], [8.7, 14.3, 17.4]])
                        before = independent_guest_energy(potential, entries + [(0, prefix)], mesh)
                        expected = [independent_guest_energy(potential, entries + [(0, np.vstack([prefix, trial]))], mesh) - before
                                    for trial in trials]
                        np.testing.assert_allclose(native_site_probe(sim, excluded, prefix, trials), expected,
                                                   atol=3e-8, rtol=5e-12)
                # Scratch probes may update temporary contexts but cannot commit state.
                np.testing.assert_allclose(sim.total_energy(), saved, atol=2e-9, rtol=2e-12)

    def test_whole_molecule_trial_batches_match_full_differences(self):
        for kind in ("rigid", "flexible"):
            molecule, potential, local = electrostatic_fixture(kind, skew=True)
            for method in ("direct", "pme"):
                mesh = (32, 16, 32) if method == "pme" else None
                with self.subTest(kind=kind, method=method), FastGCMC([molecule], potential, potential.cell, 700., [1e-4],
                      reciprocal_method=method, pme_mesh=(32, 16, 32)) as sim:
                    sim.add_particle(0, local + [5., 8., 11.])
                    sim.add_particle(0, local + [19., 23., 24.])
                    trials = local[None] + np.array([[9., 14., 17.], [29.8, 30., 33.], [4., 18., 14.]])[:, None, :]
                    for excluded in (None, 1):
                        entries = [entry for i, entry in enumerate(sim.entries()) if i != excluded]
                        before = independent_guest_energy(potential, entries, mesh)
                        expected = [independent_guest_energy(potential, entries + [(0, trial)], mesh) - before for trial in trials]
                        np.testing.assert_allclose(native_molecule_probe(sim, excluded, trials), expected,
                                                   atol=4e-8, rtol=5e-12)

    def test_pme_matches_independent_assignment_fft_hamiltonian(self):
        for kind in ("mono", "rigid", "flexible"):
            for skew in (False, True):
                molecule, potential, local = electrostatic_fixture(kind, skew)
                mesh = (16, 32, 16)
                with self.subTest(kind=kind, skew=skew), FastGCMC([molecule], potential, potential.cell, 800., [1e-4],
                      reciprocal_method="pme", pme_mesh=mesh, trials=3) as sim:
                    sim.add_particle(0, local + [3., 7., 14.])
                    sim.add_particle(0, local + [29.8, 20., 25.])
                    np.testing.assert_allclose(sim.total_energy(), independent_guest_energy(potential, sim.entries(), mesh),
                                               atol=4e-8, rtol=5e-12)

    def test_pme_moves_preserve_same_discrete_hamiltonian(self):
        molecule, potential, local = electrostatic_fixture("flexible", skew=True, charge_scale=.2)
        mesh = (16, 16, 32)
        with FastGCMC([molecule], potential, potential.cell, 800., [2/np.linalg.det(potential.cell)],
                      reciprocal_method="pme", pme_mesh=mesh, trials=4, seed=711) as sim:
            sim.add_particle(0, local + [3., 7., 14.])
            for _ in range(6):
                sim.run(80, sample_every=20)
                np.testing.assert_allclose(sim.total_energy(), independent_guest_energy(potential, sim.entries(), mesh),
                                           atol=4e-8, rtol=5e-12)
            for name in sim.MOVE_NAMES:
                assert sim.counters[name]["accepted"] > 0

    def test_mesh_refinement_converges_to_direct_ewald(self):
        molecule, initial, local = electrostatic_fixture("rigid", skew=True)
        potential = ClassicalPotential([molecule], initial.cell, initial.cutoff,
                                       initial.host_positions, initial.host_sigma, initial.host_epsilon, initial.host_charge,
                                       coulomb="ewald", alpha=initial.alpha, kmax=(10, 10, 10), algorithm="cached")
        entries = [(0, local + [7., 13., 23.]), (0, local + [19., 2., 6.])]
        reference = independent_guest_energy(potential, entries)
        errors = []
        for mesh in (16, 32, 64):
            with FastGCMC([molecule], potential, potential.cell, 700., [1e-4],
                          reciprocal_method="pme", pme_mesh=mesh) as sim:
                for species, coordinates in entries:
                    sim.add_particle(species, coordinates)
                errors.append(abs(sim.total_energy() - reference))
        assert errors[1] < errors[0], errors
        assert errors[2] < errors[1], errors
        assert errors[2] < .02, errors

    def test_single_ion_background_translation_is_uniform_and_always_accepted(self):
        # With one charge, |S(k)|^2=q^2 at every position; self and background
        # terms are also constant. This is an exactly flat canonical target.
        molecule = Molecule("single ion", [0.], [0.], charge=[.9])
        cell = np.array([[30., 0., 0.], [6., 33., 0.], [-2., 4., 35.]])
        potential = ClassicalPotential([molecule], cell, 8., coulomb="ewald", alpha=.27,
                                       kmax=(3, 4, 5), allow_charge_background=True)
        with FastGCMC([molecule], potential, cell, 300., [0.], move_weights={"translate": 1.},
                      max_translation=20., seed=2313) as sim:
            sim.add_particle(0, np.array([[4., 6., 8.]]))
            expected = independent_guest_energy(potential, sim.entries())
            result = sim.run(2000, sample_every=20)
            assert sim.counters["translate"]["accepted"] == 2000
            np.testing.assert_allclose(result["energies_kelvin"], expected, atol=3e-8, rtol=4e-12)

    def test_ewald_zero_charge_gcmc_matches_poisson_for_multiple_trial_counts(self):
        molecule = Molecule("ideal reference", [0.], [0.], charge=[0.])
        cell = np.diag([28., 29., 30.])
        potential = ClassicalPotential([molecule], cell, 8., coulomb="ewald", alpha=.27, kmax=2)
        target = 2.7
        for trials in (1, 7):
            with self.subTest(trials=trials), FastGCMC([molecule], potential, cell, 300.,
                      [target/np.linalg.det(cell)], trials=trials, seed=4300+trials,
                      move_weights={"insert": .5, "delete": .5}) as sim:
                result = sim.run(60000, burnin=2000, sample_every=5, record_energy=False)
                count = np.array(result["counts"])[:, 0]
                assert abs(count.mean()-target) < .12, (trials, count.mean())
                assert abs(count.var()-target) < .28, (trials, count.var())


if __name__ == "__main__":
    unittest.main()
