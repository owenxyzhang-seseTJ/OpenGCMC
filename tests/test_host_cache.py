"""Normal FastGCMC entry point, cache publication and rollback on Linux."""
import os
import signal
import sys

import pytest
from tools.longrange.fixtures import workload_fixture, make_potential, simulation_arguments
from poremc.fast import FastGCMC


def sample(cache=None, seed=772, host_atoms=26):
    molecule, physics, initial = workload_fixture(host_atoms=host_atoms, kmax=2)
    pot = make_potential(molecule, physics)
    with FastGCMC(molecule, pot, physics['cell'], physics['temperature'], [physics['activity']],
                  sampling_mode='delayed', host_grid_mesh=8, host_grid_cache=cache,
                  seed=seed, **simulation_arguments(physics)) as sim:
        sim.add_particle(0, initial[0])
        result = sim.run(50, sample_every=5)
        return (result['counts'], result['energies_kelvin'], sim.counters), sim.acceleration_profile['host_grid']


def test_cold_warm_and_max_length_name(tmp_path):
    name_max = os.pathconf(tmp_path, 'PC_NAME_MAX') if os.name == 'posix' else 80
    path = tmp_path / 'nested' / ('x' * name_max)
    off, unused = sample()
    cold, first = sample(path)
    warm, second = sample(path)
    assert off == cold == warm
    assert unused['cache_writes'] == unused['cache_loads'] == 0
    assert first['cache_writes'] == second['cache_loads'] == 1
    assert second['builds'] == 0
    assert path.read_bytes()[:8] == b'PMCHG061'


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux file size limit and signal semantics')
def test_partial_write_preserves_destination(tmp_path):
    import resource
    cache = tmp_path / 'cache.bin'
    reference, _ = sample(cache)
    before = cache.read_bytes()
    limit = resource.getrlimit(resource.RLIMIT_FSIZE)
    handler = signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    try:
        resource.setrlimit(resource.RLIMIT_FSIZE, (512, limit[1]))
        with pytest.raises(RuntimeError, match='host grid cache (write|flush).*cache.bin'):
            sample(cache, host_atoms=27)
    finally:
        resource.setrlimit(resource.RLIMIT_FSIZE, limit)
        signal.signal(signal.SIGXFSZ, handler)
    assert cache.read_bytes() == before
    assert list(tmp_path.iterdir()) == [cache]
    recovered, stats = sample(cache)
    assert recovered == reference and stats['cache_loads'] == 1


def test_failed_publish_keeps_existing_directory(tmp_path):
    cache = tmp_path / 'directory'
    cache.mkdir()
    keep = cache / 'existing'
    keep.write_bytes(b'preserve')
    with pytest.raises(RuntimeError, match='host grid cache rename failed'):
        sample(cache)
    assert keep.read_bytes() == b'preserve'
    assert list(tmp_path.iterdir()) == [cache]


def test_empty_host_and_zero_lj_charge_channels(tmp_path):
    from poremc.topology import Molecule

    molecule, physics, _ = workload_fixture(host_atoms=0, kmax=2)
    molecule = Molecule('zero sites', [0.], [0.], charge=[0.], rigid_positions=[[0.,0.,0.]])
    path = tmp_path / 'empty-fields.bin'
    for expected in ('cache_writes','cache_loads'):
        pot = make_potential(molecule, physics)
        with FastGCMC(molecule, pot, physics['cell'], 350., [physics['activity']],
                      sampling_mode='delayed', host_grid_mesh=8, host_grid_cache=path) as sim:
            assert sim.acceleration_profile['host_grid'][expected] == 1
            assert sim.trial_surrogate_molecule_energies([[12.,9.,4.]]).item() == 0.
