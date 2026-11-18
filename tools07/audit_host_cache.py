#!/usr/bin/env python3
"""Run cache OFF, cold write and warm read through FastGCMC on a real input or a charged built-in fixture."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def run_worker(args):
    sys.path.insert(0, str(args.source.resolve()))
    sys.path.insert(1, str(ROOT))
    import numpy as np
    from poremc.fast import FastGCMC
    from poremc.cli import construct_case
    from tools06.fixtures import workload_fixture, make_potential, simulation_arguments
    cache = None if args.state == 'off' else str(args.cache)
    start = time.perf_counter()
    if args.input:
        cfg = json.loads(args.input.read_text())
        settings = cfg.setdefault('sampling', {})
        settings.update(driver='cpp', sampling_mode='delayed', seed=args.seed,
                        host_grid_mesh=args.mesh, host_grid_cache=cache)
        cfg.setdefault('potential', {})['threads'] = 1
        sim, *_ = construct_case(cfg, args.input.resolve().parent)
    else:
        molecule, physics, initial = workload_fixture(host_atoms=args.host_atoms, kmax=args.kmax)
        pot = make_potential(molecule, physics)
        sim = FastGCMC(molecule, pot, physics['cell'], physics['temperature'], [physics['activity']],
                       seed=args.seed, sampling_mode='delayed', host_grid_mesh=args.mesh,
                       host_grid_cache=cache, **simulation_arguments(physics))
        for xyz in initial[:5]:
            sim.add_particle(0, xyz)
    construction = time.perf_counter() - start
    with sim:
        result = sim.run(args.steps, sample_every=max(1, args.steps//10))
        signature = hashlib.sha256()
        for key in ('counts', 'energies_kelvin'):
            signature.update(np.ascontiguousarray(result[key]).tobytes())
        for _, xyz in sim.entries():
            signature.update(np.ascontiguousarray(xyz).tobytes())
        signature.update(json.dumps(sim.counters, sort_keys=True).encode())
        stats = sim.acceleration_profile['host_grid']
        print(json.dumps(dict(state=args.state, native_sha256=sha(Path(sim._lib._name).read_bytes()),
                            construction_seconds=construction,
                            builds=stats['builds'], writes=stats['cache_writes'], loads=stats['cache_loads'],
                            trajectory_sha256=signature.hexdigest(), energy_K=sim.total_energy(),
                            file_bytes=args.cache.stat().st_size if cache else 0)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, help='Optional original PoreMC system JSON; relative assets resolve at the input directory')
    p.add_argument('--mesh', type=int, choices=[8,16,32,64,128], default=64)
    p.add_argument('--steps', type=int, default=128)
    p.add_argument('--seed', type=int, default=725)
    p.add_argument('--workdir', type=Path, default=Path('.'), help='Parent on the filesystem being tested')
    p.add_argument('--output', type=Path)
    p.add_argument('--source', type=Path, default=ROOT, help='Root of a compiled PoreMC checkout')
    p.add_argument('--host-atoms', type=int, default=4416)
    p.add_argument('--kmax', type=int, default=13)
    p.add_argument('--state', choices=('off', 'cold', 'warm'), help=argparse.SUPPRESS)
    p.add_argument('--cache', type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.steps <= 0:
        p.error('--steps must be positive')
    if args.state:
        return run_worker(args)
    args.workdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONFAULTHANDLER='1')
    records = []
    with tempfile.TemporaryDirectory(dir=args.workdir.resolve(), prefix='poremc-host-grid-') as directory:
        cache = Path(directory) / 'framework-grid.bin'
        for state in ('off', 'cold', 'warm'):
            cmd = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--state', state, '--cache', str(cache)]
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if proc.returncode:
                raise RuntimeError(f'{state}: subprocess exit {proc.returncode}\n{proc.stderr}')
            records.append(json.loads(proc.stdout))
        assert [(r['builds'],r['writes'],r['loads']) for r in records] == [(1,0,0),(1,1,0),(0,0,1)]
        assert all(r['trajectory_sha256']==records[0]['trajectory_sha256'] and r['energy_K']==records[0]['energy_K'] for r in records)
        out = dict(input=str(args.input) if args.input else 'synthetic aromatic12, first five fixture molecules',
                   host_atoms=None if args.input else args.host_atoms, steps=args.steps,
                   kmax=None if args.input else args.kmax, seed=args.seed, mesh=args.mesh,
                   cache_sha256=sha(cache.read_bytes()), cache_and_trajectory_match=True, records=records)
        data = json.dumps(out, indent=2, allow_nan=False) + '\n'
        if args.output:
            args.output.write_text(data)
        print(data, end='')

if __name__ == '__main__':
    main()
