#!/usr/bin/env python3
"""Measure exact and delayed native CPU sampling on one OpenGCMC JSON input."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
MODES = ('exact', 'delayed16', 'delayed32', 'delayed64', 'delayed128',
         'biased16', 'biased32', 'biased64', 'biased128')


def worker(args):
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from poremc.cli import clean_json, construct_case

    path = args.input
    config = json.loads(path.read_text())
    if config.get('surrogate'):
        raise ValueError('Native benchmarking requires a classical potential without an external surrogate.')
    potential = config.setdefault('potential', {})
    if potential.get('coulomb') != 'ewald':
        raise ValueError('This benchmark requires potential.coulomb=ewald.')
    potential.update(backend='native', algorithm='cached', threads=1)

    sampling = config.setdefault('sampling', {})
    sampling.pop('cuda_library', None)
    sampling.update(driver='cpp', reciprocal_method='direct', reciprocal_backend='cpu',
                    seed=args.seed, sampling_mode='exact', ewald_half_spectrum=True,
                    ewald_k_cutoff=0., host_grid_cache=None, grid_root_bias=0.)
    if args.mode != 'exact':
        biased = args.mode.startswith('biased')
        mesh = int(args.mode.removeprefix('biased' if biased else 'delayed'))
        if biased and config.get('root_bias'):
            raise ValueError('A grid-biased mode requires an input without an explicit root_bias.')
        cache = args.cache_dir / f'{path.stem}-{mesh}.bin' if args.cache_dir is not None else None
        sampling.update(sampling_mode='delayed', host_grid_mesh=mesh, host_grid_cap=args.cap,
                        host_grid_cache=str(cache) if cache else None,
                        grid_root_bias=.95 if biased else 0.)

    start = time.perf_counter()
    sim, _, _, _, _ = construct_case(config, path.parent)
    construction = time.perf_counter() - start
    with sim:
        initial_energy = sim.total_energy()
        initial_n = int(sim.counts[0])
        if args.burnin:
            sim.run(args.burnin, sample_every=args.burnin, record_energy=False)
        before = sim.counters
        result = sim.run(args.steps, sample_every=args.sample_every, record_energy=False)
        after = sim.counters
        changes = {
            name: {key: after[name][key] - before[name][key] for key in ('attempted', 'accepted')}
            for name in sim.MOVE_NAMES
        }
        final_energy = sim.total_energy()
        reference = sim.potential.total_energy(sim.entries())
        counts = np.asarray(result['counts']).reshape(-1)
        seconds = result['production_seconds']
        row = dict(
            mode=args.mode, seed=args.seed, steps=args.steps, burnin=args.burnin,
            sample_every=args.sample_every, construction_seconds=construction,
            production_seconds=seconds,
            attempted_moves_per_second=result['attempted_moves_per_second'],
            accepted_moves_per_second=sum(v['accepted'] for v in changes.values()) / seconds,
            production_acceptance=changes, initial_molecules=initial_n,
            initial_energy_K=initial_energy, final_energy_K=final_energy,
            final_recompute_error_K=abs(final_energy - reference), final_molecules=int(sim.counts[0]),
            count_mean=float(counts.mean()) if counts.size else None, count_trace=counts.tolist(),
            long_range=sim.long_range_profile, acceleration=sim.acceleration_profile,
        )
    print(json.dumps(clean_json(row), allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=lambda path: Path(path).resolve(), help='Complete OpenGCMC system JSON')
    parser.add_argument('--steps', type=int, default=10000, help='Production attempted moves')
    parser.add_argument('--burnin', type=int, default=1000)
    parser.add_argument('--sample-every', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=6106)
    parser.add_argument('--modes', default='exact,delayed32,delayed64', help=', '.join(MODES))
    parser.add_argument('--cap', type=float, default=1e5, help='Reference field cap in K')
    parser.add_argument('--cache-dir', type=Path, help='Optional directory for reusable fixed-host fields')
    parser.add_argument('--output', type=Path, default=Path('longrange_benchmark.json'))
    parser.add_argument('--mode', choices=MODES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.steps, args.repeats, args.sample_every) < 1 or args.burnin < 0:
        parser.error('Positive steps, repeats and sample-every and nonnegative burnin are required.')
    if args.mode:
        return worker(args)

    modes = [mode.strip() for mode in args.modes.split(',')]
    if 'exact' not in modes or not set(modes) <= set(MODES) or len(modes) != len(set(modes)):
        parser.error('Include exact and choose unique supported exact, delayed or biased modes.')
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    data = dict(
        scope='Paired single-thread native CPU runs with the same finite direct Ewald target.',
        platform=platform.platform(), case=args.input.stem, modes=modes,
        cache='persistent' if args.cache_dir else 'in memory',
        rows=[], summary=[],
    )
    if args.output.resolve() == args.input:
        parser.error('The benchmark output must differ from the simulation input.')
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        pending = args.output.with_name(args.output.name + '.tmp')
        pending.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
        pending.replace(args.output)

    for repeat in range(args.repeats):
        order = modes if repeat % 2 == 0 else reversed(modes)
        for mode in order:
            command = [
                sys.executable, str(Path(__file__).resolve()), str(args.input),
                '--mode', mode, '--steps', str(args.steps), '--burnin', str(args.burnin),
                '--sample-every', str(args.sample_every), '--seed', str(args.seed + repeat),
                '--cap', str(args.cap),
            ]
            if args.cache_dir:
                command += ['--cache-dir', str(args.cache_dir)]
            row = json.loads(subprocess.check_output(command, text=True, env=env))
            data['rows'].append(row)
            save()
            print(f"{mode}, seed={row['seed']}: {row['attempted_moves_per_second']:.2f} attempts/s",
                  flush=True)

    references = {row['seed']: row for row in data['rows'] if row['mode'] == 'exact'}
    for mode in modes:
        rows = [row for row in data['rows'] if row['mode'] == mode]
        ratios = []
        for row in rows:
            reference = references[row['seed']]
            if abs(row['initial_energy_K'] - reference['initial_energy_K']) > 1e-5:
                raise RuntimeError('Paired modes produced different initial target energies.')
            ratios.append(row['attempted_moves_per_second'] / reference['attempted_moves_per_second'])
        data['summary'].append(dict(
            mode=mode, median_attempts_s=statistics.median(row['attempted_moves_per_second'] for row in rows),
            median_accepted_s=statistics.median(row['accepted_moves_per_second'] for row in rows),
            paired_speedups=ratios, median_speedup=statistics.median(ratios),
            max_recompute_error_K=max(row['final_recompute_error_K'] for row in rows),
        ))
    save()
    print(json.dumps(data['summary'], indent=2))


if __name__ == '__main__':
    main()
