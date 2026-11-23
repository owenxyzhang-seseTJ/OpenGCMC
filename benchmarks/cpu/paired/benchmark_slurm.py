"""Verify multi-pressure checkpoint dispatch on the live Slurm controller."""

import argparse
import json
from pathlib import Path
import subprocess

from poremc.scheduling import RunStore
from poremc.slurm import run_slurm


def run(model, directory, partition):
    store = RunStore(directory / 'runstore')
    plan = store.plan(
        json.loads(model.read_text()),
        base_directory=model.parent,
        pressure_pa=[20000, 100000, 500000],
        replicas=4,
        block_steps=12000,
        count_mcse=0.20,
        log_selectivity_mcse=0.10,
    )
    job = store.submit(plan['plan_id'], 9918392, 'mfi-cpu-dispatch')
    duplicate = store.submit(plan['plan_id'], 9918392, 'mfi-cpu-dispatch')
    run_id = job['run_id']
    if duplicate != {'run_id': run_id, 'created': False}:
        raise RuntimeError('The repeated request did not resolve to the original run')
    state = run_slurm(store, run_id, workers=4, poll_seconds=5, partition=partition)
    result = store.result(run_id)
    waves = store.root / 'runs' / run_id / 'slurm'
    job_ids = [json.loads(file.read_text())['job'] for file in sorted(waves.glob('*/job.json'))]
    result.update(plan=plan, repeated_submission=duplicate, slurm_jobs=job_ids)
    (directory / 'dispatch.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    if job_ids:
        with (directory / 'slurm_jobs.txt').open('w') as log:
            subprocess.run(
                ['sacct', '-X', '-j', ','.join(job_ids), '-P', '-o',
                 'JobID,State,Elapsed,ExitCode,NodeList,AllocCPUS'],
                stdout=log,
                check=True,
            )
    if state['status'] != 'completed' or result['blocks'] != 24 or result['chains'] != {'completed': 12}:
        raise RuntimeError(f'Incomplete workflow: {state}')
    print(run_id, len(job_ids), 'Slurm arrays, 12 chains, 24 blocks', flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('model', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('--partition', default='intel')
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run(args.model, args.output, args.partition)
