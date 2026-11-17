#!/usr/bin/env python3
"""Run three bounded gRASPA repeats, record device conditions, and compare locally."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def device_state(max_temperature):
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=name,temperature.gpu,utilization.gpu,memory.used',
         '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    state = result.stdout.strip()
    temperatures = [float(line.split(',')[1]) for line in state.splitlines()]
    if any(t >= max_temperature for t in temperatures):
        raise RuntimeError(f'device temperature reached the {max_temperature:g} C limit; no further reference run started')
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', default='graspa')
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--max-temperature', type=float, default=87)
    parser.add_argument('--timeout', type=float, default=90)
    args = parser.parse_args()
    executable = shutil.which(args.executable)
    if executable is None:
        parser.error(f'executable not found: {args.executable}')
    executable = str(Path(executable).resolve())
    source = ROOT / 'benchmarks/inputs/graspa'
    model = json.loads((source / 'model.json').read_text())
    initial_device = device_state(args.max_temperature)
    processes = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
        capture_output=True, text=True, check=False)
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source, directory / 'inputs')
    (directory / 'device_before.txt').write_text(initial_device + '\n')
    metadata = {'date_UTC': datetime.now(timezone.utc).isoformat(),
                'executable': executable, 'CUDA_LAUNCH_BLOCKING': '1',
                'shared_device_existing_task': (bool(processes.stdout.strip())
                                                if processes.returncode == 0 else None),
                'device_temperature_limit_C': args.max_temperature}
    (directory / 'execution.json').write_text(json.dumps(metadata, indent=2) + '\n')
    environment = dict(os.environ, CUDA_LAUNCH_BLOCKING='1',
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    records = []
    for seed in model['seeds']:
        before = device_state(args.max_temperature)
        run = directory / f'timed_seed_{seed}'
        shutil.copytree(source, run)
        config = (run / 'simulation.input').read_text()
        config, count = re.subn(r'(?m)^RandomSeed[ \t]+\d+[ \t]*$', f'RandomSeed {seed}', config)
        if count != 1:
            raise ValueError('expected exactly one RandomSeed field')
        (run / 'simulation.input').write_text(config)
        with (run / 'console.log').open('w') as log:
            subprocess.run([executable], cwd=run, env=environment,
                           stdout=log, stderr=subprocess.STDOUT,
                           timeout=args.timeout, check=True)
        text = (run / 'console.log').read_text()
        times = re.findall(r'Work took\s+([-+0-9.eE]+)\s+seconds', text)
        if len(times) != 1 or 'END OF PROGRAM' not in text:
            raise ValueError(f'incomplete or ambiguous gRASPA timing in {run}')
        record = {'seed': seed, 'gRASPA_sampling_seconds': float(times[0]),
                  'steps': 1000, 'device_before': before}
        # Persist completed timing before the post-run temperature check.
        records.append(record)
        (directory / 'graspa_timings.json').write_text(json.dumps(records, indent=2) + '\n')
        record['device_after'] = device_state(args.max_temperature)
        (directory / 'graspa_timings.json').write_text(json.dumps(records, indent=2) + '\n')
    subprocess.run([sys.executable, str(ROOT / 'tools/analyze_graspa_comparison.py'),
                    str(directory), '--output', str(directory / 'graspa_comparison.json')],
                   env=environment, check=True)


if __name__ == '__main__':
    main()
