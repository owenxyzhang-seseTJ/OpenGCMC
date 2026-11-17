#!/usr/bin/env python3
"""Run the retained RASPA3 protocol, then its same-system native comparison."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', default='raspa3')
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=180)
    args = parser.parse_args()
    executable = shutil.which(args.executable)
    if executable is None:
        parser.error(f'executable not found: {args.executable}')
    executable = str(Path(executable).resolve())
    source = ROOT / 'benchmarks/inputs/cubtc'
    model = json.loads((source / 'model.json').read_text())
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source / 'model.json', directory / 'model.json')
    metadata = {'date_UTC': datetime.now(timezone.utc).isoformat(),
                'executable': executable, 'reference_runs': [],
                'OMP_NUM_THREADS': '1', 'protocol': 'conventional K=1 insertion/deletion'}
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    for seed in model['seeds']:
        run = directory / f'seed_{seed}'
        run.mkdir()
        for name in ('ControlledCuBTC.cif', 'force_field.json', 'probe.json'):
            shutil.copy2(source / name, run / name)
        config = json.loads((source / 'simulation.json').read_text())
        config['RandomSeed'] = seed
        (run / 'simulation.json').write_text(json.dumps(config, indent=2) + '\n')
        with (run / 'console.log').open('w') as log:
            subprocess.run([executable], cwd=run, env=environment,
                           stdout=log, stderr=subprocess.STDOUT,
                           timeout=args.timeout, check=True)
        metadata['reference_runs'].append({'seed': seed, 'directory': run.name})
        (directory / 'execution.json').write_text(json.dumps(metadata, indent=2) + '\n')
    subprocess.run([sys.executable, str(ROOT / 'tools/analyze_raspa3_comparison.py'),
                    str(directory), '--output', str(directory / 'raspa3_comparison.json')],
                   env=environment, check=True)


if __name__ == '__main__':
    main()
