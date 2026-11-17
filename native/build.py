#!/usr/bin/env python3
"""Build PoreMC's standalone C++17 energy library; no Python headers needed."""
import argparse
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', default=os.environ.get('CXX', 'g++'))
    p.add_argument('--portable', action='store_true', help='Omit -march=native for a portable binary')
    p.add_argument('--no-openmp', action='store_true')
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    extension = '.dylib' if sys.platform == 'darwin' else '.so'
    output = root.parent / 'poremc' / ('libporemc' + extension)
    flags = ['-std=c++17', '-O3', '-DNDEBUG', '-fPIC', '-shared', '-fno-fast-math', '-ffp-contract=off']
    if not args.portable:
        flags += ['-march=native']
    if not args.no_openmp:
        flags += ['-fopenmp']
    command = shlex.split(args.compiler) + flags + [str(root / 'kernels.cpp'), str(root / 'fast_mc.cpp'), '-o', str(output)]
    print(' '.join(shlex.quote(x) for x in command))
    t0 = time.perf_counter()
    subprocess.run(command, check=True)
    metadata = {'command': command, 'compiler': subprocess.check_output(shlex.split(args.compiler)+['--version'], text=True).splitlines()[0],
                'platform': platform.platform(), 'build_seconds': time.perf_counter()-t0, 'fast_math': False,
                'portable': args.portable, 'openmp': not args.no_openmp, 'library': output.name}
    (root / 'build_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Built {output}')


if __name__ == '__main__':
    main()
