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
    p.add_argument('--cuda', action='store_true', help='Also build the native CUDA reciprocal-candidate plugin')
    p.add_argument('--nvcc', default=os.environ.get('NVCC', 'nvcc'))
    p.add_argument('--cuda-arch', default='sm_86', help='CUDA target; default sm_86 for RTX 30-series')
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    extension = '.dylib' if sys.platform == 'darwin' else '.so'
    output = root.parent / 'poremc' / ('libporemc' + extension)
    flags = ['-std=c++17', '-O3', '-DNDEBUG', '-fPIC', '-shared', '-fno-fast-math', '-ffp-contract=off']
    if not args.portable:
        flags += ['-march=native']
    if not args.no_openmp:
        flags += ['-fopenmp']
    command = shlex.split(args.compiler) + flags + [str(root / name) for name in ('kernels.cpp','fast_mc.cpp','ewald_state.cpp','host_grid.cpp')]+['-o', str(output)]
    if sys.platform.startswith('linux'):
        command += ['-ldl']
    print(' '.join(shlex.quote(x) for x in command))
    t0 = time.perf_counter()
    subprocess.run(command, check=True)
    metadata = {'command': command, 'compiler': subprocess.check_output(shlex.split(args.compiler)+['--version'], text=True).splitlines()[0],
                'platform': platform.platform(), 'build_seconds': time.perf_counter()-t0, 'fast_math': False,
                'portable': args.portable, 'openmp': not args.no_openmp, 'library': output.name}
    if args.cuda:
        cuda_output=output.with_name('libporemc_ewald_cuda'+extension)
        cuda_command=shlex.split(args.nvcc)+['-std=c++17','-O3','--fmad=false','-arch='+args.cuda_arch,
                        '-Xcompiler','-fPIC','-shared',str(root/'ewald_cuda.cu'),'-o',str(cuda_output)]
        print(' '.join(shlex.quote(x) for x in cuda_command))
        cuda_t0=time.perf_counter();subprocess.run(cuda_command,check=True)
        metadata['cuda']={'command':cuda_command,'build_seconds':time.perf_counter()-cuda_t0,
                          'architecture':args.cuda_arch,'library':cuda_output.name,
                          'compiler':subprocess.check_output(shlex.split(args.nvcc)+['--version'],text=True)}
    (root / 'build_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Built {output}')


if __name__ == '__main__':
    main()
