#!/usr/bin/env python3
"""Compile and run the independent scalar-field host-grid audit."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(ROOT/'evidence06/host_grid_module.json'))
    args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='poremc06-host-grid-') as temp:
        temp=Path(temp);binary=temp/'audit'
        command=['g++','-std=c++17','-O3','-fno-fast-math','-ffp-contract=off','-fopenmp',
                 '-I',str(ROOT/'native'),str(ROOT/'tests06/host_grid_audit.cpp'),
                 str(ROOT/'native/host_grid.cpp'),str(ROOT/'native/kernels.cpp'),'-o',str(binary)]
        subprocess.run(command,check=True)
        run=subprocess.run([str(binary),str(temp/'field.cache')],check=True,text=True,capture_output=True)
        result=json.loads(run.stdout)
        result['verification']='Analytic Fourier field, analytic short-range pair, periodic wrapping, batched site indexing, capped reference, Boltzmann reweighting and zero-weight node limit, temperature-independent lazy disk cache and integrity rebuild'
        result['compile_flags']=command[1:6]
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
