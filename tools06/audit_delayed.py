#!/usr/bin/env python3
"""Export the independent 0.6 delayed-CBMC checks as machine-readable evidence."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default=str(ROOT/'evidence06/delayed_physics_audit.json'))
    args=p.parse_args()
    spec=importlib.util.spec_from_file_location('delayed_audit',ROOT/'tests06/test_delayed_physics.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cases=('prefix_and_residual','nonuniform_root_poisson','canonical_quadrature','all_moves_and_restart','cache_identity')
    evidence=dict(version='0.6',energy_unit='U/k_B in K',
        native_library_sha256=hashlib.sha256((ROOT/'poremc/libporemc.so').read_bytes()).hexdigest(),checks={})
    for case in cases:
        start=time.perf_counter()
        result=getattr(module,'check_'+case)()
        evidence['checks'][case]=dict(passed=True,seconds=time.perf_counter()-start,**result)
        print(case,'passed',flush=True)
    Path(args.output).write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
    print(args.output)


if __name__=='__main__':main()
