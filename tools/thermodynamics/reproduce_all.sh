#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
out=benchmarks/runs/thermodynamics
mkdir -p "$out"
python -m pytest tests/test_thermodynamics.py -q --junitxml="$out/all_tests.xml" | tee "$out/all_tests.log"
for task in response salts umbrella molecular convergence performance; do
  python tools/thermodynamics/reproduce.py "$task"
done
for voltage in -0.1 0 0.1; do
  python tools/thermodynamics/reproduce.py interface --voltage "$voltage"
done
