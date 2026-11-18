#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
out=benchmarks/runs/electrochemistry
examples=examples/electrochemistry
mkdir -p "$out"
python -m pytest tests -q --junitxml="$out/all_tests.xml" | tee "$out/all_tests.log"
python -m poremc.reactive_cli run "$examples/hydrated_interface.json" -o "$out/hydrated_interface" > "$out/hydrated_interface.log"
python -m poremc.reactive_cli calibrate "$examples/hydrated_redox_calibration.json" -o "$out/hydrated_redox_calibration" > "$out/redox_calibration.log"
python -m poremc.reactive_cli calibrate "$examples/hydrated_acid_calibration.json" -o "$out/hydrated_acid_calibration" > "$out/acid_calibration.log"
python tools/electrochemistry/update_reference_demo.py
python -m poremc.reactive_cli run "$out/hydrated_calibrated_pipeline.json" -o "$out/hydrated_calibrated_pipeline" > "$out/calibrated_pipeline.log"
python tools/electrochemistry/run_demonstrations.py
python tools/electrochemistry/collect_results.py | tee "$out/result_collection.log"
