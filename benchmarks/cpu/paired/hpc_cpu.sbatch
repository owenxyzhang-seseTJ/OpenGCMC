#!/usr/bin/env bash
#SBATCH --job-name=opengcmc-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-10%2

set -euo pipefail

stage=${1:?Provide the absolute staged study directory}
benchmark_python=${2:?Provide the absolute Python executable with NumPy and SciPy}
raspa_binary=${3:?Provide the absolute Linux RASPA3 executable}
study=${4:-"$stage/study"}
groups=(controls xylene_low xylene_mid xylene_high ton_butane irmof_mixture nist cfcmc blind cache two_pore blind_budget ton_butane_dilute xylene_low_confirm holdout_same_candidates kernel_controls)
group=${groups[SLURM_ARRAY_TASK_ID]}
dest="$stage/results/$group"
mkdir -p "$dest"

scratch=$(mktemp -d "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/opengcmc-cpu.XXXXXXXX")
trap 'rm -rf "$scratch"' EXIT
cp -R "$stage/OpenGCMC" "$scratch/OpenGCMC"
export PYTHONPATH="$scratch/OpenGCMC"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
{
    date -u
    uname -sm
    lscpu
    "$benchmark_python" --version
    g++ --version | head -1
    printf 'allocated CPUs %s\n' "$SLURM_CPUS_PER_TASK"
} > "$dest/hardware.txt"
(
    cd "$scratch/OpenGCMC"
    "$benchmark_python" native/build.py --portable --no-openmp
) > "$dest/build.log" 2>&1

case "$group" in
    blind)
        "$benchmark_python" -B "$study/benchmark_blind_policy.py" \
            --protocol "$study/blind_protocol.json" --execute-to "$dest" --cache "$scratch/fields"
        ;;
    blind_budget)
        "$benchmark_python" -B "$study/benchmark_precision.py" \
            "$study/inputs/holdout_co2_uio66.json" --mode delayed --mesh 32 \
            --cache "$scratch/fields" --warmup 20000 --seed-offset 928000 \
            --budgets 240000 480000 960000 1440000 1920000 --output "$dest/holdout-delayed-long.json"
        ;;
    cache)
        "$benchmark_python" -B "$study/benchmark_cache.py" \
            "$study/inputs" "$scratch/fields" "$dest/cache_timing.json"
        ;;
    two_pore)
        "$benchmark_python" -B "$scratch/OpenGCMC/tools/benchmark_two_pore.py" "$dest/two_pore.json"
        ;;
    xylene_low_confirm|holdout_same_candidates)
        "$benchmark_python" -B "$study/run_cpu_suite.py" "$group" \
            --output "$dest" --cache "$scratch/fields"
        ;;
    kernel_controls)
        "$benchmark_python" -B "$scratch/OpenGCMC/tools/benchmark_incremental_kernels.py" \
            --output "$dest/incremental_kernels.json"
        (
            cd "$scratch/OpenGCMC"
            "$benchmark_python" -B -m pytest -q tests/test_delayed_physics.py \
                tests/test_upgrade_sampling.py tests/test_native_longrange_physics.py
        ) > "$dest/physics_tests.log" 2>&1
        ;;
    *)
        "$benchmark_python" -B "$study/run_cpu_suite.py" "$group" \
            --output "$dest" --cache "$scratch/fields" --raspa3 "$raspa_binary"
        ;;
esac
