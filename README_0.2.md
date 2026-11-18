# PoreMC 0.2 — Cache-first flexible-guest Monte Carlo

Independent MIT-licensed CPU/CUDA research implementation. This revision starts from the **27 source/configuration files supplied in `PoreMC_GPU_preliminary_20261117.zip`**, preserved under `baseline/source/`. It does not copy RASPA3/gRASPA implementation code.

## What changed

- Exact cached spatial searches: finer fractional cells, conservative periodic stencils, materialized CSR query-cell → candidate-atom lists, bounded memory, exact fallback.
- Cached pair-parameter rows and bound native queries. Static host data are shared across independent C++ chains.
- Full-support voxel/alias proposals with `1/(V q)` Rosenbluth correction in **both forward growth and old-state retracing**, in Python and C++.
- Guest environments are retained across rejected moves; selected molecules are excluded by original atom indices rather than rebuilding an omitted environment.
- Flexible growth reuses a local coordinate frame across candidates. Whole-molecule energy evaluation batches all external atom energies. Buffers and DSF constants are reused.
- Content-addressed, atomic disk caches for host grids and proposal-energy fields. Temperature reweights a cached energy field without recomputing that field.
- Delayed acceptance reuses already-computed surrogate energy differences.
- Independent energy requests can be coalesced; independent native replicas can share one immutable host cache.
- New CUDA source uses cached CSR lists, resident parameter tables, reusable device workspaces, and an optional device-side Rosenbluth reduction.

The executable native fast path remains single-component LJ/DSF, rigid bodies or tree-flexible molecules. The general Python path retains mixtures, full Ewald and delayed acceptance. The new CUDA source implements pair energies, **not a fully GPU-resident MC state machine**.

## Install and run

Python ≥3.10, NumPy, SciPy and a C++17 compiler are required.

```bash
python -m pip install -e .
python native/build.py
python -m poremc validate examples/flexible_cached_bias.json
python -m poremc run examples/flexible_cached_bias.json -o run_cached
```

The examples use **synthetic probe force fields**, not calibrated IPA/DCM/MOF models. Supply validated molecular definitions, host parameters and fugacities for materials predictions. Coordinates are Å, pair energies are `U/k_B` in K, charges are elementary charge units, and activities are Å⁻³.

On a compiler without OpenMP, use `python native/build.py --no-openmp`. Rebuild on the destination machine; the default build uses `-march=native`. Use `--portable` when distributing a local binary to different CPUs. No fast-math is enabled.

### Exact search bias

`root_bias` in JSON enables a frozen proposal field:

```json
"root_bias": {
  "kind": "host_alias",
  "shape": [16, 16, 16],
  "uniform_fraction": 0.05,
  "cache_dir": "../.poremc_cache"
}
```

This changes **where candidates are proposed**, not the target potential. Every voxel retains nonzero support. Do not remove the proposal-density correction from either insertion or retracing.

### Energy-grid delayed acceptance

`examples/grid_delayed_acceptance.json` evaluates a cached host surrogate first and applies the exact target-energy correction to surviving moves. Grid interpolation alone is not treated as the exact Hamiltonian. The native fast driver does not yet implement this two-stage path; select `driver="python"`.

### Parallel independent chains

```python
from poremc.batching import run_native_replicas
result = run_native_replicas(
    species, potential, cell, temperature, activities,
    seeds=[37, 43, 59, 71], max_workers=4,
    simulation_options={"trials": 8},
    run_options={"steps": 100000, "burnin": 10000,
                 "sample_every": 20, "record_energy": False},
)
```

Use a native `ClassicalPotential` with `algorithm="cached", threads=1`. The host is immutable and shared; each chain owns its particles, random generator and acceptance decisions. Do not modify or close the shared potential while chains are running. Rebuild a potential object after changing its cell, host geometry or force field.

## GPU paths

The **uploaded original CUDA kernel** was executed on an RTX 3070 Ti Laptop GPU and passed 16 checks; raw evidence is preserved verbatim. This revision's cached CUDA kernel was **not compiled with nvcc or executed on a GPU in the authoring environment**. CPU plan export and a host-C++ source-syntax check passed. There are no invented updated GPU timings.

A standalone nvcc harness avoids the CuPy dependency that blocked the original Python adapter test:

```bash
python native/build.py
python tools/run_gpu_benchmark.py --arch sm_86 --repeats 7
```

For another CUDA installation, use `--nvcc /path/to/nvcc` or `POREMC_CUDA_ROOT`. Use the appropriate `--arch` for the device. The script records actual old/new paired energy checks, kernel timings, coordinate-upload/copy-back timings, setup time, temperature and device metadata. It stops timing a case when the pre-case temperature reaches the configured limit. This is a short benchmark, not continuous thermal protection.

CPU-only preparation is independently reproducible:

```bash
python tools/run_gpu_benchmark.py --prepare-only
```

The optional Python CUDA adapter additionally requires CuPy. `examples/flexible_cuda.json` selects it. `CudaEnvironment.energy_device()` avoids host transfers for resident FP64 trial coordinates; returned views are valid only until the next call. A CUDA environment belongs to its construction device and stream and is not a shared concurrent workspace.

## Verification and reproducible experiments

```bash
python -m pip install pytest matplotlib
python -m pytest -q tests --junitxml=benchmarks/pytest.xml
python tools/cuda_syntax_check.py
python tools/run_cpu_benchmarks.py --section kernels
python tools/run_cpu_benchmarks.py --section mc --case flex_host --steps 40000 --skip-build
python tools/run_cpu_benchmarks.py --section mc --case rigid_host --steps 40000 --skip-build
python tools/run_cpu_benchmarks.py --section mc --case dense_guest --steps 20000 --skip-build
python benchmarks/search_worker.py --seed 37
python benchmarks/search_worker.py --seed 43
python benchmarks/search_worker.py --seed 59
python tools/benchmark_parallel.py
python tools/summarize_results.py
```

The fixed-N `dense_guest` case is a translation-only environment-reuse stress test; its particle-number ESS is undefined. Do not call it a complete GCMC benchmark. Root-search benchmarks are exactly soluble **noninteracting voxel-well models**, not real MOF adsorption predictions. All speedup classes have separate denominators and must not be multiplied.

## Measured results in the delivered records

On the recorded AMD EPYC CPU environment, comparing with the uploaded 0.1 source rebuilt using the same optimization flags:

| Experiment | Median gain / observed range |
|---|---:|
| Full flexible-guest GCMC, 8,000 host atoms | 1.21×, 1.14–1.23× |
| Full rigid-guest GCMC, 32,000 host atoms | 1.22×, 1.15–1.32× |
| Fixed-N dense-guest translation stress test | 1.87×, 1.83–1.89× |
| Four independent chains vs sequential execution | 3.24×, 3.06–3.42× |
| 64 coalesced energy requests vs separate calls | 4.00× |
| Full-support search bias, analytic voxel-well model, ESS/s | 38.22×, 31.48–42.73× |

The source of truth is `benchmarks/*.json`, not the rounded table. CPU kernels have a one-time preparation cost; see the report's amortization analysis. The historical laptop GPU timings and current CPU timings were not measured on the same machine.

## Report

`report/report.tex` is the rewritten Chinese technical report. Build with XeLaTeX and BibTeX:

```bash
python report/make_figures.py
python report/build_report.py
```

The figure script uses Arial when installed and its metric-compatible substitute Arimo otherwise; fonts are not bundled. `report/references.bib`, vector PDF/SVG figures, generated numerical tables and raw benchmark JSON are included.

## Source map

`native/kernels.cpp`: CSR and type caches; bound query ABI.
`native/fast_mc.cpp`: retained guest snapshots, batched growth geometry, full C++ loop.
`poremc/proposals.py`, `cbmc.py`: normalized spatial proposals and reversible CBMC.
`poremc/cache.py`, `accelerators.py`: persistent preprocessing and exact-corrected surrogates.
`poremc/batching.py`: request coalescing and independent replicas.
`poremc/cuda.py`, `cached_cuda.cu`: resident CUDA data path.
`tools/run_gpu_benchmark.py`, `gpu_cached_driver.cu`: no-CuPy GPU verification harness.
