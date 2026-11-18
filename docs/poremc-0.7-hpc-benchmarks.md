# OpenGCMC/PoreMC 0.7 HPC benchmarks

These measurements use the real `Ni_TPT_TATBA` framework and a charged 12-site `p-xylene` guest in a 2x2x2 supercell. The framework contains 4,416 sites. Runs use one Slurm CPU core, one OpenMP/BLAS thread and the complete rectangular direct-Ewald target with `kmax=(13,13,13)`.

## Cache repair

The 0.7 cache audit compared the same 128 production moves with the shared host grid disabled, written on the cold path and loaded on the warm path.

- Cold write completed without the 0.6 segmentation fault.
- Off, cold and warm runs produced identical count, sampled-energy, final-position and acceptance traces.
- The cache file is 9,437,416 bytes for this mesh.
- Construction time was 19.27 s without the cache, 18.37 s for the cold build and 0.24 s for the warm read.

Raw result: `benchmarks/results/poremc07_cache_test_hpc_20261118.json`.

## 0.6 to 0.7 throughput

Low-loading run, three paired seeds, 5,000 production attempts and 500 burn-in attempts per run:

| Mode | 0.6 moves/s | 0.7 moves/s | Paired median speedup |
|---|---:|---:|---:|
| exact | 153.44 | 286.32 | 1.87x |
| delayed32 | 3,376.75 | 6,411.99 | 1.87x |
| delayed64 | 3,935.39 | 6,563.99 | 1.62x |

The exact-array count traces matched between 0.6 and 0.7 for all three seeds.

Higher-loading run, three paired seeds, 3,000 production attempts and 300 burn-in attempts per run:

| Mode | 0.6 moves/s | 0.7 moves/s | Paired median speedup |
|---|---:|---:|---:|
| delayed32 | 1,327.24 | 1,678.72 | 1.27x |
| delayed64 | 1,292.59 | 1,535.84 | 1.19x |

All paired count traces matched. The gain comes from the CPU reciprocal contraction and contiguous framework interpolation while preserving the finite-Ewald target.

Raw results:

- `benchmarks/results/poremc07_low_benchmark_hpc_20261118.json`
- `benchmarks/results/poremc07_high_benchmark_hpc_20261118.json`

## Scope

The measurements validate the CPU path. The optional CUDA tensor plugin is compiled and numerically cross-checked in `evidence07`, but these HPC throughput values do not measure device-resident sampling.
