# PoreMC/OpenGCMC 0.6 HPC benchmarks

These measurements compare the native CPU sampler on one real charged MOF case. They do not represent a complete isotherm or a GPU result.

## Test case

- Host: 2×2×2 `Ni_TPT_TATBA`, 4,416 framework sites.
- Guest: 12-site charged `p-xylene`.
- Long range: direct Ewald, `kmax=(13,13,13)`, full rectangular mode set.
- Sampling: insert, delete, translate, rotate and full CBMC regrow.
- Execution: one Slurm CPU core and one OpenMP/BLAS thread.

The numerical input is intentionally not included because it contains a project-specific framework and force field. The source tree provides the complete input schema and the benchmark driver.

## Cross-version result

Higher-loading screening, three paired seeds, 3,000 production attempts and 300 burn-in attempts per run:

| Mode | Median moves/s | Paired speedup vs 0.5 |
|---|---:|---:|
| 0.5 exact, full spectrum | 150.38 | 1.00× |
| 0.6 exact, half spectrum | 117.32 | 0.80× |
| 0.6 delayed, 32³ reference grid | 1,136.77 | 7.82× |
| 0.6 delayed, 64³ reference grid | 1,132.06 | 7.85× |

Low-loading run matched approximately to the completed RASPA2 point, using three seeds, 10,000 production attempts and 1,000 burn-in attempts per run:

| Mode | Median moves/s | Paired speedup vs 0.5 |
|---|---:|---:|
| 0.5 exact, full spectrum | 166.50 | 1.00× |
| 0.6 delayed, 32³ reference grid | 2,954.35 | 17.55× |
| 0.6 delayed, 64³ reference grid | 3,591.54 | 21.57× |

The half-spectrum exact path alone was slower on this workload. The production gain comes from fixed-host grid screening, selected complete-molecule residuals and cache reuse, not from reducing the target Ewald mode set.

Raw result: `benchmarks/results/poremc06_hpc_benchmark_nocache_20261118.json`.

## RASPA2 comparison

The same material has completed RASPA2 production points. At `N=5.53` molecules per 2×2×2 supercell, RASPA2 ran 100,000 cycles in 795.43 s with 20 attempts per cycle:

- RASPA2: 125.72 cycles/s, or about 2,514 attempted moves/s.
- 0.6 delayed32: 2,954.35 moves/s, about 1.18× RASPA2.
- 0.6 delayed64: 3,591.54 moves/s, about 1.43× RASPA2.

The PoreMC medians are from three seeds and equilibrated near `N=5`. At the next completed RASPA2 point, `N=17.63`, RASPA2 measured 75.76 cycles/s, or about 1,515 attempted moves/s. Higher-loading throughput remains a separate regime and should be benchmarked at matched occupancy.

Raw result: `benchmarks/results/poremc06_vs_raspa_0p1Pa_20261118.json`.

## Known limitation

On the Tongji HPC runtime, writing a persistent host-grid cache currently crashes in `HostGrid::Impl::save()`. The measurements above disable only the persistent disk cache; host-grid construction, interpolation, CBMC screening, residual correction and the complete move loop remain active.

The cache write path must be fixed and tested with the target Python/libstdc++ runtime before enabling `host_grid_cache` in production.
