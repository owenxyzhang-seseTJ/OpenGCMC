# PoreMC 0.5 — native long-range Monte Carlo

PoreMC 0.5 runs complete finite 3D Ewald and optional particle-mesh Ewald inside the **FastGCMC C++ move loop**. It supports one rigid or tree-flexible molecular component, with insertion, deletion, translation, rotation and full CBMC regrowth. Python configures the calculation and receives results; there is no Python callback inside the native sampling loop.

## Implemented in this release

| Capability | Implementation |
|---|---|
| Native EwaldState | Reciprocal modes and coefficients, accepted structure factor, net charge, charge-square sum, empty-host reference |
| Fixed-framework cache | One native host transformation per engine; the Python host snapshot is lazy, avoiding duplicate initialization |
| Incremental moves | Per-molecule contributions; selected-molecule exclusion; charged growth prefixes; acceptance-only updates |
| Correct molecular electrostatics | Rigid primary-pair exclusions; flexible exclusions and 1–4 bare-Coulomb correction; periodic images retained |
| CPU acceleration | Harmonic phase recurrence; candidate parallelism through OpenMP; existing real-space spatial/parameter caches |
| GPU candidate batches | Native CUDA plugin computes ΔS and energy for all candidates; fixed coefficients and work buffers remain on device |
| PME / FFT | Cubic B-splines, native radix-2 FFT, deconvolved influence function, cached grid potential and exact sparse changes of the chosen mesh Hamiltonian |
| Reproducibility | Native long-range diagnostics, readonly production-path energy probes, checkpoint v2 with persistent reciprocal state |

The 0.3 thermodynamics/electrolyte/electrode and 0.4 hydration/reaction/kinetics/transport modules are retained. Their interfaces and examples are documented in [README_0.3.md](README_0.3.md) and [README_0.4.md](README_0.4.md). This release extends the classical FastGCMC engine; electrode-response and reactive samplers keep their separate existing drivers.

## Install and run

Python 3.10+, NumPy, SciPy and a C++17 compiler are needed. Build the library locally; the source archive does not bundle a machine-specific shared library.

```bash
python -m pip install -e ".[test]"
python native/build.py
python -m poremc run examples05/flexible_ewald.json -o run-ewald
python -m poremc run examples05/rigid_pme.json -o run-pme
```

For a portable CPU build use `python native/build.py --portable`; `--no-openmp` builds without OpenMP.

Existing Ewald configurations select the native loop with `sampling.driver = "cpp"`. The relevant JSON fields are:

```json
{
  "potential": {
    "kind": "classical", "cutoff_A": 9.0,
    "backend": "native", "algorithm": "cached",
    "coulomb": "ewald", "alpha": 0.28, "kmax": [5, 5, 5]
  },
  "sampling": {
    "driver": "cpp", "trials": 8,
    "reciprocal_method": "direct", "reciprocal_backend": "cpu"
  }
}
```

This fragment supplements the species, cell, temperature and reservoir fields in the complete examples.

### PME selection

Keep `potential.coulomb = "ewald"` and set `sampling.reciprocal_method = "pme"`, `sampling.pme_mesh = [32,32,32]`. Each mesh dimension is a power of two, at least 8. The spline order is four. The real-space convention remains unchanged. `kmax` is used by direct Ewald; the PME reciprocal approximation is controlled by its mesh.

PME energies are exact changes of the selected discrete mesh Hamiltonian. They approach the periodic Coulomb result as the mesh and Ewald split converge. A readonly candidate batch does not run FFTs; accepted updates refresh the mesh potential. Use `python tools05/benchmark_pme.py` to measure candidate and accepted-update costs for this implementation. Cached direct Ewald is the default.

### CUDA reciprocal batches

On an RTX 30-series machine with CUDA installed:

```bash
python native/build.py --cuda --cuda-arch sm_86
python tools05/validate_cuda_ewald.py --output evidence05/rtx_ewald_validation.json
python -m poremc run examples05/flexible_ewald_cuda.json -o run-cuda
```

`reciprocal_backend="cuda"` requires a functioning CUDA device; `"auto"` selects CUDA when available and otherwise stays in **native C++ CPU**. `long_range_profile` reports the actual backend. `CUDA_VISIBLE_DEVICES` and `POREMC_CUDA_DEVICE` select a device. A custom plugin path can be supplied as `cuda_library`.

The GPU handles direct-Ewald reciprocal candidate batches. Geometric proposals, real-space energy, acceptance and state commits remain in native C++; PME uses CPU FFT. The validation script checks batches of 1/8/32/128/512 candidates, charged prefixes and a complete native sampling run, and measures calls including transfers and synchronization.

## Python interface

```python
from poremc.fast import FastGCMC
from poremc.potentials import ClassicalPotential

potential = ClassicalPotential(
    [molecule], cell, cutoff=9., coulomb="ewald",
    alpha=.28, kmax=(5, 5, 5), algorithm="cached"
)
with FastGCMC(molecule, potential, cell, 300., [activity],
              trials=8, reciprocal_method="direct") as sim:
    result = sim.run(100000, burnin=10000)
    print(sim.long_range_profile)
    sim.save_checkpoint("state.json")
```

- `sim.trial_molecule_energies(trials, exclude=i)` evaluates whole-molecule candidates with molecule `i` removed from the environment.
- `sim.trial_site_energies(prefix, trials, exclude=i)` evaluates the next CBMC atom against the charged prefix.
- Both probes use production C++ code without consuming RNG state or committing coordinates.
- `sim.total_energy()` returns the selected native Hamiltonian relative to the empty framework. For PME this is the mesh energy; `potential.total_energy(...)` remains the independent direct-Ewald reference.
- `FastGCMC.from_checkpoint(path, potential)` restores the persistent long-range state, RNG, coordinates and counters. Exact CPU continuation requires the same Hamiltonian, native binary and thread configuration.

## Physical conventions

Coordinates are Å; charge is in units of e; energies are `U/kB` in kelvin. The periodic electrostatics use 3D conducting (tin-foil) boundary conditions, unshifted `erfc(alpha*r)/r` in real space, reciprocal energy, self energy and uniform-background correction. LJ shifting is a separate setting. Complete charged host/molecule models require `allow_charge_background=True`; neutral molecular models are the default. Intermediate CBMC prefixes include the Q-squared background term even when the complete molecule is neutral.

Primary intramolecular Coulomb exclusions and 1–4 factors subtract the appropriate **bare Coulomb** term. Periodic image interactions remain. Molecule coordinates are internally unwrapped. PME uses cubic assignment, discrete B-spline deconvolution and omitted Nyquist planes to preserve a real symmetric triclinic mesh operator; this convention is documented in the report and independently tested.

## Verification and measured performance

Final CPU regression: **327 passed**, plus **28 parameter subtests**. Independent tests include full-sum Ewald and NumPy FFT references, all five move types, charged prefixes, rigid image interactions, scaled pairs, ideal-gas count statistics, constant-energy single-ion translations and exact CPU checkpoint continuation.

Complete-loop CPU benchmarks against the unmodified **PoreMC 0.4 Python Ewald driver**, using matched physical inputs, the same existing C++ real-space kernel, one thread, three paired seeds, 300 burn-in and 3000 production attempts:

| Task | 0.4 median attempts/s | 0.5 median attempts/s | Median paired speedup |
|---|---:|---:|---:|
| Rigid charged dipole | 381.20 | 16,909.53 | 44.68× |
| Flexible charged four-site chain | 47.23 | 4,782.49 | 101.25× |
| Fixed 128 charged atoms, translations | 64.84 | 21,891.24 | 339.32× |

These are complete attempted-move throughputs with production energy recording disabled. Initialization is outside the timed production region. They compare to the original 0.4 Ewald path and do not estimate RASPA3 speedup or independent-sample efficiency. Source records include all repetitions, acceptance statistics and final energy rechecks. CPU: AMD EPYC 9V74; compiler: g++ 14.2.0. See `evidence05/complete_loop_benchmark.json`.

The CUDA plugin has passed full NVCC 13 compilation/linking/loading. Its device kernels also passed NVRTC 12.9 and PTXAS compilation for sm_86. **No physical GPU execution or new GPU timing was available in this environment.** `evidence05/cuda_compile/` records the exact verification scope; run the provided RTX script for hardware measurements.

## Reproduce

```bash
python -m pytest -q tests tests_legacy tests03 tests04
python tools05/audit_physics.py
(cd baseline_04 && python native/build.py)
python tools05/benchmark_native_ewald.py --repeats 3
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools05/benchmark_pme.py
python report_05/generate_results.py --require-complete
python report_05/build_report.py
```

The LaTeX report connects the grand-canonical measure, CBMC acceptance, Ewald increments, intramolecular corrections, mesh quadratic forms and CUDA data flow to the actual code. It is compiled with XeLaTeX, Noto CJK, Liberation Sans, DejaVu Sans Mono and Latin Modern Math. Numerical tables are generated from the recorded JSON files.

`baseline_04/` preserves the original 0.4 source for the complete-loop comparison. `tests_legacy/PROVENANCE.md` documents recovered legacy tests. Historical 0.2/0.3/0.4 reports and measurements keep their original scope. The independent implementation remains MIT licensed; inherited third-party data retain their source attributions.
