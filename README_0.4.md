# PoreMC 0.4

Explicit hydration, constant pH, Faradaic reaction states and physical-time currents, built on PoreMC 0.3's cached molecular Monte Carlo, Ewald and constant-potential electrode infrastructure.

## What is implemented

| Layer | Implemented capabilities |
|---|---|
| Explicit solvent | Rigid SPC/E water, generic rigid multisite molecules, whole-molecule translation/rotation/insertion/deletion, candidate-biased grand exchange, primary-cell intramolecular exclusions, full finite Ewald/slab, hydration statistics |
| Reactive equilibrium | Discrete protonation/redox/PCET states; dummy proton topology; calibrated pKa/E0 offsets; constant-potential compensating electrodes; symmetric staged NCMC with reversible solvent propagation and full rollback |
| Free energies | Fixed-composition alchemical windows, cross-state energy matrix, inherited MBAR, overlap diagnostics, endpoint BAR/MBAR and block bootstrap, actual export to kinetic state tables |
| Acceleration | Inherited native C++ kernels and spatial/parameter caches; contiguous molecule self-exclusion; exact local charge-gradient/Hessian cache with electrode Schur complement and stale-state checks |
| Kinetics | Butler–Volmer, Marcus, Marcus–Hush–Chidsey and bounded Metropolis rate laws; paired detailed balance; Gillespie SSA; continuous-waveform master equation; Faradaic/displacement current bookkeeping |
| Transport | SI-unit 1D diffusion, finite-volume electrode boundary, buffered proton-coupled reaction, concentration profiles, chronoamperometry and cyclic voltammetry |

The molecular equilibrium chain has **no physical clock**. Physical current uses separately supplied rates/diffusivities, not MC moves per second. The atomistic-to-kinetics connection is a calibrated coarse-grained workflow; it is not an on-the-fly reactive ab-initio trajectory.

This version treats fixed-topology chemical states and rigid molecular solvent. It does not discover arbitrary bond-breaking reactions, water autoionization/Grotthuss transfer, an evolving unbuffered pH field, or electronic structures. The original tree-flexible guest engine remains available separately; arbitrary flexible reactive molecules are not silently accepted by the new rigid-molecule interface.

## Install and test

Python >= 3.10, NumPy, SciPy and a C++17 compiler are required. The reference CPU backend is available without the native library. GPU execution is not required for the new functionality.

```bash
python -m pip install -e . pytest matplotlib
python native/build.py
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m pytest tests tests03 tests04 -q
```

Release validation: **222 tests passed** (134 inherited and 88 new). See `results_04/all_tests.xml`. `results_04/clean_build_verification.json` records the independent clean-source/build/test verification. Tests contain analytic distribution, full-versus-incremental energy, electrostatic gauge, charge conservation, NCMC rollback, detailed balance, MHC independent quadrature, current and diffusion checks.

## Run explicit hydration and constant pH

```bash
python -m poremc.reactive_cli run \
  examples04/hydrated_calibrated_pipeline.json -o run_water
python -m poremc.reactive_cli run \
  examples04/pcet_four_state.json -o run_pcet
python -m poremc.reactive_cli run \
  examples04/dummy_proton_titration.json -o run_dummy
```

The hydration examples use generic X+/Y− ions and illustrative site parameters. The “calibrated pipeline” offsets are from short **same-fixture** seven-window trials. They are not transferable experimental pKa/E0 fits. The production demonstration accepted only 2 of 395 reaction proposals (including burn-in); it is a pipeline/numerical consistency check, not a converged titration or rate prediction. The uncalibrated demonstration is also retained and accepted 0 of 409 reaction proposals.

`examples04/pcet_four_state.json` is a separate analytic control, not a real aqueous chemical model.

### Input conventions

- Atomistic coordinates: Å; energy and LJ epsilon: K (`U/kB`); charge: units of `e`; electrode potential: V.
- Explicit SPC/E uses dielectric = 1. Reusing the primitive-model dielectric ~78 would double-screen the solvent and is rejected.
- `log_activities` uses a 1 Å length reference. Match the same force field, cutoff, electrostatic and water energy-zero convention in bulk calibration; molarity is not activity.
- Protonation state fields: `bound_protons`, `bound_electrons`, `intrinsic_kelvin`, `intrinsic_dimensionless`, `log_degeneracy`.
- `ReactionEdge` requires `Δcharge = Δbound_protons − Δbound_electrons`. Interpolated bridge states are not physical output states.
- Acid helper: `reference_deprotonation_kelvin` is for **HA → A−**. A window run in A− → HA order gives the opposite sign.
- Redox helper: `reference_reduction_kelvin` is for Ox → Red. This term is subtracted before adding `e(E−E0)`.
- Periodic molecules remain unwrapped internally. Restricted slit axes reject atoms outside their physical accessible region.
- The rigid-molecule path requires all internal distances below the nonbonded cutoff.

## Calibrate free energies and connect to rates

```bash
python -m poremc.reactive_cli calibrate \
  examples04/hydrated_redox_calibration.json -o redox_windows
```

`poremc.calibration.network_from_alchemical` maps conditional physical endpoint free energies and electrode charges to kinetic states and explicitly supplied reaction edges. The actual linkage using release trajectories is in `tools04/collect_results.py::kinetic_curves`. The resulting table is `results_04/hydration_to_current_bridge.json`.

The window sampler has fixed solvent/ion composition. It does not claim to estimate the full open-solvent grand free energy. Endpoint/MBAR overlap is saved, but short-window overlap does not prove equilibration. Use longer independent windows and standard-state reference systems for research predictions.

```bash
python -m poremc.reactive_cli kinetics \
  examples04/two_contact_current.json -o current_events
python -m poremc.reactive_cli kinetics \
  examples04/voltammetry.json -o site_cv
```

SSA uses seconds and one independent reactive site; state populations are **time-weighted**. Positive current is anodic. Total external current is Faradaic current plus electrode charging; missing state-charge information is not reported as zero displacement current. The user supplies the number of sites or area to obtain macroscopic current or current density. Marcus's prefactor and BV/MHC's zero-driving rate have different meanings and are documented in code/report.

## Physical diffusion/current model

```bash
python -m poremc.reactive_cli transport \
  examples04/diffusion_chronoamperometry.json -o chrono
python -m poremc.reactive_cli transport \
  examples04/diffusion_voltammetry.json -o diffusion_cv
```

`poremc.transport.DiffusionRedox` uses SI units: m, s, mol/m³, m/s for heterogeneous `k0`, V and A. It assumes buffered pH and supporting electrolyte, and omits electromigration/convection. Surface half-cell resistance is eliminated analytically. Closed-cell mass and open-cell steady flux are tested. An ideal voltage step's capacitive impulse is distinct from the finite-time current trace.

## Results and performance

- pH and Nernst analytic probability checks: maximum absolute error 0.00608 and 0.00691, respectively.
- Four-state PCET analytic check: maximum absolute probability error 0.00241.
- Diffusion chronoamperometry: maximum Cottrell error **0.137%**, specified interval 0.005–0.5 s.
- Local charge-state batch, 33 states, same frozen hydrated configuration: **43.24×** median paired CPU speedup, including cache construction, versus repeated full trial evaluation. Max energy difference ~1.02e-10 K.
- This is not total GCMC speedup and not a RASPA3 comparison. New 0.4 modules have no GPU benchmark; inherited CUDA source is retained unchanged.

Timing varies with hardware. Source results preserve all repetitions, hardware/thread settings and actual trajectories.

## Reproduce and report

```bash
bash tools04/reproduce_all.sh
# Or regenerate plots and PDF from the delivered results:
python report_04/make_figures.py
python report_04/build_report.py
```

Full reproduction overwrites `results_04/` and updates same-fixture demonstration offsets. Report compilation requires XeLaTeX/CTeX, BibTeX or bibtex8, Noto CJK fonts and Arial or Arimo; fonts are not redistributed. LaTeX source, nine editable PDF/SVG figures and their plotting script are in `report_04/`.

`README_0.3.md` and `README_0.2.md` preserve historical documentation. Original 0.2/0.3 tests, examples and validation records are retained; their dates, hardware and scopes must not be relabeled as 0.4 performance.

## License and provenance

MIT for the independent implementation. Upstream example inputs and third-party data retain their attribution/license files. No upstream RASPA/gRASPA implementation is copied into the new modules. The 0.3 source hashes are in `results_04/baseline_03_hashes.json`; the release manifest lists all packaged source/data files. Native shared libraries are excluded from the source archive and rebuilt locally.
