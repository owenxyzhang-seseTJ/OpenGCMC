# OpenGCMC figure method-search packet

Method status: **READY_FOR_PREPROCESSING**  
Execution status: **CONFIRMED**  
Inspection date: **2026-11-17**  
Figure types: paired throughput comparisons and implementation architecture schematic.  
System: controlled neutral LJ probe in a periodic CuBTC geometry; general OpenGCMC implementation capabilities are described separately.

## Sources searched and inspected

This packet uses the supplied machine-readable benchmark results, their method scripts, the retained reference-engine inputs, and current implementation source. These provide the processing method required for this software benchmark; no extra live literature query or new simulation is necessary. The public style reference informs composition, not scientific data or algorithm validation.

| ID | Source | Processing/design rule supported |
|---|---|---|
| M1 | `benchmarks/results/raspa3_comparison.json` | Three current paired records, 40,000 production attempts each, reference/OpenGCMC seconds and reported rate ratios; nominal model and timer scope |
| M2 | `tools/analyze_raspa3_comparison.py`; `tools/native_benchmark.py`; `benchmarks/results/raspa3_raw/seed_*/simulation.json` and retained RASPA3 outputs | Production C ABI sampler; actual conventional swap counts; 10,000 warm-up attempts excluded; no CBMC swap attempts in the reference comparison; final energy recomputation outside timing |
| M3 | `benchmarks/results/graspa_comparison.json`; `tools/analyze_graspa_comparison.py` | Three paired current-source timing records; exactly 1,000 scheduled attempts; matched empty starts, K = 1, no warm-up; documented synchronized-launch/shared-device conditions |
| M4 | `benchmarks/results/graspa_raw/timed_seed_*/simulation.input`, `console.log` and `graspa_timings.json`; supplied source-loop audit for [official gRASPA source](https://github.com/snurr-group/gRASPA/tree/e6da4e376f19047df6cb85b9d20685d1bf0777e8) | `UseMaxStep yes`, `MaxStepPerCycle 1`, 1,000 production cycles; “Work took” timing occurs after preparation and before final energy checks. The sum of reported performed insertion/deletion moves omits empty-deletion self-transitions and cannot define throughput |
| M5 | `native/kernels.cpp`: `Environment`, `bind`, `BoundQuery`, `poremc_environment_export_plan` | Periodic fractional-cell indexing, CSR-style neighbour offsets/indices, exact distance/cutoff evaluation and bound mixed-parameter rows. The cache is not an interpolated energy field |
| M6 | `poremc/proposals.py`: `VoxelProposal`, `build_host_proposal`; `poremc/cbmc.py`: `grow`, `retrace`; `native/fast_mc.cpp`: root proposal, `grow` | Frozen strictly positive voxel probabilities; alias sampling; relative density Vq(r); root weights include its inverse in both forward growth and retracing |
| M7 | `native/fast_mc.cpp`: `cached_environment`, `external`, `step`; `poremc/fast.py`: `FastGCMC` | Accepted-state guest environment reuse, index-range exclusion, lazy rebuild after accepted moves, single-component native CPU LJ/DSF sampling scope |
| M8 | `poremc/batching.py`: `evaluate_requests`, `run_native_replicas` | Independent trial requests batched only against matching immutable snapshots; independent chains share a fixed host and own guest state/RNG/counters |
| M9 | `poremc/engine.py`: `GCMC`, `_accept_candidate`; `poremc/potentials.py`; `poremc/electrostatics.py`; `poremc/accelerators.py` | General Python route for mixtures/Ewald and optional exact target correction after a surrogate-stage decision |
| M10 | `poremc/cuda.py`: `CudaEnvironment`, `_bind`, `energy_device`, `energy`; `poremc/cached_cuda.cu` | CUDA energy evaluation with resident neighbour plans/parameter bindings and reusable workspace. This is not a GPU-resident whole-MC implementation |
| S1 | [ChenLiu-1996/figures4papers](https://github.com/ChenLiu-1996/figures4papers), supplied Python examples | Direct comparison labels, restrained comparison palettes and readable numerical emphasis; adapt visual patterns only |
| S2 | Current scientific-figure and Nature-figure workflow guidance | Claim/evidence contract, white background, readable single-family type, explicit sample/statistics definitions and editable vector exports |

Source M4's upstream source-loop/timer audit is supplied with the benchmark context. Local inputs and analyzer guards were inspected; this planning pass did not execute gRASPA, RASPA3 or OpenGCMC sampling.

## Chosen preprocessing method

For replicate i in each experiment:

- Reference rate: Rref,i = Nref,i / tref,i, in scheduled attempts s⁻¹.
- OpenGCMC rate: Ropen,i = Nopen,i / topen,i.
- Paired rate ratio: si = Ropen,i / Rref,i. Because both counts match within each pair, si = tref,i / topen,i.
- Bar length: median of the three rates for that engine.
- Main numerical callout: median of the three paired ratios si.
- Descriptive spread: minimum and maximum si, written as a range in the caption/source data.

Panel a maps N to `actual_attempts`, tref to `raspa_production_seconds`, and topen to `opengcmc_production_seconds`. Panel b uses the two explicit `*_scheduled_attempts` fields and the two `*_sampling_seconds` fields. The `steps` field is only a consistency check. Never use gRASPA performed-move sums, accepted counts or final particle counts as N.

Keep attempt rates in base units in source CSVs; divide by 1,000 only for the displayed axis unit. Use deterministic point offsets that do not encode another variable. Do not pool experiments, resample runs, exclude outliers or normalize to a single cross-system reference. Preserve full precision and round only displayed text.

## Rationale

Scheduled attempts are an observable, consistently defined workload for both programs, including legitimate self-transitions. Dividing by retained production-loop seconds answers the limited throughput question supported by the supplied records. Three visible timing runs and a median show the actual observations with little statistical machinery. Pair-first ratios preserve the provided matched-protocol comparisons. Independent axes separate the two experiments. Following the user's latest instruction, the graphic contains software names, rate units, replicate points and paired-median ratios; full context is supplied by the external captions and `docs/benchmarks.md`.

## Expected descriptive values from raw summaries

These numbers are read from the supplied raw summaries and are not new fitted or simulated results. Recalculate them from retained times after confirmation.

| Experiment | Repeats | Median paired ratio | Observed paired-ratio range |
|---|---:|---:|---:|
| RASPA3 production comparison | 3 | 21.59569809533293 | 20.058154491359698–23.27100260172907 |
| gRASPA short-run comparison | 3 | 30.172145827510796 | 28.340968419747455–30.24961392314041 |

The displayed callouts are 21.60× and 30.17×. Do not relabel the second number as a GPU-kernel speedup or either number as an ESS gain.

## Assumptions and limits

- Both comparisons use a controlled neutral, shifted-LJ probe (σ = 3 Å, ε/kB = 5 K), 4,992-site CuBTC geometry, cutoff 12 Å, nominal 300 K and activity 3/V; no tail corrections. The RASPA3 comparison retains the analyzer's small temperature-unit matching adjustment in its raw metadata.
- RASPA3 and OpenGCMC each use 40,000 conventional single-site insertion/deletion production attempts after 10,000 warm-up attempts. Equal move-selection probabilities and K = 1 apply. RASPA3 property defaults remain in the timed scope; OpenGCMC samples count every 20 attempts.
- The gRASPA comparison is a **short-run** measurement: 1,000 scheduled single-site attempts, initially empty, K = 1, no warm-up, on the same **WSL** system. **CUDA_LAUNCH_BLOCKING=1** was required, an unrelated task was active on the **shared GPU**, and recorded temperatures were **81–82 °C**. These conditions remain complete in the external caption and `docs/benchmarks.md`, not inside the graphic; README text immediately beside the image may remain brief.
- gRASPA's stock “Work took” timer includes stock production statistics after preparation and before final energy checks. OpenGCMC's native production timer includes count sampling every 20 attempts. Setup is excluded. These scopes are recorded rather than silently made identical by subtracting unavailable components.
- The two experiments ran on different systems and have different run lengths/initialization protocols. Their absolute rates are not pooled or ranked across panels.
- Existing energy checks support the reported checked configurations. The gRASPA restart snapshot is cycle 0, not its final state; only one retained snapshot is nonempty. Do not turn those checks into a general equilibrium-accuracy or distribution-equivalence claim.
- This figure does not establish equilibrium mixing, effective sample size, long-run GPU performance, performance for real charged adsorption systems, or speed contributions of individual modules. Optional alias proposals, parallel chains, Ewald and delayed acceptance are implementation capabilities rather than isolated causes of these measured ratios.
- CSR expansion and parameter-table caching have bounded-memory fallback paths in the implementation; the diagram represents reusable geometry and query semantics, not an unconditional promise that every possible host/table is fully materialized.
- Specific tested CPU/GPU model names, version-history narratives and build-process narratives must not enter the figures or README-facing captions.

## Rejected alternatives

| Alternative | Reason rejected |
|---|---|
| One shared absolute rate axis for both experiments | Different systems and protocols; invites an unsupported cross-system comparison |
| Accepted or “performed” moves per second | Acceptance depends on the trajectory; gRASPA omits empty-deletion self-transitions from reported performed sums |
| Ratio of median bars | Differs in general from the median paired ratio; would obscure the chosen matched comparison |
| Mean ± SEM, bootstrap CI, significance stars | Three timing repeats do not justify the proposed inferential display; all observations can be shown directly |
| Log bars, broken axes, 3D effects or decorative speed gauges | The measured two-group comparisons remain readable on simple zero-origin linear axes |
| Calling the result “GPU speedup” or “ESS acceleration” | The measured OpenGCMC path is native CPU sampling; the experiment does not measure a GPU kernel or equilibrium ESS |
| Architecture arrows from a surrogate directly to unconditional acceptance | Optional delayed acceptance requires the implemented target correction |
| Drawing all MC steps inside CUDA | Current CUDA functionality is an energy backend |

## Required and produced data fields

Required quantitative fields are listed in the figure plan. No essential field is currently missing for descriptive plotting. The exact attempt denominator, retained native times, seed IDs, model metadata, engine identity and condition metadata are mandatory. Stop if they disagree rather than infer replacements.

Produced files after confirmation: `performance_replicates.csv`, `performance_pairs.csv`, `performance_summary.csv`, `preprocessing_log.json`, `architecture_nodes.csv`, and `architecture_edges.csv` under `figures/source_data/`. The log records source-relative paths, formulas, units, validation results and zero dropped rows. Raw records remain unchanged.

## Execution gate

The user approved the concrete plan on 2026-11-17 with “按此方案绘制”. The English plan and this execution packet are **CONFIRMED**: two independent performance panels with 21.60× / 30.17× callouts and no in-figure test conditions, plus a white-background teal three-layer architecture schematic. The parent workflow may now dispatch preprocessing and Python plotting. Export to `docs/assets/opengcmc_performance.{svg,png,pdf}` and `docs/assets/opengcmc_architecture.{svg,png,pdf}`; source data remain in `figures/source_data/`.
