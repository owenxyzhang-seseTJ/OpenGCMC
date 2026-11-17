# OpenGCMC figure execution plan

Status: **CONFIRMED**  
Backend: **Python / matplotlib** (the supplied benchmark and figure workflow is Python).  
Scope: two standalone English figures for the README and publication use. The user approved this concrete plan on 2026-11-17 with “按此方案绘制”. This planning pass has not performed preprocessing or rendering; the plotting workflow may now execute the approved contract.

## 1. core_goal

**Performance:** make the measured throughput difference immediately visible while keeping the two experiments distinct. The first reading should be “21.60× versus RASPA3” and “30.17× versus gRASPA,” followed by the individual timing repeats. These are protocol-specific attempt-rate ratios; their full conditions belong in the external captions and `docs/benchmarks.md`, not inside the graphic.

**Architecture:** explain how OpenGCMC moves repeated work out of trial evaluation: reuse fixed host geometry and mixed-parameter rows, batch trials, correct nonuniform proposals, and reuse the accepted guest state. Show the actual sampling and energy-backend boundaries.

Figure lanes: a two-panel quantitative comparison and a schematic-led architecture figure. No historical progression, build narrative, specific CPU/GPU model names, or unsupported per-component speed claims appear in either figure.

## 2. method_search_packet

The companion [method_search_packet.md](method_search_packet.md) records the inspected benchmark JSON, generation methods, local input protocols, implementation sources, and public style reference. Its method status is `READY_FOR_PREPROCESSING`; figure execution remains gated by this plan's confirmation status.

Use throughput = scheduled move attempts / measured production seconds. Summarize each engine with its median throughput and show all three repeats. Compute each paired ratio first and then its median; the prominent ratio is **not** the ratio of the two median bars. No fitting, smoothing, inferred samples, confidence intervals, or significance tests are proposed.

## 3. preprocessing_plan

Read raw inputs without modifying them:

| Input | Required fields | Use |
|---|---|---|
| `benchmarks/results/raspa3_comparison.json` | replicate `seed`, `actual_attempts`, `raspa_production_seconds`, `opengcmc_production_seconds`, `raspa_version`; model and timing metadata | Panel a: 40,000 production attempts per repeat after 10,000 warm-up attempts |
| `benchmarks/results/graspa_comparison.json` | replicate `seed`, `gRASPA_scheduled_attempts`, `opengcmc_scheduled_attempts`, `gRASPA_sampling_seconds`, `opengcmc_sampling_seconds`; protocol, conditions and source metadata | Panel b: 1,000 scheduled attempts per repeat, empty start, no warm-up, K = 1 |
| `poremc/` and `native/` implementation files listed in the method packet | classes, functions and supported execution scopes | Architecture nodes, arrows and limits |

After confirmation:

1. Validate three unique seed pairs in each experiment, positive finite times, matching attempt counts within each pair, and the stated timing scope. Keep full precision internally.
2. Produce one long table with `experiment`, `seed`, `engine`, `scheduled_attempts`, `sampling_seconds`, `attempts_per_second`, `backend`, and `protocol_id`. Do not include hardware model strings in figure-facing tables.
3. Produce one paired table with `experiment`, `seed`, `reference_seconds`, `opengcmc_seconds`, and `paired_rate_ratio`. Recalculate ratios from the times and compare numerically with the stored summaries using an ordinary tolerance.
4. Produce the per-engine median rate and per-experiment median/minimum/maximum paired ratio. Ranges are descriptive and are not confidence intervals. Use all repeats; drop none.
5. Save processed tables separately as `figures/source_data/performance_replicates.csv`, `performance_pairs.csv`, and `performance_summary.csv`. Save units, field mapping and any validation failures in `preprocessing_log.json`.
6. Save the architecture mapping as `architecture_nodes.csv` and `architecture_edges.csv`, with a code-source column. No scientific simulation is required.

Block execution if an attempt count or timer scope disagrees with the supplied protocol, a repeat is missing, an engine used a different model, or an architecture claim lacks a current implementation source. Do not substitute the gRASPA “Insertion/Deletion Performed” sums for scheduled attempts: empty-deletion self-transitions are omitted from those sums.

## 4. recommended_architecture

**Figure 1 — measured performance:** two equal-width panels with vertical median bars from zero, three visible replicate points per bar, and an unboxed paired-median ratio above each plot. A vertical arrow between the two bars rises from the reference median to the OpenGCMC median, and the ratio sits at its tip. No explanatory subtitle text. Use independent linear y-axes; never place the two experiments on one absolute throughput axis. Following the user's latest instruction, omit all test conditions from the graphic: no run lengths, warm-up, short-run qualifier, platform, launch synchronization, shared-device note, or CPU/GPU suffix. Keep that information complete in the external captions and `docs/benchmarks.md`; README text immediately beside the image should be brief.

**Figure 2 — algorithm architecture:** three wide, vertically arranged regions. The middle sampling loop is the hero region. A top region supplies reusable representations; the bottom region separates execution routes. User-requested richer glyphs and functional colours are part of the final contract. Abstract cell/bin and table glyphs are conceptual diagrams, not a reconstructed chemical structure or measured adsorption-energy map.

Do not add radar plots, 3D bars, broken axes, benchmark leaderboards, or extra validation panels. Two comparisons and one implementation diagram are sufficient for the available claim.

## 5. panel_map

### Figure 1: `opengcmc_performance`

Canvas: **7.2 × 3.3 in** (183 × 84 mm), white background. Two matched panels with a generous central gutter and independent scales. Each panel allocates an unboxed ratio lane above the data. There is no condition lane or protocol subtitle.

| Panel | Heading / order | Encoding and axes | Evidence role |
|---|---|---|---|
| a | Direct bar labels only: `RASPA3` then `OpenGCMC` | Horizontal median-rate bars, three slightly offset points per bar. X label: `Move attempts (10³ s⁻¹)`; zero origin; nice ticks based on that panel's maximum. | Actual RASPA3 3.1.0 production timing versus the current native cached OpenGCMC loop on the same macOS system; conditions are external to the graphic |
| b | Direct bar labels only: `gRASPA` then `OpenGCMC` | Same mark geometry and axis wording; its **own** x limits and ticks. | Actual gRASPA timing versus the current native cached OpenGCMC loop on the same WSL system under the documented short-run conditions; conditions are external to the graphic |

The numerical callouts are `21.60×` and `30.17×`, each followed by `median paired rate ratio`. Retain exact values 21.59569809533293 and 30.172145827510796 in source data. Beyond panel letters, axis ticks, software names, throughput units and the paired-median callout, add no in-figure text. The two axes must remain visually separate; neither chart suggests an absolute cross-system ranking.

### Figure 2: `opengcmc_architecture`

Canvas: **9 × 6.2 in** (229 × 157 mm). Three open visual groups use enlarged conceptual glyphs: periodic neighbourhood, sparse CSR strips, bound rows, voxel probabilities, trial conformations, forward/retrace weights, accepted-state reuse and execution icons. Keep panel letters outside each region's upper-left boundary.

| Region | Content and exact display labels | Evidence role |
|---|---|---|
| a — reusable inputs | `Periodic host + site parameters` → `Fractional-cell CSR neighbours` + `Bound mixed-parameter rows`. A separate light optional branch: `Frozen voxel alias proposal`, `q(r) > 0`. A small fan-out glyph reads `Shared host · independent chains`. | Identify fixed geometry, exact cutoff queries and reusable parameter work; host sharing does not share RNG or guest state |
| b — hero sampling loop | Left to right: `Trial batch` → `Exact energies` → `Forward + retrace CBMC` → `Accept / reject`. Put `1 / [V q(r)]` below CBMC, connected only to the optional nonuniform-proposal branch. Below the loop, `Accepted guest state` feeds `Guest environment`; short return arrows read `reject: reuse` and `accept: rebuild on next query`. Small label at the energy query: `Exclude moved molecule by index`. | Explain batching, matched proposal corrections, and accepted-state reuse without depicting an approximate field as the acceptance target |
| c — execution routes | Three aligned modules: `C++ native sampler` / `Single component · LJ / DSF`; `Python general sampler` / `Mixtures · Ewald · optional delayed acceptance`; `CUDA energy backend` / `Resident tables · reusable workspace`. A call arrow connects the Python route to CUDA energy evaluation. | Distinguish native CPU sampling, general Python sampling and the optional GPU energy backend |

The diagram must not connect the C++ native sampler to a full GPU-resident MC loop. The independent-chain glyph represents separate native chains sharing fixed host data. Name target correction in the Python route and describe its optional status in the caption, with no claim of measured acceleration.

## 6. visual_rules

Use the public figures4papers reference for readable comparisons, direct labels and restrained emphasis, while applying the current evidence and export contract. Do not reuse its data, biomedical labels, images or decorative geometry.

| Meaning | Colour | Treatment |
|---|---|---|
| OpenGCMC / exact evaluation | teal `#4D9498` | Pale teal bars/modules; darker teal points, edges and key text |
| Reference engines | slate `#9CA8B5` | Pale slate bars with slate points; identity also written directly |
| Reusable host representation | aqua `#D8EBEB` | Quiet module fill, dark teal outline |
| Optional proposal/correction | lavender `#DCD7EC` | Quiet fill and dashed optional connectors |
| Trial growth | coral `#C97762` | Warm candidate-conformation glyphs and pale coral region |
| Accepted state and reuse | green `#518872` | Decision, committed-state glyph and reuse path |
| Execution backends | blue `#527DAC` | Native, Python and CUDA energy icons |
| Text and flow | ink `#26333B` | High-contrast labels and thin arrows |

White backgrounds; no plot gridlines. The quantitative bar fills are **solid** to preserve clear length encoding; this is the explicit exception to gradient-first defaults. Schematic modules may use a very subtle low-contrast gradient, provided SVG/PDF remain vector and editable; use the same named colours throughout.

Check Arial availability once at execution. Select **Arial alone**, or **DejaVu Sans alone** if Arial is absent; match mathtext to that family. Minimum visible text **8.5 pt**, body/ticks 9 pt, axis/module headings 10 pt, panel letters 11 pt bold, performance ratio 19–21 pt. Increase canvas height or simplify labels if necessary; never shrink the minimum text. Use native subscripts/superscripts and the multiplication sign. Keep fonts and stroke widths consistent in both figures.

Bars approximately 0.42 category units high; replicate points approximately 3.5–4 pt with a clear outline and fixed non-random offsets. Three thin connectors pair the observed timing points. Median values are encoded by bar lengths; only paired ratio values receive numeric callouts. Thin axes 0.7–0.8 pt and flow arrows 0.8–1.0 pt. Lowercase panel letters, direct group labels, no framed annotation boxes, no detached legend when direct labels suffice.

## 7. annotation_plan

Performance: show all three points per bar; define the sample count, points and median bars in the external caption. Do not add an in-figure n label, individual timing values, test conditions or per-engine median number labels. No significance stars, inferential error bars, energy-equivalence badges or ESS labels. Report the ratio ranges in the caption/source data only. Pairing means matched protocol and seed identifier, not identical random trajectories.

Architecture: solid arrows encode implemented data/call flow; dashed arrows encode optional routes. Label the optional alias route directly; explain dashed optional connectors in the caption. Cache-reuse arrows must clearly return to the same accepted state. Do not add numeric speed labels to individual modules because these comparisons do not isolate their contributions.

## 8. caption_draft

**Figure 1 | Measured move-attempt throughput under two controlled protocols.** Bars show median throughput and points show all three paired timing runs. **a,** RASPA3 3.1.0 and the OpenGCMC native cached CPU sampler: 40,000 production insertion/deletion attempts after 10,000 warm-up attempts on the same macOS system. **b,** gRASPA and the OpenGCMC native cached CPU sampler: 1,000 scheduled insertion/deletion attempts from an empty state, without warm-up, on the same WSL system. gRASPA used synchronized CUDA launches on a shared GPU (81–82 °C). The median paired rate ratios are 21.60× (range 20.06–23.27×) and 30.17× (28.34–30.25×), respectively. Both use K = 1 and a controlled neutral, shifted-LJ probe in a 4,992-site CuBTC geometry (σ = 3 Å, ε/kB = 5 K, cutoff 12 Å, nominal T = 300 K, activity 3/V). Setup and final energy checks are excluded; stock reference statistics and OpenGCMC count sampling every 20 attempts remain inside their respective timing scopes. The panels have independent scales and measure protocol-specific attempt rates, not equilibrium effective sampling rates.

**Figure 2 | Reuse and execution structure of OpenGCMC.** **a,** Fixed periodic host geometry is represented by fractional-cell neighbour tables and bound mixed-parameter rows; independent chains share host data while keeping their guest state, RNG and counters separate. An optional frozen full-support voxel proposal supplies nonuniform root positions. **b,** Batched exact-energy evaluation supports forward and retraced CBMC weights; nonuniform proposals include the matching 1/[Vq(r)] factor. The native sampler reuses the accepted-state guest environment across rejected moves and excludes the moved molecule by index. **c,** Native C++ CPU sampling supports a single LJ/DSF component, while the Python sampler supports general mixtures, Ewald electrostatics and optional delayed acceptance. CUDA accelerates energy evaluation with resident tables and reusable workspaces. These are implementation capabilities; the performance comparisons use conventional K = 1 insertion/deletion and do not isolate individual algorithm contributions.

## 9. python_plotting_brief

The plan is **CONFIRMED**. Execute the following approved contract:

1. Create `figures/plot_figures.py` with separated `load_and_validate`, `write_source_data`, `draw_performance`, and `draw_architecture` functions. Use matplotlib-native text, patches and arrows; no rasterized screenshot or AI-generated schematic.
2. Implement the field mappings, formulas and medians in the method packet. Keep benchmark raw data read-only; do not rerun simulation or benchmark engines.
3. Produce the two layouts above. Make geometry and labels editable at the top of the script. Every line affecting appearance, placement, typography, axes or export must end with a useful Chinese comment.
4. Export `docs/assets/opengcmc_performance.{svg,png,pdf}` and `docs/assets/opengcmc_architecture.{svg,png,pdf}`. Processed source data remain under `figures/source_data/`. Use `svg.fonttype='none'`, `pdf.fonttype=42`, white backgrounds and 600 dpi PNG. Preserve the intended physical dimensions and use an explicit layout rectangle so tight cropping cannot change them silently.
5. Run the Chinese-comment validator, inspect the exported SVG text/font sizes and physical dimensions, and render the exact final SVG/PDF in Python. Inspect at 183 mm width and at README display size. Check every panel label, axis unit, point, formula, arrow, optional branch, and caption correspondence. Fix clipping or overlap before delivery.
6. Complete an independent rendered-image audit. Deliver only after the final named files pass; report the processed tables and script alongside them.

## 10. confirmation_status

**CONFIRMED**

User approval recorded on 2026-11-17: “按此方案绘制”. Confirmed content: two independent performance panels highlighting **21.60× / 30.17×**, no test conditions inside the graphic, and a **white-background three-layer architecture schematic**. Full gRASPA short-run / WSL synchronized / shared-GPU conditions remain in the external caption and method packet and must be carried into `docs/benchmarks.md` by the documentation owner. Outputs are fixed at `docs/assets/opengcmc_performance.{svg,png,pdf}` and `docs/assets/opengcmc_architecture.{svg,png,pdf}`; source data remain in `figures/source_data/`. The plotter may now preprocess and draw under this contract.

Final user refinements: richer conceptual glyphs, functional teal/coral/purple/green/blue colours, and direct or single-curve connectors with fewer right-angle bends. These refinements preserve the approved content and quantitative encodings.
