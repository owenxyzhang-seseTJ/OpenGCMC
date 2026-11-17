# Figure QA review

Review date: 2026-11-17. Result: **PASS for scientific, preprocessing, code and export QA**. Final delivery remains subject to the separate rendered-image audit.

## 1. raw_data_gate

**PASS.** The two machine-readable sources are `benchmarks/results/raspa3_comparison.json` and `benchmarks/results/graspa_comparison.json`. Each contains three paired timing records. All twelve processed throughput records and six paired ratios were independently compared with the retained numeric fields. The plotting workflow reads the raw JSON and writes processed tables separately; no simulator was executed during this review.

The results quantify the retained attempt-throughput protocols. The gRASPA records describe a short synchronized-launch WSL run on a shared GPU. The retained data do not supply a cross-engine equilibrium ESS comparison. These conditions are stated in the external caption and benchmark document, while the graphic omits test conditions as requested.

## 2. method_traceability

**PASS.** `method_search_packet.md` documents raw sources, protocol analyzers, current implementation mappings, chosen statistics, assumptions and rejected alternatives. `figure_plan.md` records approval of the English execution plan. Subsequent user requests authorize richer architecture glyphs and broader functional colours.

The final architecture specification should be synchronized with the actual 9 × 6.2 in, multicolour open-group drawing; the earlier 7.2 × 4.8 in canvas remains in the planning document. This is a specification-document update, not a scientific redraw requirement.

## 3. preprocessing_check

**PASS.** Source CSVs contain 12 individual rates, six paired ratios and four per-engine summary rows. No observations were dropped. Rates equal scheduled attempts divided by retained sampling seconds. Callouts are medians of paired ratios, independently recomputed as **21.59569809533293** and **30.172145827510796**. Bar lengths are medians of each engine's three rates.

RASPA3 uses 40,000 production attempts per repeat; gRASPA uses 1,000 scheduled attempts. The gRASPA performed insertion/deletion sums are not used as denominators. Geometry, site parameters, replication, guest parameters, temperature and activity agree between the two retained model records. The two experiments use separate linear axes from zero.

## 4. scientific_logic

**PASS.** No fitted trend, inferred observation, confidence interval or significance claim is introduced. Thin connectors pair timing records and do not imply identical trajectories. The external caption states the differing protocols and timer scopes.

Architecture nodes map to current implementation sources. The CSR representation supplies conservative exact candidates, not an interpolated target energy. Frozen full-support alias proposals include the matching density correction in forward and retraced CBMC. Accepted changes invalidate the guest cache; rejected moves reuse it. The C++ single-component LJ/DSF, general Python and optional CUDA energy routes are distinguished. The figure does not represent the complete Monte Carlo loop as GPU resident or assign isolated speed gains to unisolated modules.

## 5. visual_clarity

**PASS at exported dimensions.** Both current PNGs were inspected. Functional teal, coral, purple, green and blue distinguish representations, proposals/growth, state reuse and execution routes. Glyphs are explained as conceptual in the caption. Panel order, arrow direction, software labels and axis units are clear. No visible text collision or clipping was found. The renderer-based text-boundary report records no text overlaps or out-of-canvas text.

The architecture's native physical width is 9 in. For a future 183 mm print placement, retain sufficient text size when resizing the layout; its current smallest 9 pt text would scale with the artwork. README presentation and the supplied full-size vectors retain the present layout.

## 6. code_compliance

**PASS.** The independent invocation of `validate_python_comments.py figures/plot_figures.py` returned `Chinese visual-comment validation passed.` Visual-affecting instructions carry Chinese explanations. The independent SVG/script readability invocation returned `Rendered figure readability audit passed.` Only Python/matplotlib is used for plotting.

## 7. export_check

**PASS.** SVG, PDF and 600 dpi PNG files are present for both figures. SVG has native editable text (22 text elements for performance; 37 for architecture). Both PDFs contain one page and use Arial regular/bold from one font family. Measured minimum PDF font sizes are **8.5 pt** for performance and **9.0 pt** for architecture.

| Figure | Physical dimensions | PNG dimensions |
|---|---|---|
| `opengcmc_performance` | 7.2 × 3.3 in | 4,320 × 1,980 px |
| `opengcmc_architecture` | 9 × 6.2 in | 5,400 × 3,720 px |

## 8. final_image_audit_handoff

Inspect the exact files:

- `docs/assets/opengcmc_performance.png`, `.svg`, `.pdf`
- `docs/assets/opengcmc_architecture.png`, `.svg`, `.pdf`

Verify the rate unit `Move attempts (10³ / s)`, both callouts, three visible points per software, separate zero-origin axes and software-name clearance. In the architecture, check the optional alias branch, `1 / [V q(r)]`, the accept/reject return paths, index exclusion and the Python-to-CUDA energy boundary. Check that neighbourhood circles, candidate-conformation overlays and circuit glyphs remain visually distinct from text. There are no measured chemical structures or molecular images requiring chemical-geometry validation; lattice and conformation drawings are conceptual.

## 9. recommended_fixes

1. Synchronize the final architecture dimensions and richer functional palette in `figure_plan.md`.
2. Complete the separate final rendered-image audit before delivery.

No required numerical, scientific, code-comment or export corrections remain in this review.
