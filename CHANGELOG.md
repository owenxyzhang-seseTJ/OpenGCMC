## 0.6.0 — 固定框架场与两阶段原生采样（2026-11-18）

- 新增原生固定框架网格：按唯一 LJ 类型共享字段，单位电荷电势合并实空间与倒空间作用；惰性 FFT 建表及带输入指纹的磁盘缓存支持跨状态点复用。
- 新增 `sampling_mode="delayed"`：参考势完成 CBMC 生长和回溯，第一阶段通过后仅校正被选中的完整分子；全部接受判断和状态提交保留在 C++ 中。默认 `exact` 与既有调用兼容。
- 陡峭区域默认使用稳定的 Boltzmann 权重插值，改善粗网格能量线性插值对近壁位点的人工排斥；保留 `linear` 和 `cap=0` 精确 fallback 作为研究选项。cap 仅定义参考势。
- 共轭半谱保留原有限 Ewald 目标；新增平移相位更新、原生框架倒空间场 FFT，以及显式的径向 `ewald_k_cutoff` 选项与精度评估工具。
- 精确校正直接消去目标势与参考势共有的 guest 短程项；已接受分子的 exact/reference 局部项分别缓存。
- 新增全支持的网格根位置 alias 偏置，提议密度进入 Rosenbluth 权重；生产前固定概率。
- checkpoint v3 保存逐分子结构因子、长程接受态和阶段计数，支持相位递推后的精确 CPU 延续。
- delayed 校正明确使用原生 CPU；`auto` 保持 CPU，显式 CUDA 给出配置错误。exact 模式保留原生 CUDA 候选批处理。
- 新增真实 JSON 输入的同任务基准、独立物理审计、混合诊断、小型完整示例和 0.6 数学报告。最终结果保存在 `evidence06/`，历史上报、合成 fixture 与真实输入测量分别标识。

## 0.5.0 — Native long-range sampling (2026-11-18)

- Complete finite 3D Ewald in the FastGCMC C++ move loop: real, reciprocal, self, uniform-background and primary intramolecular terms.
- Native EwaldState with fixed-host and per-molecule caches, charged CBMC prefixes, acceptance-only updates and exact-state checkpoints.
- Harmonic phase recurrence reduces repeated trigonometric work while preserving the complete reciprocal mode set.
- Optional native cubic B-spline PME / radix-2 FFT; sparse Green-function candidate increments and accepted-state potential refresh.
- Optional CUDA reciprocal-candidate plugin with persistent device coefficients/buffers, CPU/auto/required-CUDA selection, and a production-path RTX validation script.
- Python/JSON configuration, readonly native trial probes, backend/cache diagnostics, and 0.5 LaTeX report.
- Recovered original legacy regression suite; one obsolete Ewald rejection expectation updated to acceptance. Final tests and performance records are in evidence05/.

# PoreMC 0.4 — 2026-11-17

Implemented explicit rigid SPC/E hydration, fixed-topology constant-pH/redox/PCET GCMC, staged NCMC, reference-state calibration and alchemical-to-kinetics export. Added physical-time BV/Marcus/MHC reaction networks, Gillespie SSA, waveform master equation and supported-electrolyte finite-volume current/CV solver.

Performance: contiguous molecular atom exclusion reuses native environments. New local charge-response cache includes Ewald, slab, primary molecular exclusions and constrained Gaussian-electrode Schur complement. Cache invalidation is tied to accepted atomic state versions.

Validation: 134 inherited + 88 new = 222 tests. Actual explicit-water pilot, alchemical windows, calibrated connectivity check, analytic titration/Nernst/PCET, stochastic currents and diffusion benchmarks saved in results_04. No new GPU or external RASPA speedup claim.

Historical changelog follows.

---

# 0.3.0 — 2026-11-17

- Add mergeable chemical-potential response, constrained neutral-salt coordinates, block uncertainty and joint electrochemical response.
- Add frozen occupancy windows to the molecular engine, MBAR with exact repeated-column compression, overlap checks, per-window block bootstrap, and Gaussian-integrated F(N,Q).
- Add neutral stoichiometric electrolyte GCMC with native short-range evaluation and incremental Ewald. Electrode-compensated single-ion channels are explicit opt-ins.
- Add fixed Gaussian constant-potential electrodes, constrained analytic marginalization, cached field updates, complete capacitance, and voltage/chemical reweighting.
- Add JSON CLI, exact-model-fingerprint restart, examples, complete trajectories and 54 new tests. Retain original 80 passing tests.
- New electrochemical model uses monoatomic primitive ions / constant dielectric; optional new GPU response interface not executed in this CPU environment.

# 0.2.0

Implemented cache-first exact search, native bound queries, retained guest environments with exclusion views, batched flexible growth frames, density-corrected alias proposals in both engines, content-addressed host-field persistence, selected-energy reuse in delayed acceptance, request coalescing, shared-host independent replicas and a new resident CUDA pair-energy path.

The original source and GPU evidence remain in `baseline/source` and `evidence`. The benchmarks compare against that source, not a newly measured RASPA3/gRASPA build. The full codebase remains a research engine rather than complete feature parity with those upstream packages.

New CUDA execution status: CPU preprocessing checked; host-C++ kernel syntax checked; nvcc/device execution pending external hardware verification. The delivered harness performs the missing checks without requiring CuPy.
