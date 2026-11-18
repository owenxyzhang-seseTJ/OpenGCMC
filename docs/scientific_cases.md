# 科研案例分析 / Scientific Case Analyses

OpenGCMC 已在带电 MOF 混合吸附、恒电势电解质界面、显式水反应自由能和扩散控制电化学四类科研任务中完成端到端计算。这些案例把完整体系输入连接到可复现的数值输出和交叉核对，用于展示软件支持的科研工作流；它们不是新的实验测量，也不构成对未测材料的标定预测。

OpenGCMC has been exercised end to end on charged-MOF mixture adsorption, constant-potential electrolyte interfaces, explicit-water reaction free energies, and diffusion-controlled electrochemistry. These cases connect complete system inputs to reproducible numerical outputs and cross-checks. They demonstrate the scientific workflows supported by the software; they are not new experimental measurements or calibrated predictions for an unmeasured material.

## 1. 带电 IRMOF-1 中的 CO₂/CH₄ 混合吸附 / CO₂/CH₄ Mixture Adsorption in Charged IRMOF-1

本案例采用 RASPA3 官方二元混合物示例中的带电 IRMOF-1 CIF、位点电荷、客体模型和力场约定。框架包含 424 个原子，净电荷约为 $-5.3\times10^{-15}\,e$。在 300 K 下，CO₂ 和 CH₄ 的储库逸度分别为 0.5 bar 和 1.5 bar。计算使用 12 Å 截断、平移 Lennard-Jones 势、无尾部修正和 Ewald 静电。

This case uses the charged IRMOF-1 CIF, site charges, guest models, and force-field conventions from the official RASPA3 binary-mixture example. The framework contains 424 atoms with a net charge of approximately $-5.3\times10^{-15}\,e$. At 300 K, the reservoir fugacities are 0.5 bar for CO₂ and 1.5 bar for CH₄. The calculation uses a 12 Å cutoff, shifted Lennard-Jones interactions, no tail correction, and Ewald electrostatics.

三条独立 OpenGCMC 链分别执行 8,000 次预热尝试和 50,000 次生产尝试。RASPA3 3.1.0 使用 3,000 个预热循环和 20,000 个生产循环。

Three independent OpenGCMC chains each use 8,000 warm-up attempts followed by 50,000 production attempts. RASPA3 3.1.0 is run with 3,000 warm-up cycles and 20,000 production cycles.

| 组分<br>Component | OpenGCMC 分子数/晶胞<br>molecules per cell | RASPA3 分子数/晶胞<br>molecules per cell |
|---|---:|---:|
| CO₂ / CO2 | 2.647，95% 分块区间<br>95% block interval [2.529, 2.768] | 2.615 ± 0.027 |
| CH₄ / CH4 | 4.257，95% 分块区间<br>95% block interval [4.128, 4.388] | 4.246 ± 0.033 |

三条链的 split $\hat R$ 对 CO₂ 和 CH₄ 分别为 1.0061 和 1.0005。以 CO₂、CH₄ 为行，以施加的 $\ln z$ 扰动为列，涨落响应矩阵为：

The three-chain split $\hat R$ values are 1.0061 for CO₂ and 1.0005 for CH₄. With CO₂ and CH₄ ordered as rows and applied $\ln z$ perturbations as columns, the fluctuation response matrix is

$$
\chi =
\begin{bmatrix}
2.642 & 0.069\\
0.069 & 4.301
\end{bmatrix}.
$$

三个偏置窗口的总占据中心分别为 3、8 和 13，每窗保留 4,500 帧。所得自由能图谱在总占据数为 6 时达到最小值。MBAR 结果与独立无偏轨迹的总变差距离为 0.0217，窗口重叠谱隙为 0.435。

Three biased windows centred on total occupancies 3, 8, and 13 retain 4,500 frames each. The resulting free-energy landscape reaches its minimum at total occupancy 6. The MBAR result and an independent unbiased trajectory differ by a total-variation distance of 0.0217; the overlap spectral gap is 0.435.

该案例在同一工作流中验证带电框架 GCMC、二元竞争吸附、响应矩阵分析、独立链收敛诊断和 MBAR 自由能重构。它属于跨引擎和数值一致性案例，不是新的实验等温线。

This case demonstrates charged-framework GCMC, binary competitive adsorption, response-matrix analysis, independent-chain convergence diagnostics, and MBAR free-energy reconstruction in one workflow. It is a cross-engine and numerical-consistency case, not a new experimental isotherm.

## 2. 恒电势界面中的 NaCl/SPC/E 水体系 / NaCl in SPC/E Water at a Constant-Potential Interface

两块理想 Gaussian 电极间距为 16 Å，电势差为 0.10 V。体系在 298.15 K 下包含 62 个刚性 SPC/E 水分子和一对 Na⁺/Cl⁻。离子采用适用于 SPC/E 水的 Joung–Cheatham 参数，水和离子的总电荷由电极补偿。

Two ideal Gaussian electrodes are separated by 16 Å and held at a 0.10 V potential difference. The cell contains 62 rigid SPC/E water molecules and one Na⁺/Cl⁻ pair at 298.15 K. The ions use Joung–Cheatham parameters for SPC/E water, and the electrodes compensate the total water and ion charge.

两个独立种子均执行 8,000 步预热和 16,000 步生产。电极电荷与独立求解的差不超过 $1.1\times10^{-15}\,e$，末态分子能量与完整重算的差不超过 $1.95\times10^{-7}\,K$。

Two independent seeds each use 8,000 warm-up steps and 16,000 production steps. The electrode charge agrees with an independent solve to within $1.1\times10^{-15}\,e$. The final molecular energy agrees with a complete recomputation to within $1.95\times10^{-7}\,K$.

| 物理量<br>Quantity | 种子 1774<br>Seed 1774 | 种子 6231<br>Seed 6231 |
|---|---:|---:|
| Na–O 配位数<br>Na–O coordination number | 5.03 | 5.09 |
| Cl–O 配位数<br>Cl–O coordination number | 6.45 | 6.46 |
| 平均共轭电极电荷<br>Mean conjugated electrode charge | +0.087 e | −0.045 e |

第一壳层配位数采用 Na–O 3.2 Å 和 Cl–O 3.8 Å 的半径。固定的水分子数和盐离子数用于隔离界面模型，不依赖尚未标定的电解质储库；平衡电容估计仍需更长采样。

The first-shell coordination numbers use radii of 3.2 Å for Na–O and 3.8 Å for Cl–O. Fixed water and salt counts isolate the interface model without relying on an uncalibrated electrolyte reservoir. Longer sampling is required for an equilibrium capacitance estimate.

该案例验证显式溶剂离子采样、受约束 Gaussian 电极电荷、局部能量一致性和电化学界面的结构配位分析。

This case demonstrates explicit-solvent ion sampling, constrained Gaussian electrode charge, local energy consistency, and structural coordination analysis for an electrochemical interface.

## 3. 铁氰化物/亚铁氰化物扩散电化学 / Ferricyanide/Ferrocyanide Diffusion Electrochemistry

输运工作流采用 25 °C 的文献扩散系数 $D_{\mathrm{ox}} = 7.26\times10^{-10}\ \mathrm{m^2\,s^{-1}}$ 和 $D_{\mathrm{red}} = 6.67\times10^{-10}\ \mathrm{m^2\,s^{-1}}$，氧化态浓度为 1 mM，平面电极直径为 3 mm。形式电势作为零点，并采用较快的异相动力学检验可逆扩散极限。

The transport workflow uses literature diffusion coefficients at 25 °C, $D_{\mathrm{ox}} = 7.26\times10^{-10}\ \mathrm{m^2\,s^{-1}}$ and $D_{\mathrm{red}} = 6.67\times10^{-10}\ \mathrm{m^2\,s^{-1}}$, 1 mM oxidized species, and a 3 mm planar electrode. The formal potential is used as the zero, and fast heterogeneous kinetics test the reversible diffusion limit.

| 计算<br>Calculation | OpenGCMC 结果<br>result | 解析对照<br>Analytical comparison |
|---|---|---:|
| −0.35 V 阶跃，0.005–0.5 s<br>−0.35 V step, 0.005–0.5 s | log–log 斜率<br>slope −0.50018 | Cottrell 斜率<br>slope −0.5；最大电流差<br>maximum current difference 0.108% |
| 100 mV/s CV 阴极峰<br>cathodic peak | −16.1747 µA | Randles–Ševčík 绝对值<br>magnitude 16.1802 µA |
| 200 mV/s CV 阴极峰<br>cathodic peak | −22.8718 µA | Randles–Ševčík 绝对值<br>magnitude 22.8823 µA |

两档扫描速度的峰电流比为 1.41404，与 $\sqrt{2}$ 一致；两条伏安曲线的峰间距均为 58.5 mV。

The peak-current ratio between the two scan rates is 1.41404, consistent with $\sqrt{2}$. Both voltammograms have a 58.5 mV peak separation.

该案例验证有限体积扩散、物理时间电流积分、计时安培、循环伏安以及与电化学解析极限的对照。

This case demonstrates finite-volume diffusion, physical-time current integration, chronoamperometry, cyclic voltammetry, and validation against analytical electrochemical limits.

## 4. 显式水反应与多状态自由能计算 / Explicit-Water Reactions and Multistate Free-Energy Calculations

反应工作流使用刚性 SPC/E 水、离子和固定分子拓扑，采样质子化、氧化还原和质子耦合电子转移状态。七窗口自由能计算每窗保留 35 帧，窗口重叠谱隙为 0.0249。固定快照的七状态电荷计算与逐状态直接计算的差不超过 $1.17\times10^{-10}\,K$。

The reaction workflow samples protonation, redox, and proton-coupled electron-transfer states with rigid SPC/E water, ions, and a fixed molecular topology. A seven-window free-energy calculation with 35 retained frames per window gives an overlap spectral gap of 0.0249. A fixed-snapshot seven-state charge calculation agrees with direct state-by-state evaluation to within $1.17\times10^{-10}\,K$.

可运行输入明确给出分子拓扑、反应边、储库约定、电荷补偿和校准字段。示例参数用于工作流和数值验证；若要预测物理 pKa 或形式电势，还需要标定的分子模型、收敛的储库活度和独立参考态。

The runnable inputs expose the required molecular topology, reaction edges, reservoir conventions, charge compensation, and calibration fields. The example parameters are controls for workflow and numerical validation; predicting a physical pKa or formal potential requires a calibrated molecular model, converged reservoir activities, and independent reference states.

该案例验证非平衡候选 Monte Carlo、显式水反应采样、多状态窗口构建、MBAR 重叠诊断和局部静电响应计算。

This case demonstrates non-equilibrium candidate Monte Carlo, explicit-water reaction sampling, multistate window construction, MBAR overlap diagnostics, and local electrostatic response evaluation.

## 适用范围 / Interpretation

上述案例覆盖 OpenGCMC 的三类科研计算模式：

- 多孔材料中的平衡吸附与混合物热力学；
- 含显式溶剂的恒电势电解质界面；
- 反应自由能与物理时间电化学电流。

The cases above cover the three scientific modes exposed by OpenGCMC:

- equilibrium adsorption and mixture thermodynamics in porous materials;
- constant-potential electrolyte interfaces with explicit solvent;
- reaction free energies and physical-time electrochemical currents.

现有核对证明实现一致性和解析极限行为。针对具体材料开展预测时，还需为目标科学问题验证结构、力场、电荷模型和储库处理，并完成体系尺寸收敛与采样误差分析。

The retained checks establish implementation consistency and analytical-limit behavior. Material-specific predictive work additionally requires a validated structure, force field, charge model, reservoir treatment, system-size convergence, and sampling-error analysis for the target scientific question.

技术基础与公式见[技术报告](OpenGCMC_Technical_Report.pdf)，可运行输入格式见[示例目录](../examples)。

The technical foundations and equations are described in the [technical report](OpenGCMC_Technical_Report.pdf), and runnable input schemas are in [examples](../examples).
