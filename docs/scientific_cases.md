# 科研案例分析 / Scientific Case Analyses

OpenGCMC 已在带电 MOF 混合吸附、恒电势电解质界面、显式水反应自由能和扩散控制电化学四类科研任务中完成端到端计算。这些案例把真实晶体、实验参数或标准物理极限连接到可复现的数值输出，用来说明 OpenGCMC 能完成哪些科研工作，以及每个结果为什么值得关注。

OpenGCMC has been exercised end to end on charged-MOF mixture adsorption, constant-potential electrolyte interfaces, explicit-water reaction free energies, and diffusion-controlled electrochemistry. These cases connect real crystals, experimental parameters, or standard physical limits to reproducible numerical outputs. They explain the research work OpenGCMC supports and why each result matters.

这些结果用于验证模型和软件，不是新的实验测量，也不代表未经验证的材料参数已经可以用于定量预测。

The results validate models and software behavior. They are not new experimental measurements and do not imply that unvalidated material parameters are ready for quantitative prediction.

## 先读：几个概念 / Concepts Used Below

**GCMC。** 巨正则 Monte Carlo 在给定温度、体积和外部气体化学势时，反复尝试插入、删除和移动分子，使体系逐渐按平衡概率分布采样。它回答的核心问题是：在指定气体环境下，材料里会有多少分子、它们如何分布。

**GCMC.** Grand canonical Monte Carlo fixes the temperature, volume, and external chemical potentials. It repeatedly attempts insertion, deletion, and displacement so that the system samples its equilibrium distribution. Its central question is how many molecules occupy a material and how they are arranged under a specified gas environment.

**自由能。** 自由能是衡量状态相对稳定性的能量量。自由能越低，该状态通常越稳定、出现概率越高；自由能差决定两种状态、构型或反应通道的相对倾向。

**Free energy.** Free energy measures the relative stability of states. A lower value generally means a more stable and more probable state. Free-energy differences determine the relative preference between configurations, phases, or reaction channels.

**配位数。** 配位数表示一个离子周围平均有多少个最近邻原子或分子。例如 Na–O 配位数说明一个钠离子周围平均有多少个水的氧原子。

**Coordination number.** A coordination number is the average number of nearest-neighbour atoms or molecules around an ion. For example, Na–O coordination reports the average number of water oxygen atoms around a sodium ion.

## 1. 带电 IRMOF-1 中的 CO₂/CH₄ 混合吸附 / CO₂/CH₄ Mixture Adsorption in Charged IRMOF-1

### 实际应用 / Practical Applications

这个场景对应天然气净化、沼气提纯、CO₂/CH₄ 分离、吸附剂筛选和变压吸附工艺设计。实际气体不是单一组分，CO₂ 与 CH₄ 会竞争同一批孔道空间和吸附位点，因此只测纯组分吸附量不足以判断材料能否用于分离。

This case maps to natural-gas upgrading, biogas purification, CO₂/CH₄ separation, adsorbent screening, and pressure-swing adsorption. Real gases contain mixtures, so CO₂ and CH₄ compete for the same pore volume and adsorption sites. Pure-component uptake alone is insufficient to determine whether a material can separate the mixture.

### 为什么需要计算 / Why the Calculation Matters

IRMOF-1 是带原子电荷的多孔框架。静电作用、孔道几何和两种气体之间的竞争会共同决定吸附量。固定实验条件做混合气 GCMC，可以在同一套物理模型下同时回答三个问题：每种气体吸附多少、一种气体的化学势变化会怎样影响另一种气体、总占据数不同时哪个状态更稳定。

IRMOF-1 is a porous framework with atomic charges. Electrostatics, pore geometry, and competition between the two guests jointly determine uptake. A fixed-condition mixture GCMC calculation answers three questions within one physical model: how much of each gas is adsorbed, how a chemical-potential change in one component affects the other, and which total-occupancy state is more stable.

### 计算了什么 / What Was Calculated

计算采用 RASPA3 官方二元混合物示例中的带电 IRMOF-1 CIF、位点电荷、客体模型和力场约定。框架包含 424 个原子，净电荷约为 −5.3 × 10⁻¹⁵ e。在 300 K 下，CO₂ 和 CH₄ 的储库逸度分别为 0.5 bar 和 1.5 bar。相互作用使用 12 Å 截断、平移 Lennard-Jones 势、无尾部修正和 Ewald 静电。

The calculation uses the charged IRMOF-1 CIF, site charges, guest models, and force-field conventions from the official RASPA3 binary-mixture example. The framework contains 424 atoms with a net charge of approximately −5.3 × 10⁻¹⁵ e. At 300 K, the reservoir fugacities are 0.5 bar for CO₂ and 1.5 bar for CH₄. Interactions use a 12 Å cutoff, shifted Lennard-Jones potentials, no tail correction, and Ewald electrostatics.

三条独立 OpenGCMC 链分别执行 8,000 次预热尝试和 50,000 次生产尝试。RASPA3 3.1.0 使用 3,000 个预热循环和 20,000 个生产循环。独立链用于检查结果是否依赖某一条随机轨迹，RASPA3 用于跨软件核对。

Three independent OpenGCMC chains each use 8,000 warm-up attempts followed by 50,000 production attempts. RASPA3 3.1.0 is run with 3,000 warm-up cycles and 20,000 production cycles. The independent chains test whether the result depends on one random trajectory, while RASPA3 provides a cross-software check.

### 结果 / Results

| 组分<br>Component | OpenGCMC 分子数/晶胞<br>molecules per cell | RASPA3 分子数/晶胞<br>molecules per cell |
|---|---:|---:|
| CO₂ / CO2 | 2.647，95% 分块区间<br>95% block interval [2.529, 2.768] | 2.615 ± 0.027 |
| CH₄ / CH4 | 4.257，95% 分块区间<br>95% block interval [4.128, 4.388] | 4.246 ± 0.033 |

三条链的 split R̂ 对 CO₂ 和 CH₄ 分别为 1.0061 和 1.0005。R̂ 接近 1 说明不同随机链给出了彼此一致的结果，而不是各自停留在差异很大的状态。

The three-chain split R̂ values are 1.0061 for CO₂ and 1.0005 for CH₄. Values close to one indicate that independent random chains agree rather than remaining trapped in substantially different states.

以 CO₂、CH₄ 为行，以施加的 ln z 扰动为列，涨落响应矩阵为：

With CO₂ and CH₄ ordered as rows and applied ln z perturbations as columns, the fluctuation response matrix is:

| χ | CO₂ 的 ln z 扰动<br>CO₂ perturbation | CH₄ 的 ln z 扰动<br>CH₄ perturbation |
|---|---:|---:|
| CO₂ 吸附量响应<br>CO₂ uptake response | 2.642 | 0.069 |
| CH₄ 吸附量响应<br>CH₄ uptake response | 0.069 | 4.301 |

对角线元素描述每种气体对自身化学势变化的响应，非对角线元素描述一种气体对另一种气体的耦合。这里非对角项较小，说明在本算例中直接响应占主导，交叉耦合存在但较弱。

Diagonal entries describe each gas responding to its own chemical-potential change. Off-diagonal entries describe coupling between the two components. The off-diagonal terms are small here, so direct responses dominate in this case while cross-coupling remains weak.

三个偏置窗口的总占据中心分别为 3、8 和 13，每窗保留 4,500 帧。所得自由能图谱在总占据数为 6 时达到最小值。MBAR 结果与独立无偏轨迹的总变差距离为 0.0217，窗口重叠谱隙为 0.435。

Three biased windows centred on total occupancies 3, 8, and 13 retain 4,500 frames each. The resulting free-energy landscape reaches its minimum at total occupancy 6. The MBAR result and an independent unbiased trajectory differ by a total-variation distance of 0.0217; the overlap spectral gap is 0.435.

### 前因后果 / What the Results Mean

OpenGCMC 与 RASPA3 的平均吸附量接近，说明两套程序在同一结构、力场和储库约定下得到一致的平衡结果。响应矩阵进一步说明混合物中两组分如何相互影响，自由能图谱则说明总占据数 6 比相邻占据状态更有利。把这些信息放在一起，研究者可以判断材料在不同气体组成和压力下的容量、竞争行为和分离趋势。

OpenGCMC and RASPA3 give similar mean uptake, showing that the two programs agree under the same structure, force field, and reservoir convention. The response matrix shows how the two components influence one another, while the free-energy landscape shows that total occupancy 6 is more favourable than neighbouring occupancy states. Together, these results help assess capacity, competition, and separation trends under different gas compositions and pressures.

### 科研价值 / Research Value

这个工作流可用于比较不同 MOF 的 CO₂/CH₄ 选择性、评估操作压力范围、识别强吸附组分造成的竞争、研究框架电荷对吸附的影响，并把吸附量与自由能信息一起用于吸附剂筛选。

This workflow can compare CO₂/CH₄ selectivity across MOFs, evaluate operating-pressure ranges, identify competition caused by strongly adsorbed components, study the effect of framework charge, and combine uptake with free-energy information for adsorbent screening.

### 适用边界 / Scope

该案例属于跨软件和数值一致性验证，不是新的实验等温线。针对真实材料做预测时，还需要验证 CIF 的完整配位与周期结构、力场参数、原子电荷、框架柔性、储库逸度以及有限尺寸和采样误差。

This case is a cross-software and numerical-consistency study, not a new experimental isotherm. Predictive work on a real material still requires validated periodic structure and coordination, force-field parameters, atomic charges, framework flexibility, reservoir fugacity, finite-size behavior, and sampling uncertainty.

## 2. 恒电势界面中的 NaCl/SPC/E 水体系 / NaCl in SPC/E Water at a Constant-Potential Interface

### 实际应用 / Practical Applications

这个场景对应超级电容器、电池电极界面、电容去离子、电催化、电化学传感器和腐蚀过程。真实电化学界面不是真空中的静态电荷面，而是水分子、离子和电极电荷相互重排形成的双电层。离子的水合结构、局部浓度和电极响应会直接影响电容、离子传输和反应速率。

This case maps to supercapacitors, battery interfaces, capacitive deionization, electrocatalysis, electrochemical sensors, and corrosion. A real electrochemical interface is not a static charged surface in vacuum. Water, ions, and electrode charge reorganize into an electric double layer. Ion hydration, local concentration, and electrode response affect capacitance, transport, and reaction rates.

### 为什么需要计算 / Why the Calculation Matters

恒电势条件下，电极电荷会随溶液构型改变，不能事先固定为一个常数。显式水又决定离子的第一配位壳和局部介电环境。该算例同时处理水、Na⁺、Cl⁻、Ewald 静电和 Gaussian 电极电荷，用来检查电荷约束、能量计算和界面结构是否自洽。

Under constant potential, electrode charge changes with the solution configuration and cannot be fixed in advance. Explicit water determines the ion coordination shell and local dielectric environment. This case simultaneously treats water, Na⁺, Cl⁻, Ewald electrostatics, and Gaussian electrode charge to test charge constraints, energy evaluation, and interface structure.

### 计算了什么 / What Was Calculated

两块理想 Gaussian 电极间距为 16 Å，电势差为 0.10 V。体系在 298.15 K 下包含 62 个刚性 SPC/E 水分子和一对 Na⁺/Cl⁻。离子采用适用于 SPC/E 水的 Joung–Cheatham 参数，水和离子的总电荷由电极补偿。

Two ideal Gaussian electrodes are separated by 16 Å and held at a 0.10 V potential difference. The cell contains 62 rigid SPC/E water molecules and one Na⁺/Cl⁻ pair at 298.15 K. The ions use Joung–Cheatham parameters for SPC/E water, and the electrodes compensate the total water and ion charge.

两个独立种子均执行 8,000 步预热和 16,000 步生产。计算记录电极电荷、总能量、Na–O 配位、Cl–O 配位和离子的水合壳层。

Two independent seeds each use 8,000 warm-up steps and 16,000 production steps. The calculation records electrode charge, total energy, Na–O coordination, Cl–O coordination, and ion hydration shells.

### 结果 / Results

电极电荷与独立求解的差不超过 1.1 × 10⁻¹⁵ e，末态分子能量与完整重算的差不超过 1.95 × 10⁻⁷ K。这说明电极电荷和能量账本在采样过程中保持一致。

The electrode charge agrees with an independent solve to within 1.1 × 10⁻¹⁵ e, and the final molecular energy agrees with a complete recomputation to within 1.95 × 10⁻⁷ K. Electrode charge and the energy bookkeeping therefore remain consistent during sampling.

| 物理量<br>Quantity | 种子 1774<br>Seed 1774 | 种子 6231<br>Seed 6231 |
|---|---:|---:|
| Na–O 配位数<br>Na–O coordination number | 5.03 | 5.09 |
| Cl–O 配位数<br>Cl–O coordination number | 6.45 | 6.46 |
| 平均共轭电极电荷<br>Mean conjugated electrode charge | +0.087 e | −0.045 e |

第一壳层配位数采用 Na–O 3.2 Å 和 Cl–O 3.8 Å 的半径。两个种子给出相近的配位数，说明离子周围的平均水合结构稳定。电极平均电荷随轨迹变化，反映界面构型与电极电势之间的动态耦合。

The first-shell coordination numbers use radii of 3.2 Å for Na–O and 3.8 Å for Cl–O. Both seeds give similar coordination numbers, showing a stable average hydration structure around the ions. The mean electrode charge varies with the trajectory and reflects dynamic coupling between interface configuration and electrode potential.

### 前因后果 / What the Results Mean

如果电极电荷求解错误，局部电场、离子分布和总能量会同时偏离。该算例中电荷与能量都能被独立重算，配位数也与物理上合理的离子水合壳一致。因此，这个工作流可以可靠地追踪“施加电压 → 电极电荷改变 → 水和离子重排 → 局部结构与能量改变”的因果链。

If the electrode-charge solve were wrong, the local field, ion distribution, and total energy would all drift. Here, both charge and energy can be independently recomputed, and the coordination numbers are consistent with physically reasonable ion hydration shells. The workflow can therefore track the causal chain from applied voltage to electrode-charge change, water and ion reorganization, and the resulting local structure and energy.

### 科研价值 / Research Value

该流程可用于研究双电层结构、电位诱导离子吸附、水合壳变化、不同盐浓度下的界面响应，以及电极材料和电解质对电容与离子传输的影响。

The workflow can study electric-double-layer structure, potential-induced ion adsorption, changes in hydration shells, interface response at different salt concentrations, and how electrode materials and electrolytes affect capacitance and ion transport.

### 适用边界 / Scope

固定的水分子数和盐离子数用于隔离界面模型，不依赖尚未标定的电解质储库。平衡电容估计仍需更长采样。该案例验证界面物理和数值一致性，不能直接替代特定电极材料的实验电容测量。

Fixed water and salt counts isolate the interface model without relying on an uncalibrated electrolyte reservoir. Equilibrium capacitance still requires longer sampling. This case validates interface physics and numerical consistency; it does not replace experimental capacitance measurements for a specific electrode material.

## 3. 铁氰化物/亚铁氰化物扩散电化学 / Ferricyanide/Ferrocyanide Diffusion Electrochemistry

### 实际应用 / Practical Applications

铁氰化物/亚铁氰化物是电化学中常用的可逆氧化还原对，可用于电化学传感器、氧化还原液流电池、电子转移研究、电极标定和分析化学。工业传感器和电池模拟经常需要从电位程序预测电流随时间或扫描电压的变化。

The ferricyanide/ferrocyanide couple is a common reversible redox system used in electrochemical sensors, redox-flow batteries, electron-transfer studies, electrode calibration, and analytical chemistry. Industrial sensor and battery models often need to predict current versus time or scan voltage from a prescribed potential program.

### 为什么需要计算 / Why the Calculation Matters

测到的电流同时受扩散、电极面积、浓度、扩散系数和异相电子转移速率影响。为了检查数值求解器是否可靠，可以先采用快速动力学和已知扩散系数，把体系推到可逆扩散极限，再与 Cottrell 和 Randles–Ševčík 两个教科书解析结果比较。

Measured current depends on diffusion, electrode area, concentration, diffusion coefficients, and heterogeneous electron-transfer rates. To test the numerical solver, fast kinetics and known diffusion coefficients can first place the system in the reversible diffusion limit, where results can be compared with the textbook Cottrell and Randles–Ševčík expressions.

### 计算了什么 / What Was Calculated

输运工作流采用 25 °C 的文献扩散系数 D<sub>ox</sub> = 7.26 × 10⁻¹⁰ m² s⁻¹ 和 D<sub>red</sub> = 6.67 × 10⁻¹⁰ m² s⁻¹，氧化态浓度为 1 mM，平面电极直径为 3 mm。形式电势作为零点，并采用较快的异相动力学检验可逆扩散极限。

The transport workflow uses literature diffusion coefficients at 25 °C, D<sub>ox</sub> = 7.26 × 10⁻¹⁰ m² s⁻¹ and D<sub>red</sub> = 6.67 × 10⁻¹⁰ m² s⁻¹, 1 mM oxidized species, and a 3 mm planar electrode. The formal potential is used as the zero, and fast heterogeneous kinetics test the reversible diffusion limit.

计算包括一次 −0.35 V 电位阶跃下的计时安培，以及 100 mV/s 和 200 mV/s 两档循环伏安。有限体积模型在物理时间中积分扩散方程和电极边界。

The calculation includes chronoamperometry after a −0.35 V step and cyclic voltammetry at 100 mV/s and 200 mV/s. A finite-volume model integrates the diffusion equation and electrode boundary in physical time.

### 结果 / Results

| 计算<br>Calculation | OpenGCMC 结果<br>result | 解析对照<br>Analytical comparison |
|---|---|---:|
| −0.35 V 阶跃，0.005–0.5 s<br>−0.35 V step, 0.005–0.5 s | log–log 斜率<br>slope −0.50018 | Cottrell 斜率<br>slope −0.5；最大电流差<br>maximum current difference 0.108% |
| 100 mV/s CV 阴极峰<br>cathodic peak | −16.1747 µA | Randles–Ševčík 绝对值<br>magnitude 16.1802 µA |
| 200 mV/s CV 阴极峰<br>cathodic peak | −22.8718 µA | Randles–Ševčík 绝对值<br>magnitude 22.8823 µA |

两档扫描速度的峰电流比为 1.41404，与 √2 一致；两条伏安曲线的峰间距均为 58.5 mV。

The peak-current ratio between the two scan rates is 1.41404, consistent with √2. Both voltammograms have a 58.5 mV peak separation.

### 前因后果 / What the Results Mean

计时安培在接近 Cottrell 的 −0.5 幂律时，说明扩散层增长和电流衰减符合理论。循环伏安峰电流接近 Randles–Ševčík 预测，说明浓度、面积、扫描速度和扩散系数被正确组合。扫描速度提高到两倍后，峰电流增加约 √2 倍，也与可逆扩散理论一致。

A chronoamperometric response approaching the Cottrell −0.5 power law shows that diffusion-layer growth and current decay follow theory. Cyclic-voltammetry peak currents close to the Randles–Ševčík prediction show that concentration, area, scan rate, and diffusion coefficients are combined correctly. Doubling the scan rate increases the peak current by approximately √2, again consistent with reversible diffusion theory.

### 科研价值 / Research Value

该工作流可以预测计时安培、循环伏安和扩散限制电流，用于传感器响应分析、电极面积评估、扩散系数反演、电化学池设计，以及把反应网络给出的法拉第电流接到连续介质传质模型。

The workflow can predict chronoamperometry, cyclic voltammetry, and diffusion-limited current. Applications include sensor-response analysis, electrode-area assessment, diffusivity inference, electrochemical-cell design, and coupling reaction-network Faradaic current to continuum mass transport.

### 适用边界 / Scope

该案例使用较快的异相动力学检验可逆扩散极限，并不代表所有真实电极都处于可逆状态。实际预测需要电极几何、有效面积、扩散系数、电子转移速率、双电层和溶液条件。

This case uses fast heterogeneous kinetics to test the reversible diffusion limit; real electrodes are not always reversible. Predictive work requires electrode geometry, effective area, diffusion coefficients, electron-transfer rates, double-layer treatment, and solution conditions.

## 4. 显式水反应与多状态自由能计算 / Explicit-Water Reactions and Multistate Free-Energy Calculations

### 实际应用 / Practical Applications

这个场景对应电催化、质子耦合电子转移、氧化还原活性材料、电池界面、pH 依赖吸附和腐蚀反应。对于 CO₂ 还原、析氢、氧还原或质子传导，反应状态是否有利、在给定 pH 和电位下出现多少，决定了后续速率和选择性。

This case maps to electrocatalysis, proton-coupled electron transfer, redox-active materials, battery interfaces, pH-dependent adsorption, and corrosion. For CO₂ reduction, hydrogen evolution, oxygen reduction, or proton transport, the favourability and population of reaction states at a given pH and potential influence subsequent rates and selectivity.

### 为什么需要计算 / Why the Calculation Matters

反应不能只看静态结构。质子化、氧化还原和质子耦合电子转移会改变电荷、局部电场和水的排列，而这些变化又反过来影响反应态稳定性。OpenGCMC 用显式 SPC/E 水、固定分子拓扑和反应状态能量描述这种相互影响。

Reaction behaviour cannot be inferred from a static structure alone. Protonation, redox, and proton-coupled electron transfer change charge, local fields, and water organization, and those changes feed back into reaction-state stability. OpenGCMC describes this coupling with explicit SPC/E water, fixed molecular topology, and reaction-state energies.

### 计算了什么 / What Was Calculated

反应工作流采样质子化、氧化还原和质子耦合电子转移状态。七窗口自由能计算每窗保留 35 帧，用于连接相邻热力学状态。固定快照的七状态电荷计算则与逐状态直接计算比较，用来检查局部静电响应缓存是否准确。

The reaction workflow samples protonation, redox, and proton-coupled electron-transfer states. A seven-window free-energy calculation retains 35 frames per window to connect neighbouring thermodynamic states. A fixed-snapshot seven-state charge calculation is compared with direct state-by-state evaluation to test the local electrostatic response cache.

### 结果 / Results

七窗口自由能计算的窗口重叠谱隙为 0.0249。谱隙大于零说明相邻窗口存在共同采样的构型区域，可以用于估计状态间自由能差。固定快照的七状态电荷计算与逐状态直接计算的差不超过 1.17 × 10⁻¹⁰ K。

The seven-window free-energy calculation gives an overlap spectral gap of 0.0249. A positive gap indicates that neighbouring windows share sampled configuration regions and can be used to estimate state-to-state free-energy differences. The fixed-snapshot seven-state charge calculation agrees with direct state-by-state evaluation to within 1.17 × 10⁻¹⁰ K.

随包示例中的反应提议接受率较低，因此该示例主要用于验证工作流、能量差和窗口重叠，不能据此宣称反应态总体已经充分收敛。

The reaction-proposal acceptance rate in the supplied example is low. The example therefore primarily validates the workflow, energy differences, and window overlap; it does not establish a converged reaction-state population.

### 前因后果 / What the Results Mean

反应态的自由能差决定平衡时各状态的比例，pH 和电位再通过质子、电子储库改变这些自由能。窗口重叠说明不同状态之间能够被可靠连接，局部电荷响应一致则说明显式水环境中的静电能被正确计算。经过独立校准后，这条链路可以继续计算 pKa、形式电势、反应速率和电流。

Reaction-state free-energy differences determine equilibrium populations, while pH and potential shift those free energies through proton and electron reservoirs. Window overlap shows that the states can be connected reliably, and consistent local charge response shows that electrostatics in explicit water are evaluated correctly. After independent calibration, this chain can support pKa, formal potential, rate, and current calculations.

### 科研价值 / Research Value

该流程可用于比较不同反应路径、分析溶剂和电荷如何稳定中间态、研究 pH/电位依赖的覆盖度，并把反应自由能连接到动力学网络和物理时间电流模型。

The workflow can compare reaction pathways, analyze how solvent and charge stabilize intermediate states, study pH- and potential-dependent coverages, and connect reaction free energies to kinetic networks and physical-time current models.

### 适用边界 / Scope

示例参数用于工作流和数值验证。若要预测物理 pKa 或形式电势，还需要标定的分子模型、收敛的储库活度、独立参考态、足够的窗口采样和统计误差分析。

The example parameters support workflow and numerical validation. Predicting a physical pKa or formal potential requires a calibrated molecular model, converged reservoir activities, independent reference states, sufficient window sampling, and statistical-error analysis.

## 总体适用范围 / Overall Interpretation

上述案例覆盖 OpenGCMC 的三类科研计算模式：

- 多孔材料中的平衡吸附与混合物热力学；
- 含显式溶剂的恒电势电解质界面；
- 反应自由能与物理时间电化学电流。

The cases above cover the three scientific modes exposed by OpenGCMC:

- equilibrium adsorption and mixture thermodynamics in porous materials;
- constant-potential electrolyte interfaces with explicit solvent;
- reaction free energies and physical-time electrochemical currents.

这些案例证明 OpenGCMC 能把结构、力场、储库、采样、自由能和电流连接成一个完整工作流，并通过独立链、跨软件比较、能量重算和解析极限检查实现一致性。针对具体材料发表预测结果时，还需要完成模型标定、体系尺寸收敛和采样误差分析。

These cases show that OpenGCMC can connect structure, force field, reservoir, sampling, free energy, and current in one workflow, with consistency checks from independent chains, cross-software comparison, energy recomputation, and analytical limits. Publishing predictions for a specific material still requires model calibration, system-size convergence, and sampling-error analysis.

技术基础与公式见[技术报告](OpenGCMC_Technical_Report.pdf)，可运行输入格式见[示例目录](../examples)。

The technical foundations and equations are described in the [technical report](OpenGCMC_Technical_Report.pdf), and runnable input schemas are in [examples](../examples).
