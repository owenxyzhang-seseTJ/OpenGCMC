# OpenGCMC 其他科研场景基准

六个场景在 Apple M5 工作站上使用 OpenGCMC 源码 `8085910` 运行。固定预算比较使用同一物理模型、相同尝试数和配对种子。墙钟加速取各配对时间比的中位数。采样诊断采用秩归一化分链 R̂ < 1.01，ESS/s 表示每秒有效样本数。能量差以 E/k<sub>B</sub> 的 K 值报告。[源码工作区](../../..) · [MFI、内核与 MCP 基准](../cpu_kernels_and_workflows.md)

## 一、带电 IRMOF-1 的 CO₂/CH₄ 竞争吸附

应用是含 CO₂ 气体的甲烷净化。两组分在同一孔道内竞争，模型同时检验框架与客体的静电作用。使用 RASPA3 的带电框架及力场输入，300 K，CO₂ 与 CH₄ 逸度分别为 0.5 和 1.5 bar。两个计算路径使用同一个有限 Ewald 势。每种路径六条链，预热 20,000 次并生产 150,000 次。[完整输入](inputs/irmof_mixture.json) · [RASPA3](inputs/irmof_raspa3)

| 路径 | CO₂ 分子/晶胞 | CH₄ 分子/晶胞 | CO₂/CH₄ 选择性 | 尝试/s 中位数 |
|---|---:|---:|---:|---:|
| 原生直接求能加客体重建 | 2.662 | 4.255 | 1.877 | 47,872 |
| 原生两阶段加局部更新 | 2.611 | 4.282 | 1.829 | 167,433 |

两条路径的固定预算墙钟相差 **3.43 倍**，两阶段路径的选择性 95% 区间为 1.77–1.89。两条路径都达到本组载量和选择性的诊断门槛。以六链墙钟时间之和计算，CO₂ 载量的 ESS/s 从 44.0 增至 415.6。单独替换客体索引的墙钟比为 1.02。这里的 3.43 倍包含候选生成和接受规则两处变化。在共享参考候选的补充实验中，接受规则单独带来的首达标加速为 M5 1.22 倍、Intel 1.28 倍。[固定预算配对](IRMOF1_吸附配对.json) · [同候选双平台对照](../paired/README.md)

## 二、Ni_TPT_TATBA 中的对二甲苯

该场景检验痕量芳香分子的插入采样，以及较高占据时的长程静电和客体相互作用成本。使用 4,416 位点带电框架、12 位点带部分电荷的刚体对二甲苯。温度为 298.15 K，截断半径为 12 Å，直接 Ewald 的三轴整数模式上限均为 13。低工况的四条链各生产 16,000 次尝试，高活度工况各生产 52,000 次。[0.1 Pa 输入](inputs/nitpt_pxylene_0p1Pa.json) · [高活度输入](inputs/nitpt_pxylene_higher_activity.json)

| 工况 | 直接求能 | 两阶段 32³ | 两阶段 64³ |
|---|---:|---:|---:|
| 0.1 Pa 尝试/s 中位数 | 762 | 16,186 | 19,672 |
| 0.1 Pa 配对墙钟加速 | 1.00 | 20.89 | 21.17 |
| 高活度尝试/s 中位数 | — | 2,342 | 2,233 |

墙钟计时包含求解器准备与预热。0.1 Pa 工况中，两张网格均在首条链构建框架场，随后三条复用。高活度取 z=10⁻⁸ Å⁻³，两张网格均复用已构建的场。高活度下 32³ 的尝试速率略高，两张网格的末态总能重算差均低于 1.8×10⁻⁷ K，载量 R̂ 分别为 1.720 和 1.803。[固定预算结果及逐种子计时](xylene_benchmark.json)

低活度另做四套独立化学势梯队，覆盖 0.006–0.18 Pa 的七个点，每点生产 160,000 次尝试。0.10 与 0.18 Pa 间的交换接受率为 6%–21%，四套梯队的完整端点往返数均为零。0.1 Pa 的载量 R̂ 在 32³ 独立短链中为 1.518，在梯队长链中为 1.261，两者均超出诊断门槛。两组轨迹的平均占据分别为 7.42 和 15.96 分子/超胞。约 21 倍的结果对应等尝试数计时，达到平衡载量精度所需的墙钟时间仍不确定。[七点梯队、占据轨迹索引与交换统计](xylene_exchange_benchmark.json)

## 三、TON 孔道内柔性正丁烷

反式正丁烷较伸展，两个旁式构象的端基更接近。计算检验 TON 窄孔对构象的选择，以及部分再生长能否更快地探索这些构象。框架为 IZA 理想纯硅 TON 的 2×1×4 超胞，采用 UFF Si/O 及 Lorentz–Berthelot 混合参数。客体采用 TraPPE-UA 正丁烷的 Lennard–Jones 和扭转参数，固定 1.54 Å 键长和 114° 键角。温度为 298.15 K，逸度为 100 Pa。[孔内输入](inputs/flexible_butane_adsorption.json) · [IZA TON](https://www.iza-structure.org/IZA-SC/cif/TON.cif) · [TraPPE-UA](https://doi.org/10.1021/jp972543%2B) · [UFF](https://doi.org/10.1021/ja00051a040)

以 |φ|≥120° 定义反式。同一扭转势的理想气体反式比例解析值为 **66.47%**，模拟为 **66.23%**，95% 区间为 65.27–67.18%。理想气体的箱内平均分子数为 2.3864，解析值为 2.3881。两个结果检验了柔性分子的内配分函数及逸度归一化。[理想气体与首轮配对](butane_benchmark.json)

| 长轨迹 | 分子/超胞 | 反式比例 | 再生长接受率 | 构象 ESS/s | 载量 ESS/s |
|---|---:|---:|---:|---:|---:|
| 完整再生长 | 13.992 | 91.29% | 0.56% | 5.87 | 3.20 |
| 后缀再生长 | 14.059 | 91.39% | 15.37% | 5.49 | 3.27 |

六个新的配对种子中，每条链预热 100,000 次、生产 840,000 次。三进程配对的后缀路径墙钟加速为 **1.15 倍**，按各链墙钟时间之和计算，构象和载量的 ESS/s 比分别为 0.93 和 1.02。短轨迹中构象 ESS/s 比为 1.55。长轨迹的反式比例配对差为 +0.10 个百分点，95% 区间为 −0.92 至 +1.11 个百分点。完整与后缀路径的构象 R̂ 分别为 1.008 和 1.012，载量的最大 R̂ 为 1.022。两条路径都显示反式构象富集，载量仍需更充分的混合。[长轨迹配对](butane_extended_benchmark.json)

## 四、NaCl/SPC/E 恒电势界面

这项计算对应电双层及电容的数值核查。62 个水分子和一对 Na⁺/Cl⁻ 在相距 16 Å 的两块理想高斯电极间采样，电势差为 0.10 V。四个独立长链各预热 40,000 次，生产 160,000 次。[固定组成输入](inputs/nacl_spce_constant_potential.json) · [SPC/E](https://doi.org/10.1021/j100308a038) · [Joung–Cheatham 离子参数](https://doi.org/10.1021/jp8001614)

128 个构型中，缓存电极电荷与独立约束线性求解的最大差为 1.09×10⁻¹⁴ e，投影迭代解的最大差为 3.21×10⁻¹³ e。各链末态总能重算最大差为 1.72×10⁻⁶ K。电势重加权数值导数与涨落响应的最大差为 6.09×10⁻¹⁰ e/V。18 个电极位点上，稠密与投影迭代每构型的线程计时中位数约为 39 和 464 μs。[长链与独立求解核查](interface_extended_benchmark.json)

延长后共轭电荷 R̂ 从 2.59 降至 1.89。长轨迹观测到 Na–O 和 Cl–O 平均配位数 6.03 和 7.17，两项 R̂ 约为 1.2。数值电荷约束已验证，界面慢构型转换仍限制平衡电容估计。[首轮结果与链间差异](interface_benchmark.json)

## 五、显式水中的质子化与氧化还原

这项计算检验多个中间电荷态能否共享同一显式溶剂构型的静电响应。体系含 60 个水分子、一对 Na⁺/Cl⁻ 和两个固定反应中心，每次对其中一个中心的电荷态取样。每种反应使用四个独立种子及七个自由能窗口。首次配对中，局部响应缓存将固定构型上的七态求能加速 **8.43–8.60 倍**，最大能量差 **6.26×10⁻¹⁰ K**。真实物种的 pKa 和形式电位仍需独立参考态与标定。[首轮配对](reaction_benchmark.json) · [质子化输入](inputs/hydrated_acid_calibration.json) · [氧化还原输入](inputs/hydrated_redox_calibration.json)

长轨迹每窗口预热 4,000 次，保存 600 帧，邻帧相隔 20 次位移尝试。

| 模型 | 局部缓存加速 | 对逐态求能最大差/K | 合并窗口重叠谱隙 | 最大相邻能隙 R̂，首轮 → 长轨迹 |
|---|---:|---:|---:|---:|
| 质子化 | 8.76 | 4.66×10⁻¹⁰ | 0.0212 | 3.08 → 1.82 |
| 氧化还原 | 8.81 | 7.64×10⁻¹⁰ | 0.0362 | 2.52 → 1.67 |

窗口重叠可以连接端态，能隙观测量的跨链一致性仍未达到门槛。表中的倍数是固定构型七态求能的加速。长轨迹数据保留了各窗口的约化势及各独立种子的自由能估计。[长轨迹、耗时和逐窗口诊断](reaction_extended_benchmark.json)

## 六、铁氰化物与亚铁氰化物扩散电化学

以 298.15 K、1 mM、直径 3 mm 平面电极和文献扩散系数，分别复算计时安培及两档循环伏安。在 5 ms–0.5 s 内，400、800 和 1,600 格的电流与 Cottrell 解最大相对差依次为 0.439%、0.108% 和 **0.027%**，最细网格上的 log–log 斜率为 −0.50004。100 和 200 mV/s 在各自最细网格上的阴极峰分别为 −16.175 和 −22.872 μA，峰电流比为 1.41404，与可逆扩散的 √2 相差约 0.013%。[三档网格结果](transport_benchmark.json) · [扩散系数来源](https://doi.org/10.1021/ac50160a042)

## 复算

使用安装了 `8085910` 的 Python 3.13 环境。Apple M5 上使用 `python native/build.py --portable --no-openmp` 构建 C++ 后端。下面把输入与脚本复制到独立目录，使复算产物写入该目录。每个场景用脚本内设种子，所需框架场从输入构建。[复算脚本](benchmark_cases.py) · [IRMOF-1 配对脚本](../../../tools/benchmark_mixture_upgrade.py)

```bash
workspace="$PWD"
benchmark_dir="$workspace/benchmarks/cpu/scientific"
benchmark_python=python
reproduction_dir=$(mktemp -d)
cp "$benchmark_dir/benchmark_cases.py" "$reproduction_dir/"
cp -R "$benchmark_dir/inputs" "$reproduction_dir/"
cd "$workspace"

"$benchmark_python" -m tools.benchmark_mixture_upgrade \
  "$reproduction_dir/inputs/irmof_mixture.json" --pressure-pa 200000 \
  --replicas 6 --steps 150000 --burnin 20000 \
  --cache "$reproduction_dir/field_cache/irmof" \
  --output "$reproduction_dir/IRMOF1_吸附配对.json"

cd "$reproduction_dir"
"$benchmark_python" benchmark_cases.py xylene --cache field_cache/xylene
"$benchmark_python" benchmark_cases.py xylene_exchange --workers 3 --cache field_cache/xylene
"$benchmark_python" benchmark_cases.py butane --cache field_cache/butane
"$benchmark_python" benchmark_cases.py butane --extended --workers 3 --cache field_cache/butane
"$benchmark_python" benchmark_cases.py interface --workers 3
"$benchmark_python" benchmark_cases.py interface --extended --workers 4
"$benchmark_python" benchmark_cases.py reaction --workers 3
"$benchmark_python" benchmark_cases.py reaction --extended --workers 3
"$benchmark_python" benchmark_cases.py transport
```
