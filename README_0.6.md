# PoreMC 0.6 — 固定框架网格与两阶段原生 GCMC

PoreMC 0.6 将固定框架势能预计算、参考势 CBMC 和精确长程校正接入同一个 C++ 采样循环。大量候选先查询网格，只有通过第一阶段的完整分子才进入 Ewald 校正；框架与分子缓存跨移动复用，框架网格还可跨温度、压力和随机种子复用。

`FastGCMC` 的默认设置仍为 `sampling_mode="exact"`，保持已有输入的计算方式。固定框架、高 CBMC 候选数的任务可以显式选择 `sampling_mode="delayed"`，生产设置从 `64³` 网格开始，并对比 `32³` 网格的接受率、计数轨迹和有效样本效率。

## 0.6 的改进及用途

| 功能 | 实现 | 主要用途 |
|---|---|---|
| 框架能量网格 | 每个唯一 LJ 参数类型一张网格，共享单位电荷的实空间与倒空间电势 | 将候选–框架的重复原子遍历变成局部插值 |
| 原生参考势 CBMC | 候选生长、回溯、Rosenbluth 权重和两阶段接受均在 C++ 中 | 将长程评价从全部 trial 移到被选中的完整分子 |
| 稳定的近壁提议 | 常规区域三立方插值；陡峭区域默认插值 Boltzmann 权重 | 减少粗网格对狭窄吸附位点的人工排斥 |
| 倒空间共轭半谱 | 合并正负波矢并调整能量系数 | 保留原有限 Ewald 和，减少存储和计算 |
| 平移相位更新 | 以平移相位乘旧分子结构因子 | 避免平移后重新遍历全部分子位点 |
| 局部残差缓存 | 缓存已接受分子的 exact/reference host 与分子内项；共同 guest 短程项直接相消 | 避免第二阶段重复计算不变部分 |
| 持久框架缓存 | 输入指纹、字段完整性校验、原子写入 | 复用吸附等温线和多种子任务的预处理结果 |
| 根位置偏置 | 框架网格导出的 alias 提议，保留均匀分量并校正提议密度 | 将更多插入起点放入可达孔隙 |
| checkpoint v3 | 保存接受态、逐分子倒空间贡献、随机数与阶段计数 | 延续相位递推后的浮点状态 |

原生循环处理单组分、刚性或树状柔性分子，支持插入、删除、平移、旋转和完整分子 CBMC 再生长。0.3/0.4 的热力学响应、自由能、电解质、恒电位电极、水合、反应和电流模块保留原有接口，见 [README_0.3.md](README_0.3.md) 和 [README_0.4.md](README_0.4.md)。0.5 长程算法说明保存在 [README_0.5.md](README_0.5.md)。

## 安装与快速运行

需要 Python 3.10+、NumPy、SciPy 和 C++17 编译器。源码包中的原生库在本机编译。

```bash
python -m pip install -e ".[test]"
python native/build.py
python -m poremc run examples06/small_delayed_ewald.json -o run-delayed
```

示例是一个小型带电柔性链与固定框架模型，包含全部五类移动。输出目录包括 `result.json`、输入副本、最终构型和 checkpoint。第二次运行会复用示例目录下的框架缓存。

可移植 CPU 编译使用 `python native/build.py --portable`；无 OpenMP 的编译使用 `--no-openmp`。

## 推荐输入设置

保留原来的分子、框架、力场、温度与储库输入，在 `sampling` 中加入：

```json
{
  "sampling": {
    "driver": "cpp",
    "sampling_mode": "delayed",
    "trials": 10,
    "reciprocal_method": "direct",
    "reciprocal_backend": "cpu",
    "ewald_half_spectrum": true,
    "ewald_k_cutoff": 0.0,
    "host_grid_mesh": [64, 64, 64],
    "host_grid_steep": "boltzmann",
    "host_grid_cap": 100000.0,
    "host_grid_cache": "cache/framework_guest_fields.bin",
    "grid_root_bias": 0.0
  }
}
```

CLI 中的 `host_grid_cache` 相对于**输入 JSON 所在目录**解析。Python API 中的相对路径相对于当前工作目录解析。缓存保存原始 LJ 与电势字段；改变温度、活度或随机种子可以继续加载同一文件。改变框架、位点参数、静电参数或网格尺寸时，输入指纹会触发重新构建。

| 参数 | 含义与选择 |
|---|---|
| `sampling_mode` | `exact` 为默认逐候选目标势评价；`delayed` 开启网格参考势与最终精确校正 |
| `host_grid_mesh` | Ewald 框架网格使用每维至少 8 的二次幂；`64` 用作生产起点，`32` 用于更低预处理成本的对比 |
| `host_grid_steep` | 默认 `boltzmann`；`linear` 保留为对照和插值研究选项 |
| `host_grid_cap` | 参考场的有限能量范围，单位 K；默认 `1e5`。目标势能不截断。设为 `0` 时，陡峭 stencil 回退精确框架能 |
| `grid_root_bias` | 默认 `0`；可尝试 `0.95`，将 95% 根位置概率分配给框架场偏置，5% 保留均匀分量 |
| `ewald_half_spectrum` | 默认 `true`；合并共轭波矢，保持同一有限 Ewald 目标 |
| `ewald_k_cutoff` | 默认 `0` 保留完整矩形集合；正值是以 Å⁻¹ 为单位的球形波矢截断，改变有限模式集合 |

`grid_root_bias` 与外部 `root_proposal` 二选一。偏置概率在生产前固定，接受率包含对应的根位置提议密度。

## 数学结构：参考势选择候选，目标势决定分布

令目标能量为 $U$，固定参考势为 $U_0$，残差为 $R=U-U_0$。代码先在 $U_0$ 下完成 CBMC 生长与旧构型回溯，再执行：

$$
\alpha_1=\min(1,r_0),\qquad
\alpha_2=\min\{1,\exp[-\beta\Delta R]\}.
$$

$r_0$ 包含储库活度、粒子计数、移动概率、Rosenbluth 因子和提议密度。只有第一阶段通过，才计算所选完整分子的精确残差。正反向两阶段接受概率之比恢复目标分布所需的完整比值。

候选与固定框架的相互作用来自预计算网格，guest–guest 短程作用仍按当前接受态计算。由于 guest 短程项在 $U$ 和 $U_0$ 中相同，第二阶段直接省去这部分相消项。接受后统一提交结构因子和局部缓存，拒绝时保留原接受态。

### 为什么陡峭区域插值 Boltzmann 权重

对一个较粗 cell，直接线性混合高排斥角点的能量，会把人工势垒扩展到 cell 内的吸附区域。本轮旧 `linear + 32³` pilot 的计数轨迹显示较慢的混合，加密网格改善了该现象（见 `evidence06/linear_mixing_diagnostics.json`），因此新增默认的 Boltzmann 插值：

$$
U_0(\mathbf r)=-T\log\left[\sum_{i=1}^{8}w_i(\mathbf r)e^{-U_i/T}\right].
$$

这里能量单位为 $U/k_B$，$w_i$ 是正的三线性权重。实现对非零权重角点做稳定的局部 log-sum-exp；温度只在查询时进入，不改变磁盘上的原始字段。常规区域仍采用三立方插值。

参考势的误差不会改变最终目标 Hamiltonian，但会影响通过率与混合速度。选择网格时同时比较尝试数/s、接受数/s、占据数轨迹及有效样本效率；本版报告由实测 JSON 生成这些结果。

### 倒空间模式与目标精度

`kmax=(13,13,13)` 的原矩形集合包含 19,682 个非零正负模式；共轭半谱存储 9,841 个模式并合并其系数，仍计算同一个有限 Fourier 和。平移时使用

$$
S_{\mathrm{new}}(\mathbf k)=e^{i\mathbf k\cdot\Delta\mathbf r}S_{\mathrm{old}}(\mathbf k).
$$

`ewald_k_cutoff` 则删除球外模式，需要单独检查静电精度；它与共轭半谱具有不同含义。`tools06/advise_reciprocal.py` 用于评估输入模型的倒空间选择，具体参数见其 `--help`。

## Python API 与诊断

```python
from poremc.fast import FastGCMC

with FastGCMC(
    molecule, potential, cell, temperature, [activity],
    trials=10,
    sampling_mode="delayed",
    host_grid_mesh=64,
    host_grid_steep="boltzmann",
    host_grid_cap=1e5,
    host_grid_cache="cache/framework_guest_fields.bin",
    grid_root_bias=0.95,
    reciprocal_method="direct",
    reciprocal_backend="cpu",
) as sim:
    result = sim.run(100000, burnin=10000, record_energy=False)
    print(sim.acceleration_profile)
    print(sim.long_range_profile)
    sim.save_checkpoint("state.json")
```

- `acceleration_profile`：各移动的第一阶段通过数、精确校正数、最终接受数，以及框架网格构建/加载/插值统计。
- `long_range_profile`：实际后端、存储模式数、完整模式数、结构因子贡献、FFT 和 GPU 批次数。
- `trial_molecule_energies` / `trial_site_energies`：目标势候选探针，支持排除指定旧分子。
- `trial_surrogate_molecule_energies` / `trial_surrogate_site_energies`：参考势候选探针，用于检查插值与前缀能量。
- 探针不消耗采样随机数、不提交构型。
- `FastGCMC.from_checkpoint(path, potential)`：恢复 checkpoint v3 中的逐分子结构因子、总接受态、随机数和阶段计数；精确 CPU 延续使用相同原生二进制和线程配置。

`total_energy()` 返回目标势相对于空框架的能量，不返回网格参考势。PME 仍可通过 `reciprocal_method="pme"` 选择；此时目标是指定网格的 PME Hamiltonian，`potential.total_energy(...)` 保留 direct Ewald 参考。

## CPU 与 GPU 路径

| 设置 | 实际执行 |
|---|---|
| `delayed + cpu` | 框架网格、CBMC 与所选分子精确校正均在原生 CPU |
| `delayed + auto` | 原生 CPU；诊断明确标出 `selected_correction_backend="cpu"` |
| `delayed + cuda` | 给出配置错误，指向支持 CUDA 候选批处理的 exact 模式 |
| `exact + cuda` | 0.5 引入的原生 direct Ewald CUDA 候选批处理 |
| `exact + auto` | 可用时选择 CUDA，否则保持原生 CPU |

CUDA 构建和设备验证沿用：

```bash
python native/build.py --cuda --cuda-arch sm_86
python tools05/validate_cuda_ewald.py --output evidence06/rtx_validation.json
```

按实际 GPU 设置架构目标。几何提议、实空间、接受判断和状态提交仍在 C++ 主机侧；PME FFT 当前使用 CPU。GPU 执行状态由设备验证结果和 `gpu_batches` 记录。

## 在真实 HPC 输入上比较

准备包含实际框架、客体、力场和初始构型的 PoreMC JSON，然后运行：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tools06/benchmark_input.py path/to/real_system.json \
    --steps 3000000 --burnin 10000 --repeats 3 \
    --modes baseline05,native06,delayed32,delayed64,biased64 \
    --output hpc_results/comparison_06.json
```

工具在独立进程中交替测量相同输入，自动构建缺失的 `baseline_05` 原生库，保留完整矩形 Ewald 集合并固定单线程。记录初始化时间、完整移动吞吐、接受吞吐、占据数轨迹、两阶段计数和终态能量重算误差。

`--steps` 是**尝试移动数**。若外部软件的一个 cycle 定义为 `max(20,N)` 次移动，则循环总尝试数为各 cycle 的该值之和；约 300 万次移动对应每 cycle 20 次尝试下的 150,000 cycles。比较时使用各自实际定义，并同时核对接受率与粒子数分布。

`tools06/benchmark_hostgrid.py` 使用 4,416 框架位点、12 位点客体等合成 fixture 来定位成本；这些 fixture 复现计算规模。真实 p-xylene / CIF 的性能应由上述 `benchmark_input.py` 输入导出。原始 HPC 上报信息保存在 `evidence06/user_hpc_context.json`，与本轮本机实测分别标识。

## 验证、报告与源码组织

```bash
python -m pytest -q tests tests_legacy tests03 tests04 tests06
python tools06/audit_host_grid.py
python tools06/audit_delayed.py
```

独立核验覆盖解析网格场、低温 Boltzmann 节点极限、跨温度缓存、原子/刚体/柔性长程能量、共轭半谱、前缀能量、两阶段采样分布和 checkpoint 延续。最终测试数量与性能结果见 `evidence06/` 及 `report_06/`；报告表格从原始数据生成。

| 路径 | 内容 |
|---|---|
| `native/host_grid.*` | 字段预计算、周期插值、Boltzmann 参考势与持久缓存 |
| `native/fast_mc.cpp` | 两阶段采样、CBMC、局部残差缓存和接受后提交 |
| `native/ewald_state.*` | 完整有限 Ewald、半谱、平移相位更新、框架场 FFT 和 PME |
| `native/ewald_cuda.*` | exact 模式的 CUDA 候选批处理 |
| `tools06/`、`tests06/` | 真实输入与合成基准、物理审计、回归测试 |
| `baseline_05/` | 原始 0.5 代码对照与来源记录 |
| `report_06/`、`evidence06/` | 数学推导、测量记录、报告源文件 |

单位为坐标 Å、电荷 e、能量 `U/kB`（K）。Ewald 使用三维 tin-foil 边界，包含实空间、倒空间、自能、均匀背景和 primary 分子内排除/1–4 修正。完整模型有净电荷时，输入需显式启用均匀中和背景。

项目采用 MIT 许可证。历史报告和数据保留各自的版本与来源。
