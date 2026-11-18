# PoreMC 0.3 — Thermodynamics and Electrochemical GCMC

在 PoreMC 0.2 CacheFirst 上增加化学势响应、占据自由能、盐组巨正则和恒电位电极。独立实现，MIT 许可证；原有第三方示例数据保留原许可证。

## 安装与开始

需要 Python 3.10+、NumPy、SciPy；原生能量核需要 C++17 编译器。图表需要 Matplotlib，测试需要 pytest。原生库须在当前机器编译，包中不分发针对本机的二进制。

```bash
python -m pip install -e . pytest matplotlib
python native/build.py
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m poremc.thermo_cli run examples03/interface_constant_potential.json -o run_interface
python -m poremc.thermo_cli run examples03/electrolyte_21.json -o run_salt
python -m poremc.thermo_cli run examples03/interface_constant_potential.json \
  --restart run_interface/checkpoint.json -o continued
```

以上电化学示例是均匀介电背景中的模型离子，不是标定后的真实盐水。原来的分子/柔性客体入口仍为 `python -m poremc run input.json -o output`。

## 新功能

| 模块 | 当前可运行能力 |
|---|---|
| `response.py` | 流式合并中心矩、化学势响应、块 bootstrap、独立盐组坐标、化学势与电位重加权、联合离子/电荷响应 |
| `free_energy.py` | 冻结占据偏置、ND 概率与自由能、log-space MBAR、重叠检查、分窗口块误差、完整高斯电荷的 F(N,Q) |
| `electrolyte.py` | 任意非负整数盐组插删、多价组合因子、电中性约束、固定带电框架、Ewald、局部增量、可重启采样 |
| `electrode.py` | 固定高斯电极、恒电位 Legendre 势、总电中性约束、算子缓存、解析电荷消元、电容和批量响应 |
| `engine.py` | 原有分子 GCMC 新增可选冻结占据偏置，保留原本两阶段接受和 CBMC |
| `thermo_cli.py` | JSON 运行入口、NPZ 轨迹/坐标、响应与 MBAR 后处理 |

### 两种电解质交换

无电极时，通道的离子总电荷必须为零。例如物种价态 `[2,-1]` 对应盐组计量 `[1,2]`。有恒电位电极时，可以显式开启 `allow_single_ion`：每次单离子插删的净电荷由电极补偿。溶液可以带净电荷，但离子、框架与电极组成的全系统保持电中性。

### 电化学物理模型

固定几何、固定高斯宽度、均匀介电常数；三维周期 Ewald，可选正交 z 方向的 Yeh–Berkowitz 偶极修正。该边界不等于精确二维开放边界。接受率使用包含电源做功的恒电位势，电容和 F(N,Q) 恢复电极的条件高斯涨落。

本版计算平衡双电层/电吸附，不实现 Faradaic 电子转移、电流、恒 pH 或显式分子水合。原分子路径支持柔性客体，但当前新离子驱动的溶剂为均匀介电背景；两条路径未宣称自动合并为显式水电化学引擎。

### 单位与储库

长度 Å；能量 U/kB 为 K；电荷以 e 计；电位 V。`log_activities` 为 ln(a_i × 1 Å³)，不是 ln(摩尔浓度)。浓度到活度需要同模型体相的过剩化学势。单离子活度和电极共同电势零点必须采用相同参考。

## 输出与后处理

运行输出 `config.json`、`trajectory.npz`、`positions.npz`（启用坐标时）、`summary.json`、`response.json`、`checkpoint.json`。坐标采用 flat arrays + offsets，无 pickle。重启检查模型、储库、区域、偏置与通道指纹。

```bash
python -m poremc.thermo_cli response run_interface/trajectory.npz -o response.json
python -m poremc.thermo_cli mbar results_03/molecular_windows.npz -o landscape.npz
```

`response` 子命令仅分析无偏计数轨迹；有偏轨迹先用 MBAR 或显式去偏权重。MBAR 文件包含 `u_kn,n_k,u_target,cv,edges`，数组顺序按窗口连续排列。API 可分析多维景观，CLI 的该入口为一维。

```python
from poremc.response import response_summary, SaltCoordinates
from poremc.free_energy import HarmonicCountBias, MBAR, landscape
# counts: equilibrium production array, (frames, species)
r = response_summary(counts, temperature=300, block_length=100, bootstrap=200)
bias = HarmonicCountBias((1, 1), center=6, stiffness=0.1)
# Pass count_bias=bias to the original molecular GCMC, or bias=bias to ElectrolyteGCMC.
```

`tools03/run_windows.py` 对配置文件运行独立子进程，单个窗口不并行提交相互依赖的 MC 状态。固定电极矩阵与框架邻域可复用；接受后才使动态邻域失效。MBAR 对完全相同的约化势列做精确重数压缩，不舍入、不粗分箱。

## 验证与报告

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 python -m pytest tests tests03 -q --junitxml=results_03/all_tests.xml
bash tools03/reproduce_all.sh
python report_03/build_report.py
```

交付时全部 **134 项通过（原有80 + 新增54）**。原始轨迹与结果在 `results_03/`，报告为 `report_03/report.pdf`。

重要结果：双阱自由能 RMS 误差 0.098 kBT；六条界面链终态电势约束最大残差 2.84e-16 V，保存帧全系统电荷最大残差 2.66e-14 e。32/72/128 位点的电极批量响应相对逐候选重新求解 KKT 的微基准中位加速为 90.7/110.3/132.7 倍，不等于完整 GCMC 加速，不是 RASPA3 对照。

新增设备接口 `ElectrodeResponse.to_device()` 使用可选 CuPy，固定算子设备常驻；本次无 GPU，未执行该接口。0.2 CUDA 数据仅保留作为继承回归，不用来推算新增电化学路径的性能。

## 文件与版本

- `report_03/`：本版 LaTeX、6张矢量图、由原始结果生成的表格。
- `tools03/reproduce.py`：七类验证/基准任务。
- `tests03/`：新增解析、能量、统计与重启验收。
- `README_0.2.md`, `report_0.2/`, `benchmarks/`：上一版资料，数字不属于本次新增模型。
- `evidence/PoreMC_GPU_preliminary_20261117.zip`：原80项测试使用的历史固定数据。
- `MANIFEST.sha256`：交付文件校验表，不包含非便携二进制或字体。
