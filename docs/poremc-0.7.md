# OpenGCMC / PoreMC 0.7

项目名 PoreMC，Python 导入名 `poremc`。本版针对固定框架带电客体的单组分 GCMC，加速五类原生移动，包括刚体和树状柔性分子的插入、删除、平移、旋转和 CBMC 再生长。源码采用 MIT 许可。

## 安装与运行

Python 3.10+、C++17 编译器、NumPy 和 SciPy。CPU 构建可独立使用。下列命令从解压后的源码根目录执行。

```bash
python -m pip install -e '.[test]'
python native/build.py
python -m poremc capabilities
python -m poremc run examples06/small_delayed_ewald.json -o run-example
```

已有 PoreMC JSON 可直接使用。要在固定框架任务中启用网格和持久缓存，请在已有 `sampling` 对象内合并以下字段，保留输入原有的活动度、力场、位点电荷和移动权重。

```json
{
  "driver": "cpp",
  "sampling_mode": "delayed",
  "host_grid_mesh": 64,
  "host_grid_steep": "boltzmann",
  "host_grid_cache": "cache/framework.bin",
  "reciprocal_method": "direct",
  "reciprocal_backend": "cpu",
  "ewald_half_spectrum": true,
  "ewald_k_cutoff": 0.0
}
```

网格缓存路径相对于输入 JSON 的目录。温度、活度和种子变化时可复用磁盘原始场。查询时按温度构建的权重存放在内存。Linux 在同目录生成独占临时文件，通过长度检查、分块写入、同步、关闭和原子重命名发布完整字段。0.6 与 0.7 的 PMCHG061 缓存支持双向读取。对指定输入的关缓存、冷写、热读比较可运行：

```bash
python tools07/audit_host_cache.py --input real.json --mesh 64 \
  --workdir /path/to/job/filesystem --output cache07.json
```

需要在已测的 0.6 源树中单独替换缓存写入器时，可使用 `hotfix06/PoreMC_0.6_HostCache.patch`。在原始 0.6 源码根目录执行 `patch -p1 < /path/to/PoreMC_0.6_HostCache.patch`，再运行 `python native/build.py`。该补丁只更改 `native/host_grid.cpp` 的缓存写入和容量检查。

## CPU 与 GPU 计算

直接 Ewald 的分子结构因子使用三轴谐波表，按固定的 x、y 波矢组成 z 方向连续区间。C++ 内核沿连续区间处理实部与虚部，用同一已接受态的结构因子直接计算分子替换能差。单个 CBMC 生长位点利用恒定电荷平方项计算批候选能量。三立方框架查询复用带周期边界扩展的能量数组。中等温度的陡峭区域预计算 Boltzmann 因子，极端温度保留按最小能量平移的稳定公式。

CUDA 路径使用完整 FP64 和相同的有限倒空间模式。`reciprocal_backend="cuda-tensor"` 显式选择三轴相位张量收缩。`exact` 对多个候选做批处理并返回能量，`delayed` 在通过初筛后计算新构型的结构因子并交回 CPU 校正、提交。平移复用 CPU 上的旧结构因子。默认 CPU 路径支持三维直接 Ewald 和 PME；此张量内核对应直接 Ewald。完整设置、范围和既有热力学与电化学模块见 [上一版接口说明](../README_0.6.md)。

```bash
python native/build.py --cuda --cuda-arch sm_86
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tools07/validate_tensor_gpu.py --input real.json --output rtx07.json
```

将 sm_86 换为实际设备架构。`long_range_profile` 提供后端、设备名、候选批数与 `gpu_selected_structures`。较大的 GPU 批次和单链延迟校正的同步开销不同，设备脚本分别计时。此发布环境完成了 sm_86 原生插件的 CUDA 机器码编译和链接。数值交叉验证使用主机端测试插件；GPU 吞吐需读取实际设备的结果。

## 性能与复测

用户 HPC 在真实 Ni_TPT_TATBA + 带电 p-xylene 上测得 0.6 delayed64 为 3,592 moves/s、0.6 delayed32 为 2,954 moves/s。RASPA2 在 N=5.53 时为 125.7 cycles/s，按每循环 20 次尝试折合 2,514 moves/s。0.6 的磁盘缓存当时关闭。按上述吞吐的直接比值，两个网格依次快 42.9% 和 17.5%。

本版在 4,416 框架位点、12 个芳香客体位点、10 个 trial、完整矩形 k13 Ewald 的合成任务上测试。与同机 0.6 配对，低载量 exact、delayed32、delayed64 完整循环的加速中位数分别为 1.66、1.53、1.65。平均 N=18.27、31.48、40.82 时，delayed64 分别为 1.51、1.28、1.40。原始逐种子记录见 `evidence07`。运行以下命令可在同一 HPC 输入上得到 0.6/0.7 的配对结果：

```bash
python tools07/compare.py --input real.json --steps 20000 --burnin 2000 \
  --repeats 3 --independent --cache-dir /path/to/job/filesystem/cache \
  --output comparison07.json
```

每次尝试包括提议、能量计算、接受判定及提交。初始化和最终独立能量重算单独统计。脚本选取作业允许的一个 CPU 核，交替运行版本，固定活动度、种子和输入，原始 0.6 使用内存网格，0.7 启用磁盘缓存。要重现合成体系，可将 `--input real.json` 替换为 `--synthetic low`、`mid`、`dense` 或 `high`。比较吸附量时应使用相同活动度，并检查实际粒子数轨迹。

中文报告与 LaTeX 源文件位于 `report_07/`。独立测试覆盖精确能差、分布、检查点、缓存故障与并发写入。项目回归入口：

```bash
python -m pytest -q tests tests_legacy tests03 tests04 tests06 tests07
python report_07/generate_results.py
python report_07/build_report.py
```
