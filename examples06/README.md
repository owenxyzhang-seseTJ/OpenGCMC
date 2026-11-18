# PoreMC 0.6 输入示例

## 小型柔性 Ewald 任务

在项目根目录运行：

```bash
python native/build.py
python -m poremc run examples06/small_delayed_ewald.json -o run-delayed
```

`small_delayed_ewald.json` 是四位点带电柔性链与两个固定框架位点的演示模型，包含插入、删除、平移、旋转和完整再生长。参数用于演示与接口验证。运行 100 次预热、600 次生产尝试，每个生长阶段 6 个 trial。

示例使用 `sampling_mode="delayed"`、`host_grid_mesh=32` 和默认 Boltzmann 陡峭区插值。缓存文件位于 `examples06/cache/small_delayed_ewald.bin`；该路径由输入 JSON 所在目录解析。再次运行、改变温度或储库活度可复用原始字段。

输出目录包含 `result.json`、`input.json`、`checkpoint.json` 和 `final.xyz`。`result.json` 的 `acceleration` 字段记录两阶段接受统计及网格缓存命中。

## 在实际材料输入上选择网格

- 保留实际材料的晶胞、客体、力场、Ewald 参数和储库定义。
- 从 `host_grid_mesh=64` 开始；对比 `32` 的吞吐、接受数、占据数轨迹和有效样本效率。
- `host_grid_steep="boltzmann"` 是默认值；`linear` 用于复现旧参考场的调参实验。
- 需要加强孔隙中的插入起点选择时，设置 `grid_root_bias=0.95`；其余 5% 保留均匀提议。
- `host_grid_cap=1e5` 只限制参考场，最终接受仍使用完整目标势校正。
- `ewald_k_cutoff=0` 保持原矩形模式集合；半谱合并不会改变该目标。

真实输入的同任务性能比较使用项目根目录下的 `tools06/benchmark_input.py`。脚本的 `--steps` 单位是尝试移动数。
