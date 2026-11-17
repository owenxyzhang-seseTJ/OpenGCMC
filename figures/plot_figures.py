#!/usr/bin/env python3
"""Reproduce the OpenGCMC figures from retained benchmark JSON and code mappings.

Only descriptive processing is performed; this script does not run simulations.
"""
from __future__ import annotations

import csv
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "opengcmc-figure-mplconfig"))
import matplotlib as mpl
mpl.use("Agg")  # 使用无窗口后端，保持批量导出可重现
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "figures/source_data"
ASSETS = ROOT / "docs/assets"
PALETTE = {"teal": "#247C81", "pale_teal": "#B5D8D8", "slate": "#667887", "pale_slate": "#D5DEE5", "ink": "#26333B", "muted": "#62737B", "line": "#C5D6D8", "lavender": "#DFD9ED", "purple": "#8066A6", "coral": "#C97762", "pale_coral": "#F0CABB", "amber": "#B88A49", "green": "#518872", "pale_green": "#C4DFCF", "blue": "#527DAC", "pale_blue": "#C5D8EC"}  # 统一青绿主色、灰蓝对照色及淡紫可选路径
FONT = "Arial" if any(f.name == "Arial" for f in font_manager.fontManager.ttflist) else "DejaVu Sans"  # 只选择一种实际存在的字体
mpl.rcParams.update({"font.family": FONT, "font.size": 9, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 10, "svg.fonttype": "none", "pdf.fonttype": 42, "axes.linewidth": .75, "text.color": PALETTE["ink"], "axes.labelcolor": PALETTE["ink"], "xtick.color": PALETTE["ink"], "ytick.color": PALETTE["ink"], "savefig.facecolor": "white"})  # 设置全局单字体、文字层级、可编辑矢量文字及白底导出
mpl.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": FONT + ":italic", "mathtext.bf": FONT + ":bold", "mathtext.default": "regular"})  # 数学标签与正文使用同一字体家族


def load_and_validate():
    """Validate the retained pairs and derive rates without modifying raw data."""
    experiments = []
    rows = []
    pairs = []
    for name, reference, n_expected, ref_field, own_field in [
        ("raspa3", "RASPA3", 40000, "raspa_production_seconds", "opengcmc_production_seconds"),
        ("graspa", "gRASPA", 1000, "gRASPA_sampling_seconds", "opengcmc_sampling_seconds"),
    ]:
        path = ROOT / f"benchmarks/results/{name}_comparison.json"
        raw = json.loads(path.read_text())
        records = raw["replicates"]
        assert len(records) == 3 and len({r["seed"] for r in records}) == 3
        assert raw["framework_sites"] == 4992
        assert raw["model"]["temperature_K"] == 300
        assert raw["model"]["guest_sigma"] == 3 and raw["model"]["guest_epsilon"] == 5
        assert "excluded" in raw["timing_scope"].lower()
        if name == "graspa":
            assert raw["execution_conditions"]["CUDA_LAUNCH_BLOCKING"] == "1"
            assert "1000" in raw["sampling_protocol"] and "no warm-up" in raw["sampling_protocol"]
        rates = {reference: [], "OpenGCMC": []}
        ratios = []
        for record in records:
            nref = record["actual_attempts"] if name == "raspa3" else record["gRASPA_scheduled_attempts"]
            nown = record["actual_attempts"] if name == "raspa3" else record["opengcmc_scheduled_attempts"]
            assert nref == nown == n_expected
            tref, town = float(record[ref_field]), float(record[own_field])
            assert all(math.isfinite(t) and t > 0 for t in (tref, town))
            ratio = (nown / town) / (nref / tref)
            assert math.isclose(ratio, record["throughput_ratio"], rel_tol=1e-12)
            ratios.append(ratio)
            for engine, steps, seconds, backend in [(reference, nref, tref, "GPU" if name == "graspa" else "CPU"), ("OpenGCMC", nown, town, "CPU cached native")]:
                rate = steps / seconds
                rates[engine].append(rate)
                rows.append(dict(experiment=name, seed=record["seed"], engine=engine, scheduled_attempts=steps, sampling_seconds=seconds, attempts_per_second=rate, backend=backend, protocol_id=name + "_controlled_lj"))
            pairs.append(dict(experiment=name, seed=record["seed"], reference_seconds=tref, opengcmc_seconds=town, paired_rate_ratio=ratio))
        median = statistics.median(ratios)
        assert math.isclose(median, raw["summary"]["median_throughput_ratio"], rel_tol=1e-12)
        experiments.append(dict(experiment=name, reference=reference, rates=rates, median_paired_ratio=median, minimum_paired_ratio=min(ratios), maximum_paired_ratio=max(ratios), raw=raw))
    return experiments, rows, pairs


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_source_data(experiments, rows, pairs):
    DATA.mkdir(parents=True, exist_ok=True)
    save_csv(DATA / "performance_replicates.csv", rows)
    save_csv(DATA / "performance_pairs.csv", pairs)
    summary = []
    for e in experiments:
        for engine, values in e["rates"].items():
            summary.append(dict(experiment=e["experiment"], engine=engine, median_attempts_per_second=statistics.median(values), median_paired_rate_ratio=e["median_paired_ratio"], minimum_paired_rate_ratio=e["minimum_paired_ratio"], maximum_paired_rate_ratio=e["maximum_paired_ratio"], repeat_count=len(values)))
    save_csv(DATA / "performance_summary.csv", summary)
    log = dict(status="PASS", confirmation="User approved the two-panel performance and three-layer architecture plan; in-figure test conditions omitted as requested.", dropped_rows=0, input_rows=12, output_rows=len(rows), formulas={"attempts_per_second": "scheduled_attempts / sampling_seconds", "paired_rate_ratio": "(OpenGCMC attempts / seconds) / (reference attempts / seconds)", "bar": "median of three per-engine attempt rates", "callout": "median of three paired rate ratios"}, units={"length": "angstrom", "energy": "U/kB in K", "rate": "scheduled attempts per second", "time": "seconds"}, source_files=[f"benchmarks/results/{e['experiment']}_comparison.json" for e in experiments], font_family=FONT, validation=["three unique seed pairs per experiment", "positive finite times", "matching scheduled counts within each pair", "matched 4992-site neutral probe model", "raw timing scopes retained", "stored and recomputed ratios agree", "no rows discarded", "no simulation executed"])  # 在审计记录中保存实际采用的单字体设置
    (DATA / "preprocessing_log.json").write_text(json.dumps(log, indent=2) + "\n")
    nodes = [
        dict(id="host", label="Periodic host and site parameters", source="poremc/kernels.py; native/kernels.cpp"),
        dict(id="csr", label="Exact fractional-cell CSR neighbours", source="native/kernels.cpp: Environment; poremc_environment_export_plan"),
        dict(id="pairs", label="Bound mixed-parameter rows", source="native/kernels.cpp: bind; BoundQuery"),
        dict(id="alias", label="Frozen full-support voxel alias proposal", source="poremc/proposals.py: VoxelProposal; build_host_proposal"),
        dict(id="trials", label="Trial batch", source="poremc/batching.py: evaluate_requests; poremc/cbmc.py: grow"),
        dict(id="energy", label="Exact energies and index exclusion", source="native/fast_mc.cpp: external; native/kernels.cpp"),
        dict(id="cbmc", label="Forward and retraced CBMC with 1/(Vq)", source="poremc/cbmc.py: grow; retrace; native/fast_mc.cpp: grow"),
        dict(id="decision", label="Exact acceptance decision", source="native/fast_mc.cpp: step; poremc/engine.py: _accept_candidate"),
        dict(id="state", label="Accepted guest state", source="native/fast_mc.cpp: cached_environment; step"),
        dict(id="guest", label="Reusable guest environment", source="native/fast_mc.cpp: cached_environment; external"),
        dict(id="native", label="C++ native CPU sampler; single component LJ/DSF", source="poremc/fast.py: FastGCMC"),
        dict(id="python", label="Python general sampler; mixtures, Ewald, delayed acceptance", source="poremc/engine.py: GCMC; poremc/electrostatics.py; poremc/accelerators.py"),
        dict(id="cuda", label="Optional CUDA energy backend", source="poremc/cuda.py: CudaEnvironment; poremc/cached_cuda.cu"),
        dict(id="replicas", label="Shared host, independent state and RNG", source="poremc/batching.py: run_native_replicas"),
    ]
    edges = [
        ("host", "csr", "build once", "native/kernels.cpp"), ("host", "pairs", "bind parameters", "native/kernels.cpp"), ("host", "alias", "optional proposal field", "poremc/proposals.py"),
        ("csr", "energy", "exact candidates", "native/kernels.cpp"), ("pairs", "energy", "reuse mixed rows", "native/kernels.cpp"), ("alias", "cbmc", "optional forward/retrace density correction", "poremc/cbmc.py; native/fast_mc.cpp"),
        ("trials", "energy", "evaluate batch", "poremc/batching.py"), ("energy", "cbmc", "trial energies", "poremc/cbmc.py"), ("cbmc", "decision", "weights", "native/fast_mc.cpp"),
        ("decision", "state", "accept commits and invalidates guest cache", "native/fast_mc.cpp"), ("state", "guest", "lazy rebuild on change", "native/fast_mc.cpp"), ("decision", "guest", "reject reuses accepted environment", "native/fast_mc.cpp"), ("guest", "energy", "exclude moved molecule by original index", "native/fast_mc.cpp"),
        ("python", "cuda", "optional energy call only", "poremc/potentials.py; poremc/cuda.py"), ("host", "replicas", "share immutable host only", "poremc/batching.py"),
    ]
    save_csv(DATA / "architecture_nodes.csv", nodes)
    save_csv(DATA / "architecture_edges.csv", [dict(source=a, target=b, relation=c, code_source=d) for a, b, c, d in edges])


def audit_text(fig, name):
    """Use renderer-measured text extents for a geometry-aware overlap check."""
    fig.canvas.draw()  # 在真实渲染器上取得最终文字边界
    renderer = fig.canvas.get_renderer()
    all_text = [t for t in fig.findobj(mpl.text.Text) if t.get_visible() and t.get_text().strip()]
    items = []
    for t in all_text:
        box = t.get_window_extent(renderer=renderer)
        if box.width > 0 and box.height > 0:  # 只审查有实际可见面积的文字
            items.append((t, box))
    issues = []
    canvas = fig.bbox
    for t, b in items:
        if b.x0 < 0 or b.y0 < 0 or b.x1 > canvas.x1 or b.y1 > canvas.y1:
            issues.append("outside canvas: " + t.get_text())
    for (ta, a), (tb, b) in itertools.combinations(items, 2):
        dx = min(a.x1, b.x1) - max(a.x0, b.x0)
        dy = min(a.y1, b.y1) - max(a.y0, b.y0)
        if dx > 1 and dy > 1:
            issues.append("text overlap: " + ta.get_text() + " / " + tb.get_text())
    return dict(figure=name, status="PASS" if not issues else "FAIL_AND_REDRAW", text_items=len(items), issues=issues)


def export(fig, stem):
    ASSETS.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):  # 按固定图示数量组织可编辑的视觉元素
        fig.savefig(ASSETS / f"{stem}.{ext}", dpi=600, facecolor="white", edgecolor="none")  # 导出固定物理尺寸、600dpi预览及可编辑矢量文件


def draw_performance(experiments):
    fig = plt.figure(figsize=(7.2, 3.4), facecolor="white")  # 采用双栏宽度与竖向柱状图所需的高度
    for j, e in enumerate(experiments):  # 按预定几何位置组织面板或示意元素
        x0 = .135 + j * .48  # 给每栏的数值刻度标签保留左边距
        ax = fig.add_axes((x0, .17, .30, .66))  # 固定两栏独立坐标轴的位置和大小
        top = 430 if j == 0 else 46  # 每栏独立线性范围，均从零开始，并在柱顶上方留出倍数标签空间
        ticks = [0, 100, 200, 300, 400] if j == 0 else [0, 10, 20, 30, 40]  # 两栏按各自量级设置易读刻度
        offsets = np.array([-.12, 0, .12])  # 三次重复测量在同一柱宽内的固定横向偏移
        medians = {}  # 保存中位吞吐量，供箭头端点与柱高保持一致
        for i in range(3):  # 用淡线连接同一配对测量的两个真实点
            ax.plot([offsets[i], 1 + offsets[i]], [e["rates"][e["reference"]][i] / 1000, e["rates"]["OpenGCMC"][i] / 1000], color=PALETTE["teal"], lw=.7, alpha=.18, zorder=.5)  # 配对线置于柱体之下，避免压住数据点
        for x, engine in [(0, e["reference"]), (1, "OpenGCMC")]:  # 每栏两根竖向柱：参照软件与 OpenGCMC
            values = np.asarray(e["rates"][engine]) / 1000  # 换算为 10³ 次/秒，与坐标刻度一致
            medians[engine] = statistics.median(values)  # 柱高取三次测量的中位数
            open_engine = engine == "OpenGCMC"
            fill = PALETTE["pale_teal"] if open_engine else PALETTE["pale_slate"]  # 按方法身份而非数值大小分配稳定颜色
            edge = PALETTE["teal"] if open_engine else PALETTE["slate"]  # 深色点与轮廓保持清晰
            ax.bar(x, medians[engine], width=.52, color=fill, edgecolor=edge, linewidth=.7, zorder=1)  # 实色柱从零基线表示中位吞吐量
            ax.scatter(x + offsets, values, s=18, color=edge, edgecolors="white", linewidths=.55, zorder=3)  # 显示全部三个测量点并使用固定横向偏移
        y0, y1 = medians[e["reference"]], medians["OpenGCMC"]  # 箭头从参照中位值指向 OpenGCMC 中位值
        ax.plot([.26, .5], [y0, y0], color=PALETTE["slate"], lw=.7, ls=(0, (2, 2)), zorder=.6)  # 虚线把参照柱顶水平引到箭头起点，明确箭头的起算高度
        ax.add_patch(FancyArrowPatch((.5, y0), (.5, y1), arrowstyle="-|>", mutation_scale=14, linewidth=1.7, color=PALETTE["teal"], shrinkA=0, shrinkB=0, zorder=2))  # 竖向上升箭头直接表达性能提升方向
        ax.text(.5, y1 + top * .022, f"{e['median_paired_ratio']:.2f}×", fontsize=17, fontweight="bold", color=PALETTE["teal"], ha="center", va="bottom", zorder=4, bbox=dict(boxstyle="round,pad=.16", facecolor="white", edgecolor="none"))  # 倍数标注紧贴箭头尖端，白底避免配对线穿过文字
        ax.set_xlim(-.55, 1.55)  # 让两根柱居中并保留标签宽度
        ax.set_ylim(0, top)  # 保留真实零基线，避免视觉夸大
        ax.set_xticks([0, 1], [e["reference"], "OpenGCMC"])  # 直接标注软件，无需额外图例
        ax.set_yticks(ticks)  # 使用各栏独立、等间隔的线性刻度
        ax.set_ylabel("Move attempts (10³ / s)", labelpad=6)  # 给出吞吐量单位，不加入测试条件
        ax.tick_params(axis="x", length=0, pad=8)  # 去除类别刻度线并保持标签间距
        ax.tick_params(axis="y", length=3, width=.7, pad=4)  # 克制地显示数值轴刻度
        for side in ("right", "top"):  # 按固定图示数量组织可编辑的视觉元素
            ax.spines[side].set_visible(False)  # 删除多余边框，突出真实数据柱
        for side in ("left", "bottom"):  # 保留较淡的坐标轴线作为数值参照
            ax.spines[side].set_color(PALETTE["slate"])
        ax.get_xticklabels()[1].set_color(PALETTE["teal"])  # 用主色突出 OpenGCMC 类别名
        ax.get_xticklabels()[1].set_fontweight("bold")  # 使用字重强化方法身份
        fig.text(.035 + j * .48, .95, "ab"[j], fontsize=11, fontweight="bold", va="center")  # 将面板字母放在数据区外左上方
    qa = audit_text(fig, "opengcmc_performance")
    export(fig, "opengcmc_performance")  # 同步导出当前画布的全部交付格式
    plt.close(fig)  # 导出后释放画布
    return qa


def text(ax, x, y, label, size=9, weight="normal", color=None, align="center", valign="center"):
    return ax.text(x, y, label, fontsize=size, fontweight=weight, color=color or PALETTE["ink"], ha=align, va=valign, linespacing=1.2)  # 统一图解文字的大小、字重、对齐和行距


def module(ax, x, y, w, h, tint="pale_teal", border=True):
    base = np.array(to_rgb(PALETTE[tint]))  # 从语义调色板取得模块色
    for k in range(36):  # 按固定图示数量组织可编辑的视觉元素
        f = .20 + .30 * k / 35  # 渐变幅度克制，让正文和箭头占主导
        color = tuple((1 - f) * np.ones(3) + f * base)  # 混合白色生成淡色矢量渐变条带
        ax.add_patch(Rectangle((x, y + k * h / 36), w, h / 36 + .0003, facecolor=color, edgecolor="none", zorder=0))  # 用矢量窄条保留可编辑的低对比渐变
    if border:  # 根据图标职责选择显示对应边界或细节
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=PALETTE["line"], linewidth=.7, zorder=1))  # 用细边界定义算法模块而非装饰框


def arrow(ax, start, end, optional=False, rad=0, color=None):
    patch = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8, linewidth=.85, color=color or (PALETTE["purple"] if optional else PALETTE["teal"]), linestyle=(0, (3, 2)) if optional else "-", connectionstyle=f"arc3,rad={rad}", zorder=2)  # 实线代表实现流程，虚线专指可选调用
    ax.add_patch(patch)  # 将方向箭头置于模块填充上方


def elbow(ax, points, optional=False, color=None):
    c = color or (PALETTE["purple"] if optional else PALETTE["teal"])  # 可选路径使用统一淡紫语义
    for start, end in zip(points[:-2], points[1:-1]):  # 按固定图示数量组织可编辑的视觉元素
        ax.plot([start[0], end[0]], [start[1], end[1]], color=c, lw=.85, linestyle=(0, (3, 2)) if optional else "-", zorder=2)  # 正交连线绕开文字与模块内部
    arrow(ax, points[-2], points[-1], optional=optional, color=c)  # 通过方向连线表达数据流或状态复用关系


def molecule_glyph(ax, centre, scale=.035, tone="teal", variant=0, opacity=1):
    x, y = centre  # 以可编辑中心坐标放置抽象树状候选构象
    shapes = [[(-1, 0), (-.33, .6), (.33, -.15), (1, .5)], [(-1, -.2), (-.33, .35), (.33, .75), (1, .15)], [(-1, .4), (-.33, -.1), (.33, .4), (1, -.1)]]  # 三种示意构象保持相同树状连接
    coords = np.array(shapes[variant % 3]) * [scale, scale * 1.5] + [x, y]  # 按比例生成横向构象草图
    ax.plot(coords[:, 0], coords[:, 1], lw=1.2, color=PALETTE[tone], alpha=opacity, zorder=3)  # 用连杆表示同一分子的候选构象
    ax.scatter(coords[:, 0], coords[:, 1], s=24, c=PALETTE[tone], alpha=opacity, edgecolors="white", linewidths=.45, zorder=4)  # 球节点仅为分子增长的抽象示意


def lattice_glyph(ax, x, y, sx=.04, sy=.028, neighbourhood=False):
    for i in range(3):  # 按固定图示数量组织可编辑的视觉元素
        ax.plot([x + i * sx * .22, x + 2 * sx + i * sx * .22], [y + i * sy, y + i * sy], color=PALETTE["line"], lw=.9)  # 倾斜水平线表示周期胞内网格
        ax.plot([x + i * sx, x + i * sx + 2 * sx * .22], [y, y + 2 * sy], color=PALETTE["line"], lw=.9)  # 第二组网格方向体现周期几何
    for i in range(3):  # 按固定图示数量组织可编辑的视觉元素
        for j in range(3):  # 按固定图示数量组织可编辑的视觉元素
            ax.scatter(x + i * sx + j * sx * .22, y + j * sy, s=22, color=PALETTE["teal"] if (i + j) % 2 else PALETTE["slate"], edgecolors="white", linewidths=.4, zorder=3)  # 交替低饱和节点使宿主网格易读
    if neighbourhood:  # 根据图标职责选择显示对应边界或细节
        ax.add_patch(Circle((x + sx * 1.22, y + sy), .038, fill=False, edgecolor=PALETTE["teal"], lw=1, linestyle=(0, (3, 2)), zorder=2))  # 虚线圆强调局部候选邻域而非假定真实结构


def chip_glyph(ax, centre, tone="teal", energy=False):
    x, y = centre  # 将执行后端图标置于各自标题上方
    ax.add_patch(Rectangle((x - .037, y - .024), .074, .048, facecolor="white", edgecolor=PALETTE[tone], lw=1.1))  # 芯片边界表示执行后端的职责范围
    for k in [-.023, -.008, .008, .023]:  # 按固定图示数量组织可编辑的视觉元素
        ax.plot([x + k, x + k], [y + .024, y + .034], color=PALETTE[tone], lw=.9)  # 上侧引脚构成简洁执行图标
        ax.plot([x + k, x + k], [y - .024, y - .034], color=PALETTE[tone], lw=.9)  # 下侧引脚保持视觉对称
    if energy:  # 根据图标职责选择显示对应边界或细节
        for i in range(3):  # 按固定图示数量组织可编辑的视觉元素
            for j in range(4):  # 按固定图示数量组织可编辑的视觉元素
                ax.add_patch(Rectangle((x - .026 + .015 * j, y - .013 + .011 * i), .009, .006, facecolor=PALETTE[tone], edgecolor="none", alpha=.65))  # 小并行单元示意CUDA能量计算而非整条MC驻留
    else:  # 根据图标职责选择显示对应边界或细节
        ax.plot([x - .025, x - .012, x, x + .012, x + .025], [y, y + .012, y - .012, y + .012, y], color=PALETTE[tone], lw=1)  # 芯片内部折线表示本机采样循环


def draw_architecture():
    fig = plt.figure(figsize=(9.0, 6.2), facecolor="white")  # 放大图示以容纳空间结构、候选构象及状态循环
    ax = fig.add_axes((.025, .025, .95, .95))  # 使用开放分组和均匀白边，避免满屏文字方框
    ax.set_xlim(0, 1)  # 归一化坐标便于整体编辑
    ax.set_ylim(0, 1)  # 保持三层算法结构的固定相对顺序
    ax.set_axis_off()  # 科研架构采用无坐标轴画布
    text(ax, .004, .974, "a", 12, "bold", align="left")  # 面板字母位于开放分组左端
    text(ax, .049, .974, "Reusable representations", 11, "bold", align="left")  # 第一层标题说明可复用对象
    lattice_glyph(ax, .09, .864, sx=.040, sy=.027, neighbourhood=True)  # 设置图示元素的位置、方向和可读标签
    text(ax, .14, .818, "Periodic host\n+ site parameters", 9.5)  # 周期空间图示下直接标注输入内容
    for i, count in enumerate([3, 5, 2]):  # 根据预设位置组织面板或图示元素
        yy = .865 + .018 * i  # 为三行CSR候选索引保留充足间距
        ax.add_patch(Rectangle((.296, yy), .012, .010, facecolor=PALETTE["lavender"], edgecolor=PALETTE["purple"], lw=.6))  # 偏移条作为CSR行指针示意
        for j in range(count):  # 按固定图示数量组织可编辑的视觉元素
            ax.add_patch(Rectangle((.321 + .022 * j, yy), .016, .010, facecolor=PALETTE["teal"], edgecolor="none", alpha=.5 + .18 * (i % 2)))  # 非等长索引行直观表示稀疏候选表
    text(ax, .367, .818, "Exact CSR\nNeighbours", 9.5)  # 用文字补充稀疏行图示的算法含义
    for i in range(3):  # 按固定图示数量组织可编辑的视觉元素
        for j in range(5):  # 按固定图示数量组织可编辑的视觉元素
            ax.add_patch(Rectangle((.548 + .026 * j, .861 + .018 * i), .021, .013, facecolor=PALETTE["teal"], edgecolor="white", lw=.4, alpha=.35 + .22 * i))  # 参数表以行色深强调预绑定参数复用
    ax.add_patch(Rectangle((.542, .856), .140, .022, fill=False, edgecolor=PALETTE["teal"], lw=.8))  # 高亮一行复用的混合参数
    text(ax, .61, .818, "Bound mixed\nparameter rows", 9.5)  # 参数表图示下保留准确名称
    for i in range(4):  # 按固定图示数量组织可编辑的视觉元素
        for j in range(6):  # 按固定图示数量组织可编辑的视觉元素
            ax.add_patch(Rectangle((.78 + .024 * j, .857 + .014 * i), .020, .011, facecolor=PALETTE["purple"], alpha=.16 + .13 * ((i + j * 2) % 5), edgecolor="white", lw=.3))  # 非均匀体素颜色仅表示提议概率的概念分布
    text(ax, .849, .818, "Frozen voxel alias\nq(r) > 0", 9.5, color=PALETTE["purple"])  # 明确体素图是全支持提议而非能量目标
    arrow(ax, (.213, .884), (.286, .884))  # 固定宿主用于构建精确邻域
    arrow(ax, (.446, .884), (.534, .884))  # 邻域与位点参数形成绑定查询表示
    # 独立链扇出图位于宿主下方，避免穿过空间和参数表连线。
    ax.plot([.099, .116, .116], [.749, .749, .722], color=PALETTE["slate"], lw=.8)  # 共享宿主只分发固定数据
    ax.plot([.116, .116], [.749, .776], color=PALETTE["slate"], lw=.8)  # 扇出竖线连接三个独立链入口
    for i, yy in enumerate([.722, .749, .776]):  # 按固定图示数量组织可编辑的视觉元素
        ax.plot([.116, .14, .16, .18], [yy, yy, yy + .007 * ((i % 2) * 2 - 1), yy], color=PALETTE["teal"], lw=.9)  # 不同轨迹示意各链独有的状态与随机数
        ax.scatter([.18], [yy], s=12, color=PALETTE["teal"], zorder=3)  # 每条独立链以单独状态节点收束
    text(ax, .282, .749, "Shared host\nIndependent chains", 9, color=PALETTE["muted"])  # 扇出图的标签不共享guest状态
    text(ax, .004, .681, "b", 12, "bold", align="left")  # 第二层突出核心采样循环
    text(ax, .049, .681, "Exact trial-to-state loop", 11, "bold", align="left")  # 保留清楚的算法主线标题
    # 浅色开放带只组织核心循环，不作为四个重复的文字框。
    module(ax, .043, .474, .197, .164, tint="pale_coral", border=False)  # 用浅色开放区域组织相应算法图示
    module(ax, .265, .474, .194, .164, tint="pale_teal", border=False)  # 淡青区域承托精确能量图示
    module(ax, .495, .474, .228, .164, tint="lavender", border=False)  # 淡紫区域组织正反向CBMC权重
    module(ax, .75, .474, .207, .164, tint="pale_green", border=False)  # 淡绿区域突出状态接受决策  # 以低对比渐变承托图示和主流程
    for i, yy in enumerate([.537, .563, .590]):  # 按固定图示数量组织可编辑的视觉元素
        molecule_glyph(ax, (.14, yy), scale=.045, variant=i, tone="coral", opacity=.38 + .25 * i)  # 三个分子构象直接表达批量树状增长候选
    text(ax, .14, .497, "Trial batch", 9.5, "bold", color=PALETTE["coral"])  # 候选构象下直接标注批量输入
    lattice_glyph(ax, .322, .548, sx=.030, sy=.019, neighbourhood=False)  # 设置图示元素的位置、方向和可读标签
    ax.add_patch(Circle((.362, .567), .042, facecolor="none", edgecolor=PALETTE["teal"], lw=.9, linestyle=(0, (3, 2))))  # 精确邻域圆展示候选过滤仍需实际距离查询
    ax.scatter([.362], [.567], s=46, color="#C69051", edgecolors="white", linewidths=.7, zorder=5)  # 琥珀色查询点与宿主区分
    for point in [(.322, .548), (.352, .548), (.359, .567), (.329, .567)]:  # 按固定图示数量组织可编辑的视觉元素
        ax.plot([.362, point[0]], [.567, point[1]], color="#C69051", alpha=.55, lw=.7, zorder=2)  # 局部查询连线表示实际候选相互作用
    text(ax, .363, .497, "Exact energies", 9.5, "bold", color=PALETTE["teal"])  # 精确能量只依赖候选过滤后的真实计算
    weights = [.025, .055, .038, .070]  # 图形仅示意正权重，不对应测得数据
    for i, hh in enumerate(weights):  # 按固定图示数量组织可编辑的视觉元素
        ax.add_patch(Rectangle((.558 + i * .027, .534), .017, hh, facecolor=PALETTE["purple"], edgecolor="white", lw=.5, alpha=.4 if i != 3 else .95))  # 柱形图标示意CBMC权重和一次选择
    ax.scatter([.647], [.613], s=20, color=PALETTE["purple"], zorder=4)  # 选择节点强调由权重形成候选选择
    arrow(ax, (.551, .619), (.663, .619), rad=-.25, color=PALETTE["purple"])  # 上弧表达正向增长权重
    arrow(ax, (.663, .525), (.551, .525), rad=-.22, color=PALETTE["purple"])  # 反弧表达对原构象的回溯计算
    text(ax, .611, .488, "Forward + retrace CBMC", 9.5, "bold", color=PALETTE["purple"])  # 正反向弧与文字共同保留算法含义
    diamond = np.array([[.865, .615], [.916, .568], [.865, .521], [.814, .568], [.865, .615]])  # 菱形用于显示接受与拒绝的决策语义
    ax.fill(diamond[:, 0], diamond[:, 1], facecolor="white", edgecolor=PALETTE["green"], lw=1)  # 决策图标采用白心细描边
    text(ax, .865, .568, "A / R", 10, "bold", color=PALETTE["green"])  # 用简洁标识防止菱形内部拥挤
    text(ax, .865, .652, "Accept / reject", 9.5, "bold", color=PALETTE["green"])  # 决策名称上移，为下方直接状态连线留出空间
    arrow(ax, (.21, .566), (.29, .566))  # 构象批次进入精确局部能量查询
    arrow(ax, (.415, .566), (.528, .566))  # 精确能量用于形成CBMC权重
    arrow(ax, (.69, .566), (.804, .566))  # 权重比进入详细平衡决策
    arrow(ax, (.367, .791), (.363, .644))  # 用单条直线把精确候选表连接到能量查询
    arrow(ax, (.61, .791), (.4, .635))  # 参数表直接斜连至精确能量模块，避免多次直角折返
    arrow(ax, (.849, .791), (.611, .641), optional=True)  # 可选体素提议以单条虚线直接进入权重模块
    text(ax, .775, .688, "Optional", 9, color=PALETTE["purple"])  # 可选标签与斜向虚线分离，保持清晰留白
    text(ax, .611, .449, "1 / [V q(r)]", 9.5, color=PALETTE["purple"])  # 明确正反向均使用相同的提议密度校正
    # 状态缓存用两个微型周期盒表示，循环箭头围绕实际状态而非文字卡片。
    lattice_glyph(ax, .327, .326, sx=.030, sy=.019, neighbourhood=False)  # 设置图示元素的位置、方向和可读标签
    molecule_glyph(ax, (.367, .347), scale=.023, tone="purple", variant=1)  # 放置抽象图示并保持模块边界清晰
    text(ax, .367, .300, "Guest environment", 9.5, "bold")  # 缓存微型环境下说明可复用对象
    lattice_glyph(ax, .76, .326, sx=.030, sy=.019, neighbourhood=False)  # 设置图示元素的位置、方向和可读标签
    molecule_glyph(ax, (.8, .347), scale=.023, variant=0, tone="green")  # 设置图示元素的位置、方向和可读标签
    text(ax, .8, .300, "Accepted guest state", 9.5, "bold", color=PALETTE["green"])  # 已接受状态以独立微型胞体展示
    text(ax, .799, .270, "Invalidate → lazy rebuild", 9, color=PALETTE["muted"])  # 状态改变后使guest缓存失效并按需重建
    arrow(ax, (.865, .518), (.8, .382), color=PALETTE["green"])  # 接受分支直接提交状态，去掉外侧多段绕行
    text(ax, .887, .433, "accept", 9, color=PALETTE["green"])  # 接受标签置于简短斜线右侧
    arrow(ax, (.744, .347), (.423, .347))  # 已接受状态在改变时产生新的缓存环境
    text(ax, .580, .372, "on change", 9, color=PALETTE["muted"])  # 变化事件独立置于状态连线上方
    arrow(ax, (.365, .389), (.365, .467))  # 缓存环境返回精确能量查询
    text(ax, .247, .422, "Index exclusion", 9, color=PALETTE["muted"])  # 原始索引排除保留在能量查询侧
    arrow(ax, (.828, .544), (.41, .382), rad=-.20, color=PALETTE["green"])  # 拒绝复用采用一条平滑回弧，直接返回原有客体环境
    text(ax, .582, .424, "reject: reuse", 9, color=PALETTE["green"])  # 复用标签靠近平滑回弧并避开下方状态改变连线
    text(ax, .004, .191, "c", 12, "bold", align="left")  # 第三层清楚展示执行边界
    text(ax, .049, .191, "Execution routes", 11, "bold", align="left")  # 标题将实现路线与上方算法逻辑分开
    chip_glyph(ax, (.18, .115), tone="blue")  # 放置抽象图示并保持模块边界清晰
    text(ax, .18, .063, "C++ native sampler", 10, "bold")  # C++图标对应本机完整采样循环
    text(ax, .18, .031, "Single component · LJ / DSF", 9)  # 保留真实实现范围
    ax.add_patch(Circle((.513, .115), .037, facecolor="white", edgecolor=PALETTE["blue"], lw=1))  # 开放圆形节点代表通用Python编排入口
    text(ax, .513, .115, "{ }", 13, "bold", color=PALETTE["blue"])  # 大括号图标表达可编程通用接口
    text(ax, .513, .063, "Python general sampler", 10, "bold")  # 通用层与native路线并列展示
    text(ax, .513, .031, "Mixtures · Ewald · target correction", 9)  # 同一文字行简洁列出通用层能力
    chip_glyph(ax, (.842, .115), tone="blue", energy=True)  # 放置抽象图示并保持模块边界清晰
    ax.add_patch(Rectangle((.785, .080), .114, .079, fill=False, edgecolor=PALETTE["blue"], linestyle=(0, (3, 2)), lw=.8))  # 虚线外界明确CUDA只负责可选能量调用
    text(ax, .842, .063, "CUDA energy backend", 10, "bold")  # 避免把能量后端误读成全GPU采样
    text(ax, .842, .031, "Resident tables · workspace", 9)  # 以直接标签说明设备驻留和工作区复用
    arrow(ax, (.566, .115), (.779, .115), optional=True)  # Python通用层可调用CUDA能量后端
    qa = audit_text(fig, "opengcmc_architecture")
    export(fig, "opengcmc_architecture")  # 同步导出当前画布的全部交付格式
    plt.close(fig)  # 导出后关闭画布避免状态泄漏
    return qa


def main():
    experiments, rows, pairs = load_and_validate()
    write_source_data(experiments, rows, pairs)
    qa = [draw_performance(experiments), draw_architecture()]
    (DATA / "layout_qa.json").write_text(json.dumps(qa, indent=2) + "\n")
    print(json.dumps(qa, indent=2))
    if any(item["issues"] for item in qa):
        raise SystemExit("Text geometry requires correction; see source_data/layout_qa.json")


if __name__ == "__main__":
    main()
