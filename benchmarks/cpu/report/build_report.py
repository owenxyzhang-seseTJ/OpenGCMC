"""Build the editable OpenGCMC experiment report from archived measurements."""

from pathlib import Path
import sys

from layout import Report, accent, bracket, frac, mi, mn, mo, mtext, sub, sup, under
from report_data import HERE, REAL_CASES, checkpoint, read, real_run, seconds, timing_runs


TITLE = 'OpenGCMC 两阶段采样与双平台 CPU 实验报告'
FIGURES = HERE / 'figures'
NIST_PURE = 'https://www.nist.gov/mml/csd/chemical-informatics-group/tmmc-reference-adsorption-isotherms-co2-metal-organic-frameworks'
NIST_MIX = 'https://www.nist.gov/mml/csd/chemical-informatics-group/tmmc-reference-adsorption-isotherms-n2co2-mixtures-metal-organic'
EQ, MINUS, PLUS = mo('='), mo('−'), mo('+')


def sym(name, index):
    return sub(mi(name), mi(index, upright=len(index) > 1))


def minimum(value):
    return mi('min', True) + bracket(mn(1) + mo(',') + value)


def magnitude(value):
    return bracket(value, '|', '|')


def formal_summary(r):
    r.title(TITLE, '计算日期 2026 年 11 月 23 日　　Apple M5 与HPC Intel Xeon CPU Max 9468')
    r.heading('摘要')
    r.para(
        '本报告检验两阶段接受能否在有限 Ewald 目标势下保持巨正则平衡分布，并测量它在指定统计精度下的计算成本。'
        '使用可枚举双孔系统判定孔位和构象采样，以 NIST 的转移矩阵蒙特卡洛数据复核真实吸附分布。'
        '性能部分分别测量接受规则、原生增量内核、框架场缓存以及与 RASPA3 的整程序差异。'
    )
    r.para(
        '双孔模型中，最粗参考势复现了整个总分子数分布，却把 A/B 占据比从 1.1463 改成 0.8724。'
        '按精确残差校正后，两种接受规则都恢复两孔的占据和构象真值。'
        '与 NIST 参考分布的总变差在纯 CO₂ 工况约为 0.024，N₂/CO₂ 联合计数约为 0.086。'
    )
    r.para(
        '共享候选对照中，两阶段在 IRMOF-1 的首达标速度分别为单阶段的 1.22 倍（M5）和 1.28 倍（HPC）。'
        '在 UiO-66 中，相同候选下单阶段更快，M5 为 18.19 对 49.28 s，HPC为 38.09 对 98.05 s。'
        'UiO-66 的两阶段生产时间约减三成，载量相关因子增至 2.06 倍，超过两台机器约 1.44 倍的成本边界。'
    )
    r.para(
        '与 RASPA3 的匹配实验中，两端均获得 IRMOF-1 混合物和低载 TON 正丁烷的达标时间。'
        '对二甲苯低压点的 OpenGCMC 达标，RASPA3 到预算上限仍未通过。'
        '对二甲苯中高压点和 TON 100 Pa 的两程序结果均作为右截尾保留。'
    )
    r.table(
        '表 1　实验问题与证据',
        ('问题', '实验', '验收观测'),
        [
            ('两阶段的平衡分布', '双孔枚举与独立 NIST 复核', '概率流、P(N)、孔占据、构象及联合计数'),
            ('减少精算的净收益', '共享候选配对、逐候选诊断、网格粗化', '精算次数、接受率、相关因子、首达标墙钟'),
            ('真实吸附的端到端性能', 'Ni_TPT_TATBA、TON、IRMOF-1 对 RASPA3', '载量、反式比例、组分与联合计数'),
            ('冻结规则能否迁移', 'IRMOF-1 开发、UiO-66 留出及事后配对', '先导预测、确认墙钟、热缓存同候选首达标'),
            ('原生执行与调度', '增量更新、冷热场和 MCP/Slurm 实测', '配对轨迹、完整批跑成本及持久提交'),
        ],
        (1.52, 2.46, 2.94),
        left=(0, 1, 2),
        note='全部墙钟结果来自 CPU。报告中的计时、收敛诊断和末态能量分别对应原始检查点、抽样轨迹和能量对账记录。',
    )


def design(r):
    r.heading('1 实验设计与统一判据', 1, True)
    r.heading('1.1 计算平台')
    r.table(
        '表 2　两台 CPU 平台与软件配置',
        ('项目', 'Apple M5', 'HPC Intel 节点'),
        [
            ('CPU 与操作系统', 'M5 4P+6E，macOS 26.5 ARM64', 'Xeon CPU Max 9468，Linux x86_64'),
            ('原生编译器与 Python', 'Apple Clang 21.0.0，Python 3.13.13', 'GCC 12.3.0，Python 3.11.7'),
            ('并行配置', '单链单进程，四链同机并行', '每组分配 4 个 CPU，四链同机并行'),
            ('RASPA3', '3.1.0', '3.0.30'),
        ],
        (1.62, 2.65, 2.65),
        left=(0, 1, 2),
        note='原生库使用本地源码 206122c，采用可移植 C++17 构建，禁用 OpenMP、fast math 和浮点乘加合并。',
    )
    r.heading('1.2 计时和链间诊断')
    r.para(
        '每个计时工况启动四条独立链，两条空孔，另两条使用分别生成的预装构型。'
        '配对使用相同的构型文件和相同编号的种子。从并行执行入口计时，取四条链到同一检查点的最大墙钟。'
        '计时涵盖模型装载、缓存读取或现场建场、预热、生产及观测采集。'
        '诊断由保存的样本离线复算。Slurm 排队、原生编译和调度轮询另有记录。'
    )
    r.para(
        '每个观测量的蒙特卡洛标准误差（MCSE）取链内和链间两种估计的较大者。'
        '链内估计使用积分相关因子和 Geyer 初始正序列，链间估计根据各链均值的离散度。'
        '分链秩归一化 R̂ 取折叠与未折叠诊断的较大者。表 3 每项有数值门槛的观测均须满足对应 MCSE，且 R̂ < 1.01。'
        '满足判据的最早保存检查点定义首达标。右截尾表示在所保存检查点中截至预算上限没有通过。'
    )
    r.table(
        '表 3　预设 MCSE 门槛',
        ('体系', '载量门槛／分子每超胞', '附加门槛或诊断'),
        [
            ('ZIF-8／纯 CO₂', 'CO₂ 0.25', '完整 P(N) 与 NIST 比较'),
            ('ZIF-8／N₂+CO₂', 'N₂ 0.15，CO₂ 0.40', 'P(N₂ ≥ 2 且 CO₂ ≥ 13) 0.025'),
            ('IRMOF-1／CO₂+CH₄', 'CO₂、CH₄ 均 0.20', 'P(CO₂ ≥ 3 且 CH₄ ≥ 4) 0.025'),
            ('Ni_TPT_TATBA／对二甲苯', '0.02／0.50／40 Pa\n0.20／0.65／1.20', '组间起跑均值及完整分布'),
            ('TON／正丁烷', '2／100 Pa 对应 0.15／0.35', '反式比例 0.025／0.015'),
            ('UiO-66／CO₂', 'CO₂ 0.25', '留出执行规则的时间预测'),
        ],
        (2.14, 2.29, 2.49),
        left=(0, 1, 2),
        note='事件中的 CO₂、CH₄、N₂ 指其在超胞中的分子数。OpenGCMC 的预算用尝试数，RASPA3 用周期，各周期移动数随占据数变化。',
    )


def detailed_balance(r):
    r.heading('2 两阶段 CBMC 的详平衡', 1, True)
    r.heading('2.1 正向生长与反向回溯')
    r.para(
        'U 和 U₀ 分别表示目标势和参考势。构型测度归一化，根的均匀测度为体积测度除以 V。'
        '根从空间密度 ρ 抽样时，相对密度为 Vρ。柔性内坐标按模型规定的测度抽取。'
    )
    n = sym('N', 's')
    zv = sym('z', 's') + mi('V')
    r.math(mi('π') + bracket(mi('X')) + mo('∝')
           + mi('exp', True) + bracket(MINUS + mi('β') + mi('U') + bracket(mi('X')))
           + under(mo('∏'), mi('s')) + frac(sup(bracket(zv), n), n + mo('!')))
    r.para(
        '上式在有序坐标上积分，Nₛ! 处理同种分子的不可分辨性，β 为热能倒数。'
        '下文使用无序构型，组合因子并入坐标测度，反向删去新分子给出 1/(Nₛ+1)。'
    )
    w = sym('w', 'ij')
    g = sym('g', 'i')
    u0 = sub(mi('U'), mn(0))
    exp_term = mi('exp', True) + bracket(MINUS + mi('β') + sub(mi('Δ'), mi('i')) + u0)
    wbar = sub(accent(mi('w'), '¯'), mi('i'))
    k = sym('K', 'i')
    r.math(w + EQ + frac(exp_term, g) + mo(',') + wbar + EQ
           + frac(mn(1), k) + under(mo('∑'), mi('j')) + w + mo(',')
           + mi('W') + EQ + under(mo('∏'), mi('i')) + wbar)
    r.para(
        '正向每一步生成 Kᵢ 个试探，按 wᵢⱼ 除以权重和选择候选。'
        '反向在相同前缀中把实际坐标放入随机槽位，再生成 Kᵢ−1 个试探。'
        '未选试探的密度在增广路径中抵消。被选槽位的正向因子是 gᵢwᵢⱼ 除以权重和，反向槽位的因子为 1/Kᵢ。'
        '因此每一步的正反提议比为参考玻尔兹曼因子除以平均试探权重。'
    )
    r.heading('2.2 插入 删除和再生长')
    r.para(
        'p₊、p₋ 为插入和删除概率，Nₛ 为移动前组分 s 的分子数，zₛ 为其活度。'
        '物种选择与状态无关，正反相消。删除均匀选一个同类分子，并在其余分子的环境中回溯。'
    )
    pplus, pminus = sub(mi('p'), mo('+')), sub(mi('p'), mo('−'))
    wnew, wold = sym('W', 'new'), sym('W', 'old')
    r.math(sub(mi('r'), mn(0) + mo(',') + PLUS) + EQ
           + frac(pminus, pplus) + frac(zv + wnew, n + PLUS + mn(1)) + mo(',')
           + sub(mi('r'), mn(0) + mo(',') + MINUS) + EQ
           + frac(pplus, pminus) + frac(n, zv + wold))
    r.para(
        '再生长共享分子、环境和后缀起点，固定前缀的能量与选择概率相消。内配分函数按生长的归一化测度计算。'
        '插入、删除、再生长的目标残差变化依次是新残差、负的旧残差及新旧残差之差。'
    )
    r.math(sym('z', 's') + EQ + frac(sym('f', 's'), sym('k', 'B') + mi('T') + sub(mi('q'), mtext('int')))
           + mo(',') + sub(mi('r'), mn(0) + mo(',') + mtext('regrow')) + EQ + frac(wnew, wold))
    r.heading('2.3 接受概率与平衡概率流')
    r0, rr = sub(mi('r'), mn(0)), sub(mi('r'), mi('R'))
    r.math(mi('R') + EQ + mi('U') + MINUS + u0 + mo(',')
           + rr + EQ + mi('exp', True) + bracket(MINUS + mi('β') + mi('Δ') + mi('R')))
    r.math(sub(mi('a'), mi('m')) + EQ + minimum(r0 + rr) + mo(',')
           + sub(mi('a'), mi('d')) + EQ + minimum(r0) + minimum(rr))
    r.para(
        '反向路径的参考比与残差比均取倒数。因为 min(1, r)/min(1, 1/r) = r，两种正反接受比均为两因子之积。'
        '参考提议覆盖目标允许移动时，增广路径上的概率流守恒，对试探积分后即得边际详平衡。'
        '同一路径的两阶段接受概率至多等于单阶段，两因子推动相反决定时严格更低。'
    )


def ewald_boundary(r):
    r.heading('3 精确残差与单位有效样本成本', 1, True)
    r.heading('3.1 直接 Ewald 的显式残差')
    r.para(
        'H 为固定框架，m 为选中的完整分子。B 包含框架和其余客体，再生长的旧分子和新分子使用同一个 B。'
        '以下用 S 表示结构因子，Q 表示净电荷。精确局部能减去参考局部能，并加上向 B 插入 m 的倒空间、自能和背景增量。'
    )
    rb = sym('R', 'B') + bracket(mi('m'))
    host = sub(mi('U'), mi('H') + mo(',') + mi('m'))
    li = sup(host, mi('LJ+real', True))
    grid = sup(host, mi('grid', True))
    gsum = under(mo('∑'), mi('i') + mo('<') + mi('j') + mo('∈') + mi('m'))
    er = frac(mi('erf', True) + bracket(mi('α') + sym('r', 'ij')), sym('r', 'ij'))
    r.math(rb + EQ + li + MINUS + grid
           + MINUS + sym('k', 'e') + gsum + sym('q', 'i') + sym('q', 'j') + er
           + PLUS + sub(mi('E'), mi('rec', True)) + PLUS + sub(mi('E'), mi('self', True))
           + PLUS + sub(mi('E'), mi('bg', True)))
    skb, skm = sub(accent(mi('S'), '¯'), mi('B,k')), sym('S', 'm,k')
    r.math(sub(mi('E'), mi('rec', True)) + EQ + under(mo('∑'), mi('k'))
           + sym('c', 'k') + bracket(mn(2) + mi('Re', True) + bracket(skb + skm)
                                    + PLUS + sup(magnitude(skm), mn(2))))
    qm, qb = sym('Q', 'm'), sym('Q', 'B')
    r.math(sub(mi('E'), mi('self', True)) + EQ + sym('c', 'self')
           + under(mo('∑'), mi('i') + mo('∈') + mi('m')) + sup(sym('q', 'i'), mn(2))
           + mo(',') + sub(mi('E'), mi('bg', True)) + EQ + sym('c', 'bg')
           + bracket(mn(2) + qb + qm + PLUS + sup(qm, mn(2))))
    r.para(
        '参考框架场包含 LJ 和静电近似。误差函数项来自分子内 Ewald 排除写法之差，rᵢⱼ 采用完整分子的内部距离。'
        '两套势共有的客体间 LJ、实空间静电及成键力场项代数抵消。'
        '在完整倒空间集合上，cₖ 取 2πkₑ/V 乘以 exp(−k²/4α²)/k²。共轭半谱使用相应倍权。'
        '自能系数为 −kₑα/√π，均匀背景系数为 −πkₑ/(2α²V)。'
        '本报告比较的吸附分子净电荷为零，背景增量为零。'
    )
    r.heading('3.2 相关性与失速边界')
    r.para(
        '对共享的提议核积分后，两阶段的非对角转移测度逐点至多等于单阶段。'
        '在同一目标分布上，这给出 Peskun 顺序。对均值为零的观测，'
        '两阶段的 Dirichlet 形式小于或等于同候选单阶段。'
        '泊松方程可解时，逆算子反转偏序，因此两阶段的平衡渐近方差弱增。'
    )
    p_d, p_m = sym('P', 'd'), sym('P', 'm')
    minus_p = bracket(mi('I') + MINUS + mi('P'))
    r.math(mi('I') + MINUS + p_d + mo('⪯') + mi('I') + MINUS + p_m + mo(',')
           + sup(sym('σ', 'P'), mn(2)) + EQ + mn(2)
           + bracket(mi('h') + mo(',') + sup(minus_p, MINUS + mn(1)) + mi('h'), '⟨', '⟩')
           + MINUS + bracket(mi('h') + mo(',') + mi('h'), '⟨', '⟩'))
    r.para(
        '设 c 的两个下标 q、E 分别指每次尝试的候选成本与单次精算成本，p 为每次尝试触发精算的概率，g 为指定观测的积分相关因子。'
        '单位有效样本成本与 g 乘以每次尝试的平均成本成正比。m、d 分别表示同候选单阶段和两阶段。'
    )
    cq, ce = sym('c', 'q'), sym('c', 'E')
    gd, gm = sym('g', 'd'), sym('g', 'm')
    pd, pm = sym('p', 'd'), sym('p', 'm')
    gamma = mi('γ')
    r.math(gd + bracket(cq + PLUS + pd + ce) + mo('<') + gm + bracket(cq + PLUS + pm + ce)
           + mo('⇒') + frac(cq, ce) + mo('<') + frac(pm + MINUS + gamma + pd, gamma + MINUS + mn(1)))
    r.para(
        'γ 为两阶段相关因子除以单阶段相关因子。右侧条件适用于 γ > 1。阈值为负时，降低候选成本到零仍无法获得两阶段收益。'
        '实测中也可直接用固定尝试数的生产时间比估算最大可容许的相关因子比。该边界依赖观测量和网格精度。',
        'Report Note',
    )


def double_pore(r):
    r.heading('4 双孔枚举与数值真值', 1, True)
    r.para(
        'A、B 两孔各有空、反式和两个旁式，共 16 个状态，两孔同时占据时含构象相关的跨孔相互作用。'
        '每次生长含两试探，zV = 0.51。插入、删除、完整再生长和后缀更新的概率依次为 0.31、0.24、0.30、0.15。'
        '正向根试探使用与两个孔各三种构象对应的六项非均匀概率 0.29、0.06、0.14、0.18、0.22、0.11。'
        '以精确配分函数给出真值，再枚举各增广路径的转移概率。'
    )
    data = read('two_pore.json')
    modes = data['runs'][-1]['modes']
    rows = []
    for label, result in (
        ('枚举真值', data['exact']),
        ('最粗参考势', modes['reference_only']),
        ('同候选单阶段', modes['reference_mh']),
        ('同候选两阶段', modes['delayed']),
    ):
        conf = ['／'.join(f'{100*x:.2f}' for x in pore) for pore in result['conformation_per_pore']]
        rows.append((label, '／'.join(f'{x:.5f}' for x in result['P_N']),
                     f"{result['occupancy_A_over_B']:.5f}", *conf))
    r.table(
        '表 4　粒子数与分孔构象分布',
        ('路径', 'P(0)／P(1)／P(2)', 'A/B 占据比', 'A 孔构象／%', 'B 孔构象／%'),
        rows, (1.13, 1.93, .89, 1.485, 1.485),
        note='每孔按反式／旁式+／旁式−排序。构象概率以该孔已经占据为条件。',
    )
    r.figure(FIGURES / 'double_pore.png',
             '图 1　最粗参考势置换孔位和构象标签。a 总分子数分布重合。b 两孔的边际占据概率发生互换。')
    r.para(
        '四档失真中，两种校正路径的最大详平衡概率流误差小于 9×10⁻¹⁸。M5 与 Intel 转移矩阵最大绝对差为 1.39×10⁻¹⁷。'
        '最粗参考势下，两阶段精算触发率由单阶段的 0.694 降到 0.425，载量相关因子却从 9.97 升到 19.25。'
    )
    limits = []
    for item in data['runs']:
        ms = item['modes']
        gm, gd = (ms[p]['integrated_correlation_factors'] for p in ('reference_mh', 'delayed'))
        pm, pd = (ms[p]['potential_evaluations_per_attempt'] for p in ('reference_mh', 'delayed'))
        g = gd['N'] / gm['N']
        cost = []
        for key in ('N', 'pore_A_minus_B'):
            gamma = gd[key] / gm[key]
            cost.append('任意有限值' if abs(gamma - 1) < 1e-12
                        else f'{(pm-gamma*pd)/(gamma-1):.3f}'.replace('-', '−'))
        limits.append((f"{item['distortion']:.2f}", f'{pd:.3f}', f'{g:.3f}', *cost))
    r.table(
        '表 5　解析失速界限',
        ('参考势失真', '两阶段精算触发率', '载量相关因子比',
         '载量成本比上限', '孔占据差成本比上限'),
        limits, (.94, 1.37, 1.33, 1.47, 1.81),
        note='成本比为一次候选生成与一次精算的成本之比。失真 0 为精确参考势，1 为最粗势。候选成本取零时，最粗档两阶段的载量有效样本成本仍为单阶段的 1.18 倍。',
    )


def nist(r):
    r.heading('5 NIST 平衡分布独立复核', 1, True)
    r.para(
        '两组 ZIF-8 模型采用 NIST 公布的 2×2×2 超胞及逐位点荷电力场。300 K 下，纯 CO₂ 的逸度为 0.50 bar，'
        '混合物中 N₂ 和 CO₂ 也各为 0.50 bar。实空间截断均为 12 Å。纯气使用线性移力 LJ，混合气使用直接截断 LJ 且关闭尾项。'
        '直接 Ewald 的 α 为 0.234637 Å⁻¹，整数方向上限为 7，倒空间球形截断为 1.286101 Å⁻¹。'
    )
    r.para(
        '对 NIST 的每个转移矩阵蒙特卡洛（TMMC）副本，将 log 概率加上粒子数组合与活度对数变化的内积后重新归一化。'
        '纯气合并四个副本，混合气合并三个副本。参考及经验分布均按超胞计数，在两者的并集上计算总变差。'
    )
    r.figure(FIGURES / 'nist.png',
             '图 2　ZIF-8 的计数分布。a 纯 CO₂ 的 NIST 曲线和两条 OpenGCMC 路径的经验分布。b NIST 混合联合分布，虚线右上方为 N₂ ≥ 2 且 CO₂ ≥ 13。')
    records = []
    for case, label in (('nist_co2_zif8', '纯 CO₂'), ('nist_n2_co2_zif8', 'N₂+CO₂')):
        runs = [read(f'm5/{case}-{mode}.json') for mode in ('reference-mh', 'delayed')]
        summaries = [a['NIST'] for a in runs]
        reference = list(zip(*summaries[0]['reference_replica_means']))
        reference = '／'.join(f'{sum(col)/len(col):.3f}' for col in reference)
        average = ['／'.join(f'{x:.3f}' for x in s['empirical_mean']) for s in summaries]
        records.append((label, reference, *average,
                        '／'.join(f"{s['total_variation']:.4f}" for s in summaries)))
    r.table(
        '表 6　合并载量与经验分布总变差',
        ('体系', 'NIST 平均', '单阶段平均', '两阶段平均', '总变差 单／两'),
        records, (.95, 1.46, 1.46, 1.46, 1.59),
        note='组分按 N₂／CO₂ 排序。联合事件概率为单阶段 0.325、两阶段 0.337，NIST 副本为 0.325–0.338。单个 NIST 副本相对合并参考的联合总变差为 0.044–0.065。',
    )
    r.table(
        '表 7　单阶段与两阶段首达标  单位 s',
        ('体系', 'M5 单／两', 'Intel 单／两', '首达标尝试数 单／两'),
        [
            (label, *['／'.join(seconds(real_run(plat, 'nist', f'{case}-{mode}'))
                                for mode in ('reference-mh', 'delayed')) for plat in ('m5', 'hpc')],
             stage)
            for case, label, stage in (
                ('nist_co2_zif8', '纯 CO₂', '20,000／60,000'),
                ('nist_n2_co2_zif8', 'N₂+CO₂', '60,000／120,000'),
            )
        ], (1.0, 1.72, 1.72, 2.48),
        note='每链预热 20,000 次，生产上限 240,000 次。建场与读取采用实际缓存状态。独立参考的计数实验和分孔枚举各自提供相应观测证据。',
    )


def controls(r):
    r.heading('6 同候选的接受规则对照', 1, True)
    r.para(
        'IRMOF-1 在 300 K 下以 CO₂ 与 CH₄ 逸度 0.50 和 1.50 bar 运行。单阶段 reference-mh 和两阶段 delayed 共用预建的 32³ 框架场、'
        '0.60 根偏置、10 个试探及直接 Ewald 目标势。同候选指给定当前状态时的候选生成规则相同。'
        'exact 以目标势逐候选求能，另用 8³ 框架场和客体索引重建分别衡量网格精度与索引更新的影响。'
    )
    records = []
    obs = []
    for key, label, trigger, accept in (
        ('exact', '逐候选精算', '逐试探', '52.49%'),
        ('reference-mh', '同候选单阶段', '100%', '53.51%'),
        ('delayed', '同候选两阶段', '67.41%', '48.65%'),
        ('delayed-mesh8', '两阶段 8³', '72.15%', '22.88%'),
        ('delayed-rebuild', '两阶段索引重建', '67.41%', '48.65%'),
    ):
        runs = [read(f'{root}/control-{key}.json') for root in ('m5/controls_warm', 'hpc/controls')]
        prods = [max(c['profiles'][-1]['production_wall_s'] for c in a['chains']) for a in runs]
        records.append((label, *[seconds(a) for a in runs], '／'.join(f'{p:.2f}' for p in prods), trigger, accept))
        terms = ('count_0', 'count_1', 'joint_probability')
        rates = [[checkpoint(a)['observables'][k]['ess']/p for k in terms] for a, p in zip(runs, prods)]
        obs.append((label, *['／'.join(f'{x:.1f}' for x in a) for a in rates]))
    r.table(
        '表 8　IRMOF-1 首达标时间与接受统计  时间单位 s',
        ('路径', 'M5 首达标', 'Intel 首达标', '完整生产 M5／Intel', '精算触发', '接受率'),
        records, (1.53, .94, .99, 1.72, .87, .87),
        note='每条链生产 240,000 次。精算触发率以进入参考接受规则的有效提议为分母。最终接受率以全部移动尝试为分母。',
    )
    r.table(
        '表 9　完整生产的有效样本率  ESS/s',
        ('路径', 'M5 CO₂／CH₄／联合', 'Intel CO₂／CH₄／联合'),
        obs, (2.04, 2.44, 2.44),
        note='ESS/s 用四链合并 ESS 除以最长的一条链的生产墙钟。联合事件为 CO₂ 至少 3 个且 CH₄ 至少 4 个。',
    )
    r.para(
        '相同 32³ 候选下，两阶段把每次尝试评估的精确分子残差项从 1.3495 降到 0.9001。'
        '首达标加速为 M5 的 1.22 倍和 Intel 的 1.28 倍。CH₄ 的 ESS/s 分别降低约 22% 和 18%，'
        '与其相关因子比约 1.57 超过生产成本边界 1.22／1.29 一致。'
    )
    r.para(
        '8³ 网格下最终接受率降至 22.88%。CO₂ 首个检查点的载量为 3.325，R̂ 为 1.152。'
        '32³ 两阶段完整轨迹的载量为 2.660。8³ 要到 240,000 次才达标。'
        '32³ 的增量索引与重建输出相同的计数序列和接受统计，在这个低占据体系中重建略快。'
    )


def kernel_cache_dispatch(r):
    r.heading('7 增量内核 缓存和任务执行', 1, True)
    r.heading('7.1 增量更新内核')
    m5 = read(Path('..') / 'cpu_kernels_m5.json')
    hpc = read('hpc/live_slurm/incremental_kernels.json')
    rows = [(f"{a['molecules']} 个 LJ 客体分箱", f"{a['speedup_median']:.2f}×",
             f"{b['speedup_median']:.2f}×", '末态能量一致')
            for a, b in zip(m5['guest_cells'], hpc['guest_cells'])]
    rows.append(('32³ PME 延后刷新', f"{m5['deferred_pme']['speedup_median']:.2f}×",
                 f"{hpc['deferred_pme']['speedup_median']:.2f}×", '最大能量差 7.28×10⁻¹² K'))
    r.table(
        '表 10　相同构型与种子的内核微基准',
        ('更新路径', 'M5', 'Intel', '数值核对'),
        rows, (2.56, 1.08, 1.08, 2.20),
        note='每个内核五次配对，取时间比中位数。分箱与整体重建比较，PME 与每次接受都刷新的路径比较。两端 PME 的计数轨迹逐样本一致。',
    )
    r.heading('7.2 对二甲苯三点批跑')
    r.para(
        '对 0.02、0.50 和 40 Pa 各做 4,000 次预热和 12,000 次生产。每个种子交替完成三次独立冷启动及三个逸度共用一份场的批跑。'
        '首个状态点的建场成本计入批跑，切换逸度后的热准备单独记录。两种缓存策略的计数轨迹逐样本一致。'
    )
    r.table(
        '表 11　冷启动与批跑时间中位数',
        ('平台', '单点冷准备／s', '后续点热准备／s', '三点分开／s', '共享场批跑／s', '配对加速'),
        [
            ('M5', '2.31', '0.165', '31.29', '24.99', '1.189×'),
            ('Intel', '2.97', '0.159', '44.31', '38.71', '1.145×'),
        ],
        (.76, 1.24, 1.36, 1.17, 1.32, 1.07),
        note='加速为三个种子各自墙钟比的中位数。短预算固定，结果用于核算缓存的启动与批量共享成本。',
    )
    r.para(
        '按低、中、高逸度，单点冷启动至完成的墙钟中位数为 M5 的 4.19／6.96／20.59 s 和HPC的 5.53／9.16／29.55 s。'
        '批跑第一点构建 4 个原始场，后两点各读取 4 个。'
    )
    r.heading('7.3 本地 MCP 与HPC Slurm')
    r.para(
        '使用全硅 MFI 的 0.2、1、5 bar 三点等温线验证调度。M5 走本地 MCP，Intel 走实际 Slurm 数组。'
        'Slurm 61008 与 61009 完成 12 条链并提交 24 个样本块。重复提交返回原运行 ID。'
        '两端 24 个对应块的计数轨迹逐样本相同。从登记到最后一块提交分别为 4.53 s 和 197.27 s。'
        '集群时间包含排队、分配、计算和轮询。'
    )
    r.para(
        '采样、长程独立求和和缓存回归两端各通过 28 项。M5 另通过 5 项原生 API 回归。'
        '首次 Intel 内核作业的暂存包缺少微基准脚本，补齐并重新提交后完成。',
        'Report Note',
    )


def physical_cases(r):
    r.heading('8 真实吸附的 RASPA3 匹配对照', 1, True)
    r.para(
        '配对使用同一结构、LJ 混合规则、逐位点电荷、活度和可核对的有限 Ewald 设置。'
        '两程序读取相同的两份非空构型，另外两条链从空孔启动。RASPA3 采用其原生 CBMC 试探生成和接受。'
        '跨程序墙钟因而反映整套采样实现。'
    )
    r.table(
        '表 12　物理模型和初始化',
        ('体系', '温度 逸度和预装', '相互作用与长程设置'),
        [
            ('Ni_TPT_TATBA\n对二甲苯', '298.15 K，0.02／0.50／40 Pa\n预装 10／65／140 个分子',
             '4,416 框架位点与 12 位点带部分电荷刚体。12 Å 直接截断 LJ。Ewald α = 0.265058 Å⁻¹，k 上限各 13。'),
            ('全硅 TON\n正丁烷', '298.15 K，2／100 Pa\n预装 6／16 个分子',
             'TON 2×1×4 超胞和四位点 TraPPE-UA。LJ 8 Å 截断移势，谐和弯曲。框架及客体均为中性。'),
            ('IRMOF-1\nCO₂+CH₄', '300 K，0.50／1.50 bar\n预装 7 个 CO₂ 与 8 个 CH₄',
             '424 个框架位点，3 位点 CO₂ 与单中心 CH₄。12 Å 截断移势。Ewald α = 0.25 Å⁻¹，k 上限各 7。'),
        ], (1.37, 2.05, 3.5), left=(0, 1, 2),
    )
    r.figure(FIGURES / 'timing.png',
             '图 3　两台 CPU 上达到预设精度的墙钟。实心标记表示首达标，空心标记与向右箭头的位置为完成运行预算时的右截尾界限。TON 图示为完整再生长。')
    r.para(
        '每个平台内分别比较 OpenGCMC 与 RASPA3。末态坐标代回 OpenGCMC 后，'
        'IRMOF-1 最大总势能差为 1.46×10⁻⁴ K，TON 为 3.06×10⁻¹⁰ K。'
        '三档对二甲苯的最大差依次为 0.00152、0.04898、0.06143 K。能量以势能除以玻尔兹曼常数表示。'
        '对账固定的是同一构型，吸附分布及其混合由各自的四链诊断评估。'
    )


def xylene(r):
    r.heading('9 对二甲苯的低中高载量实验', 1, True)
    r.para(
        '该体系检验带电大型芳香客体在低占据和较高占据下的统计效率。'
        '低压 OpenGCMC 的初轮上限为 480,000 次尝试，R̂ = 1.01445。'
        '追加独立种子，以原有初始模型复测至 1,440,000 次，两台机器均在该检查点通过。'
        '低压 RASPA3 运行 8,000 生产周期。中、高压在外部轨迹启动前分别将生产上限固定为 1,600 和 800 周期。'
    )
    clock = []
    for case in REAL_CASES[3:]:
        m5 = timing_runs('m5', case)
        hpc = timing_runs('hpc', case)
        clock.append((case[0].removeprefix('对二甲苯 '),
                      '／'.join(seconds(a) for a in m5),
                      '／'.join(seconds(a) for a in hpc)))
    r.table(
        '表 13　对二甲苯首达标墙钟  单位 s',
        ('逸度', 'M5 OpenGCMC／RASPA3', 'Intel OpenGCMC／RASPA3'),
        clock, (1.06, 2.93, 2.93),
        note='＞ 后的数值为预算完成时的实测墙钟。中压两阶段计时和同候选单阶段计时归档于不同接受规则的原始记录。',
    )
    diagnose = (
        ('0.02 原生首轮', 'xylene_low', 'xylene_low-delayed'),
        ('0.02 原生复测', 'xylene_low_confirm', 'xylene_low-delayed-long'),
        ('0.02 RASPA3', 'xylene_low', 'xylene_low-raspa-cbmc'),
        ('0.50 两阶段', 'xylene_mid', 'xylene_mid-delayed'),
        ('0.50 同候选单阶段', 'xylene_mid', 'xylene_mid-reference-mh'),
        ('0.50 逐候选诊断', 'xylene_mid', 'xylene_mid-exact-cost'),
        ('0.50 RASPA3', 'xylene_mid', 'xylene_mid-raspa-cbmc'),
        ('40 两阶段', 'xylene_high', 'xylene_high-delayed'),
        ('40 RASPA3', 'xylene_high', 'xylene_high-raspa-cbmc'),
    )
    rows = []
    for label, group, name in diagnose:
        record = real_run('m5', group, name)
        step = checkpoint(record)
        o = step['observables']['count_0']
        means = record['start_group_final_means']
        unit = '周期' if 'raspa' in name else '次'
        rows.append((label, f'{step["budget"]:,} {unit}',
                     f'{o["mean"]:.3f}±{o["mcse"]:.3f}',
                     f'{o["rank_split_rhat"]:.5f}',
                     f'{means["empty"][0]:.3f}／{means["preload"][0]:.3f}'))
    r.table(
        '表 14　完整预算的对二甲苯诊断',
        ('逸度 Pa 与路径', '每链生产预算', '均值 ± MCSE', 'R̂', '空孔／预装组均值'),
        rows, (1.65, 1.30, 1.41, .76, 1.80),
        note='载量单位为分子每超胞。两台机器按相同种子得到对应诊断。中压逐候选仅用于求值成本诊断。',
    )
    d = [real_run(p, 'xylene_mid', 'xylene_mid-reference-mh') for p in ('m5', 'hpc')]
    r.para(
        f'中压同候选单阶段在相同 480,000 次尝试上限的墙钟为 {seconds(d[0])}／{seconds(d[1])} s。'
        '两阶段和单阶段的链都显示较慢的统计混合。高压两程序的 R̂ 分别为 1.761 和 2.644，'
        '两组终点载量均值仅反映各自保存的有限轨迹，中高压的均衡载量和达标速度仍未确定。'
    )


def flexible(r):
    r.heading('10 柔性正丁烷的构象与载量', 1, True)
    r.para(
        '固定 114° 键角的先导实验中，RASPA3 保存构型的实际角度偏离约束。'
        '正式配对改用两程序均实际计算的谐和弯曲势，中心 114°，力常数 62,500 K/rad²。'
        '固定键长 1.54 Å，保留 TraPPE-UA 扭转项，OpenGCMC 的内配分函数使用相同角度积分，80°–148° 覆盖八个以上的热宽。'
        '两档匹配的 M5 RASPA3 轨迹共检查 12,400 帧，键长均在 1.54±0.005 Å 内。'
    )
    r.para(
        '将两个连续键平面法向夹角的余弦不大于 −0.5 定义为反式。每个检查点用所有保存帧的反式分子数之和除以总分子数之和，'
        '对比值进行一阶误差传播。理想气体的解析载量和反式比例为 2.3881 分子每箱和 66.47%。'
        'OpenGCMC 给出 2.3921 和 66.49%，RASPA3 给出 2.3863 和 66.08%。'
    )
    controls_ton = (
        ('2 Pa', '完整再生长', 'ton_butane_dilute', 'ton_butane_dilute-delayed'),
        ('2 Pa', '后缀再生长', 'ton_butane_dilute', 'ton_butane_dilute_suffix-delayed'),
        ('2 Pa', 'RASPA3 CBMC', 'ton_butane_dilute', 'ton_butane_dilute-raspa-cbmc'),
        ('100 Pa', '完整再生长', 'ton_butane', 'ton_butane_harmonic-delayed'),
        ('100 Pa', '后缀再生长', 'ton_butane', 'ton_butane_harmonic_suffix-delayed'),
        ('100 Pa', 'RASPA3 CBMC', 'ton_butane', 'ton_butane_harmonic-raspa-cbmc'),
    )
    wall = []
    results = []
    for pressure, mode, group, name in controls_ton:
        m5, hpc = (real_run(p, group, name) for p in ('m5', 'hpc'))
        wall.append((pressure, mode, seconds(m5), seconds(hpc)))
        o = checkpoint(m5)['observables']
        a, b = o['count_0'], o['trans_fraction']
        results.append((pressure, mode, f'{a["mean"]:.3f}±{a["mcse"]:.3f}',
                        f'{b["mean"]:.4f}±{b["mcse"]:.4f}',
                        f'{a["rank_split_rhat"]:.3f}／{b["rank_split_rhat"]:.3f}'))
    r.table(
        '表 15　载量与构象联合通过时的墙钟',
        ('逸度', '路径', 'M5／s', 'Intel／s'),
        wall, (.95, 2.29, 1.84, 1.84),
        note='完整再生长与 RASPA3 对照，2 Pa 的首达标加速为 M5 13.89 倍、Intel 9.36 倍。',
    )
    r.table(
        '表 16　完整预算下的估计值与收敛诊断',
        ('逸度', '路径', '载量 ± MCSE', '反式比例 ± MCSE', 'R̂ 载量／反式'),
        results, (.74, 1.42, 1.33, 1.75, 1.68),
        note='2 Pa 的原生与 RASPA3 上限分别为 400,000 次和 8,000 周期。100 Pa 为 840,000 次和 21,000 周期。',
    )
    r.para(
        '在 2 Pa，后缀更新把该类移动的接受率从 0.79% 提到 5.15%，反式比例 ESS 从 1,062 降至 757。'
        '在 100 Pa，接受率由 0.096% 升至 4.08%，反式比例 ESS 由 30 升至 137，载量 R̂ 仍为 1.124。'
        '局部构象移动改善与粒子占据收敛可以发生在不同的时间尺度。',
    )


def mixture(r):
    r.heading('11 二元竞争吸附和分数分子对照', 1, True)
    r.para(
        'IRMOF-1 的性能基准包含三个观测量，两组分载量和 CO₂ 至少 3 个且 CH₄ 至少 4 个的联合事件。'
        'OpenGCMC 使用原生两阶段和增量客体索引，RASPA3 使用匹配的普通 CBMC。'
        'M5 的首达标时间为 1.12／10.03 s，Intel 为 3.35／20.80 s。'
        '实际首达标预算为每链 60,000 次尝试和 1,000 周期，后续记录均继续满足门槛。'
    )
    rows = []
    for source, label in (
        ('m5/irmof_mixture-delayed.json', 'OpenGCMC'),
        ('m5/irmof_mixture-raspa-cbmc.json', 'RASPA3'),
    ):
        info = read(source)
        obs = checkpoint(info)['observables']
        first = checkpoint(info, first=True)['observables']
        for name, key in (('CO₂ 载量', 'count_0'), ('CH₄ 载量', 'count_1'), ('联合事件', 'joint_probability')):
            a, b = obs[key], first[key]
            rows.append((label, name, f'{a["mean"]:.4f} ± {a["mcse"]:.4f}',
                         f'{a["rank_split_rhat"]:.5f}', f'{b["mean"]:.4f} ± {b["mcse"]:.4f}'))
    r.table(
        '表 17　IRMOF-1 的完整轨迹及首达标估计',
        ('程序', '观测', '完整均值 ± MCSE', '完整 R̂', '首达标均值 ± MCSE'),
        rows, (.99, 1.12, 1.91, .87, 2.03),
        note='载量按分子每超胞计。完整轨迹为 240,000 次尝试和 12,000 周期，首达标和终点估计分别列出。',
    )
    r.para(
        '完整轨迹的经验联合直方图总变差为 0.0555，联合事件概率为 0.3162 和 0.3074。'
        '最早检查点的联合概率较完整轨迹波动更大，双平台首达标衡量的是预设门槛的通过时间。'
        '完整轨迹中，OpenGCMC 两种起跑的 CO₂ 均值为 2.683／2.637，CH₄ 为 4.258／4.172。'
        'RASPA3 中相应为 2.665／2.664 和 4.228／4.226。'
    )
    r.heading('11.1 CB CFCMC 扩展系综实验')
    m5 = read('m5/irmof_mixture-raspa-cfcmc.json')
    hpc = read('hpc/cfcmc/irmof_mixture-raspa-cfcmc.json')
    elapsed = [max(c['elapsed_at_prefix_s'][-1] for c in run['chains']) for run in (m5, hpc)]
    r.para(
        '两个平台还各完成四条 CB/CFCMC 链，每链预热 15,000 周期、计入生产 15,000 周期、保存 1,500 个整分子计数。'
        f'最后一个计时检查点为 M5 {elapsed[0]:.2f} s 和 Intel {elapsed[1]:.2f} s。'
        '归档计数来自扩展系综。对同一个巨正则联合事件重权需同步的逐帧耦合参数 λ。'
        '因此与 OpenGCMC 的首达标比较以匹配的普通 CBMC 为基线。'
    )
    r.para(
        '结果文件还保留每条链的实际起跑计数、种子和各检查点墙钟，可用统计表按起跑分组重算。'
        'CFCMC 的原始轨迹、偏置和摘要与普通 CBMC 分别归档。',
        'Report Note',
    )


def blind_holdout(r):
    r.heading('12 UiO-66 留出实验与失速验证', 1, True)
    r.para(
        '开发体系为带电 IRMOF-1。冻结记录中，两阶段的热缓存准备最长用时 0.0207 s，精算触发率为 67.22%。'
        '规则于 2026 年 11 月 23 日 10:21 UTC 冻结。UiO-66 留出条件是 350 K、CO₂ 逸度 0.60 bar、'
        '32³ 参考场、载量 MCSE 目标 0.25 分子每超胞。用留出体系的 20,000 次先导生产尝试代入同一规则。'
        '确认链使用另一组种子。'
    )
    eps = sym('ε', 'j')
    term = sup(bracket(frac(sub(mi('MCSE', True), mi('j')), eps), '[', ']'), mn(2))
    r.math(accent(mi('T'), '̂') + EQ + mi('H') + PLUS + sym('T', 'pilot') + mi('max', True)
           + bracket(mn(1) + mo(',') + under(mi('max', True), mi('j')) + term))
    r.para(
        'H 是固定成本，先导的最长生产墙钟用于外推。两阶段的预测时间小于逐候选精算的 85%，且两阶段先导的最大 R̂ ≤ 1.10 时选择两阶段。'
        'UiO-66 两阶段先导的最长准备时间为 1.94／2.60 s（M5／HPC），精算触发率为 53.78%，R̂ = 1.0770，两端均选两阶段。'
        '240,000 次确认尝试后，MCSE 为 0.376、R̂ 为 1.01263。随后用相同确认种子和物理设置重跑至更长预算。'
    )
    r.table(
        '表 18　冻结二选一的预测与确认  单位 s',
        ('平台', '预测 逐候选／两阶段', '确认 逐候选／两阶段', '两阶段实测／预测'),
        [
            ('M5', '140.18／20.80', '243.62／74.38', '3.58×'),
            ('Intel', '138.11／15.44', '157.06／94.34', '6.11×'),
        ],
        (.81, 2.15, 2.13, 1.83),
        note='逐候选精算首达标为 120,000 次尝试，两阶段为 960,000 次，后两个检查点继续通过。二选一的快慢方向在两端吻合，短链预测均低估了两阶段时间。',
    )
    r.heading('12.1 同候选的追加实验')
    r.para(
        '冻结验证结束后，用预热的同一 32³ 框架场、相同候选生成规则和相同确认种子比较单、两阶段。'
        '每组保留到 1,920,000 次尝试。单阶段在 240,000 次首达标，两阶段在 960,000 次首达标。'
    )
    r.table(
        '表 19　同一候选生成规则下的 UiO-66',
        ('平台', '单阶段／s', '两阶段／s', '同预算生产时间比 单／两'),
        [('M5', '18.19', '49.28', '1.44'), ('Intel', '38.09', '98.05', '1.43')],
        (.81, 1.31, 1.35, 3.45),
        note='完整轨迹的载量相关因子为单阶段 9.39、两阶段 19.33。2.06 倍的相关性代价超过两个平台的成本边界。',
    )
    r.para(
        '原冻结规则只比较逐候选与两阶段。加入共享参考候选的单阶段后，实测优先路径为单阶段。'
        '该结果说明，省掉候选构象的逐次精算和省掉部分最终校正是两项不同的成本收益。'
    )


def conclusions(r):
    r.heading('13 结论与复算入口', 1, True)
    r.para(
        '双孔真值和 NIST 结果支持参考势校正对规定目标分布的保持。'
        '双孔里，粒子数边际分布可以与真值完全一致，单孔占据与构象分布仍能显著偏离。'
        'NIST 的独立支持范围为粒子计数及其联合分布。'
    )
    r.para(
        '同候选下，两阶段的总体价值由相关性与成本的乘积决定。'
        'IRMOF-1 在 32³ 时达到整体统计门槛更快，在 CH₄ 上的 ESS/s 仍低于单阶段。'
        'UiO-66 的同候选单阶段在两个 CPU 平台都更快。'
        '本次冻结判别的迁移验证覆盖一个留出体系。短先导给出的时间预测对该体系明显偏短。'
    )
    r.para(
        'RASPA3 匹配实验在低载 TON 和 IRMOF-1 上给出双方都达标的同机比较。'
        '对二甲苯的低压确认、以及中高载和 TON 100 Pa 的右截尾值都保留在统计表中。'
        '这些工况的速度结论采用相同观测的首达标墙钟。跨平台复用种子的轨迹一致性用于数值核对，'
        '不增加独立链数。'
    )
    r.heading('13.1 复算原始检查点')
    r.para(
        '报告和数据存放在基准目录。脚本用 JSON 指向各条链的 NPZ 样本，复算均值、MCSE、R̂、ESS、'
        '空孔和预装起跑均值及首达标标志。全部 516 项检查点观测已复算，与归档 CSV 逐行一致。'
        '计算时使用包含 NumPy、SciPy 和 poremc 的 Python 环境。'
    )
    r.para(
        'python export_checkpoints.py . checks_reproduced.csv\n'
        'python run_cpu_suite.py controls --output results/controls \\\n'
        '  --cache results/fields\n'
        'python run_cpu_suite.py irmof_mixture --output results/paired \\\n'
        '  --cache results/fields --raspa3 /path/to/raspa3',
        'Code Sample',
    )
    r.heading('13.2 原生与集群复算')
    r.para(
        '从与归档相同的 206122c 源码构建。M5 和 Linux 可用相同的 portable 与 no-openmp 选项。'
        'HPC的 hpc_cpu.sbatch 将模型和构型参数带到计算节点，写出机器规格和编译记录。'
        'benchmark_blind_policy.py 读取 blind_protocol.json 执行冻结规则，'
        'run_cpu_suite.py 的 holdout_same_candidates 对应后续同候选配对。'
    )
    r.para(
        'python native/build.py --portable --no-openmp\n'
        'python -m pytest -q tests/test_delayed_physics.py \\\n'
        '  tests/test_upgrade_sampling.py tests/test_native_longrange_physics.py',
        'Code Sample',
    )


def appendix_budgets(r):
    r.heading('附录 A 每链预算和数据索引', 1, True)
    r.para(
        '常规四链基础种子依次为 440713、440731、440747 和 440753，前两条空孔，后两条预装。'
        '低压对二甲苯独立复测使用基础种子加 150000，UiO-66 先导加 731000，确认和同候选复测加 928000。'
        '每个 case 的两份非空构型来自对应的 starts 文件。'
    )
    keys = (
        ('NIST 两组', '20,000', '20k／60k／120k／240k', '—'),
        ('接受规则对照和 IRMOF-1', '20,000', '20k／60k／120k／240k', '1,000／1k 3k 6k 12k'),
        ('对二甲苯低压首轮', '30,000', '40k／120k／240k／480k', '1,000／800 2,400 4,800 8,000'),
        ('对二甲苯低压确认', '30,000', '120k／360k／720k／1,440k', '—'),
        ('对二甲苯中压', '30,000', '40k／120k／240k／480k', '1,000／800 1,600'),
        ('对二甲苯高压', '30,000', '40k／120k／240k／480k', '1,000／400 800'),
        ('TON 2 Pa', '20,000', '60k／180k／400k', '1,000／800 1,600 4,000 8,000'),
        ('TON 100 Pa', '35,000', '100k／240k／480k／840k', '2,500／2k 6k 12k 21k'),
        ('UiO-66 确认', '20,000', '20k／60k／120k／240k', '—'),
        ('UiO-66 延长与复测', '20,000', '240k／480k／960k／1,440k／1,920k', '—'),
    )
    r.table(
        '表 A1　记录的生产检查点',
        ('工况', '原生预热 次', '原生生产检查点 次', 'RASPA3 预热／生产 周期'),
        keys, (1.56, .99, 2.12, 2.25),
        note='k 表示千。中压逐候选诊断预热 10,000 次，生产 8,000／20,000／40,000 次。原生每 200 次尝试保存一次计数，RASPA3 按运行摘要所列周期抽取。',
    )
    r.table(
        '表 A2　实验目录中的原始材料',
        ('内容', '路径'),
        [
            ('结构 力场 活度 初始构型', 'inputs/ 中的模型 JSON、starts JSON 和 TON.cif'),
            ('全量计数和构象统计', 'm5/ 与 hpc/ 下各组 JSON、配套 NPZ、checkpoints.csv'),
            ('双孔全转移矩阵与标准参考', 'two_pore.json、hpc/two_pore/、inputs/ 中的 *_nist.npz'),
            ('精算 配对和启动控制', 'm5/controls_warm/、hpc/controls/、两端 cache_timing.json'),
            ('RASPA3 力场及末态能量', '每个 *-raspa-cbmc/ 的输入与输出、audit_raspa3.py'),
            ('冻结规则与留出证据', 'blind_protocol.json、m5/holdout/、hpc/blind/、holdout_same_candidates/'),
            ('编译与调度', 'hpc/slurm_accounting.txt、hpc/live_slurm/、两端 hardware.txt 与测试日志'),
        ],
        (2.05, 4.87), left=(0, 1),
        note='HPC真实基准主数组为 61001 与 61003。UiO-66 延长、TON 低载、附加对照及数值回归的作业号记录于 Slurm 文件。',
    )


def appendix_references(r):
    r.heading('附录 B 扩展科研场景与参考资料', 1, True)
    r.heading('B.1 Apple M5 扩展数据')
    r.para(
        '扩展场景在 Apple M5 上执行。数值回归分别采用独立解析解、重算能量或多链统计。'
        'IRMOF-1、对二甲苯和 TON 的 M5 补充工况详见配套的《科研场景 Benchmark》，包括固定预算与化学势交换数据。'
    )
    r.table(
        '表 B1　扩展场景的实测端点与适用范围',
        ('场景', '已获得的结果', '统计或模型范围'),
        [
            ('MFI CO₂／CH₄', '11 点化学势交换，143.46 s、97 次端点往返',
             '11 点均通过载量与对数选择性门槛'),
            ('NaCl／SPC/E 恒电势', '128 构型中缓存电荷与稠密独立求解最大差 1.09×10⁻¹⁴ e',
             '长链电荷 R̂ = 1.89，平衡电容仍受慢混合限制'),
            ('显式水质子化和氧化还原', '固定构型七态求能的缓存加速为 8.76× 和 8.81×',
             '延长后最大相邻能隙 R̂ 为 1.82 和 1.67'),
            ('铁氰化物／亚铁氰化物', '细网格计时安培对 Cottrell 解最大相对差 0.027%',
             '平面电极扩散算例，峰电流比 1.41404'),
        ],
        (1.71, 2.63, 2.58), left=(0, 1, 2),
        note='扩展原始数据见 benchmarks/cpu/scientific 及其相邻的内核与调度记录。',
    )
    r.heading('B.2 数据和方法来源')
    r.reference('NIST. TMMC Reference Adsorption Isotherms for CO₂ in Metal–Organic Frameworks.',
                '纯 CO₂ 与 UiO-66 参考数据', NIST_PURE)
    r.reference('NIST. TMMC Reference Adsorption Isotherms for N₂/CO₂ Mixtures in Metal–Organic Frameworks.',
                'N₂/CO₂ 参考数据', NIST_MIX)
    r.reference('International Zeolite Association Structure Commission. TON framework CIF.',
                'TON 晶体结构', 'https://www.iza-structure.org/IZA-SC/cif/TON.cif')
    r.reference('RASPA3. Simulation code and example inputs.', 'RASPA3 源码及示例',
                'https://github.com/iRASPA/RASPA3')
    r.reference('Martin MG, Siepmann JI. Transferable Potentials for Phase Equilibria. 1. United-Atom Description of n-Alkanes. J Phys Chem B, 1998.',
                'TraPPE-UA', 'https://doi.org/10.1021/jp972543%2B')
    r.reference('Rappé AK et al. UFF, a Full Periodic Table Force Field for Molecular Mechanics and Molecular Dynamics Simulations. J Am Chem Soc, 1992.',
                'UFF', 'https://doi.org/10.1021/ja00051a040')
    r.reference('Siepmann JI, Frenkel D. Configurational Bias Monte Carlo: A New Sampling Scheme for Flexible Chains. Mol Phys, 1992.',
                'CBMC', 'https://doi.org/10.1080/00268979200100061')
    r.reference('Christen JA, Fox C. Markov Chain Monte Carlo Using an Approximation. J Comput Graph Stat, 2005.',
                '延迟接受', 'https://doi.org/10.1198/106186005X76983')
    r.reference('Peskun PH. Optimum Monte-Carlo Sampling Using Markov Chains. Biometrika, 1973.',
                'Peskun 顺序', 'https://doi.org/10.1093/biomet/60.3.607')
    r.reference('Vehtari A et al. Rank-Normalization, Folding, and Localization: An Improved R̂ for Assessing Convergence of MCMC. Bayesian Anal, 2021.',
                '分链诊断', 'https://doi.org/10.1214/20-BA1221')


def build(path):
    r = Report()
    for section in (
        formal_summary, design, detailed_balance, ewald_boundary, double_pore, nist,
        controls, kernel_cache_dispatch, physical_cases, xylene, flexible, mixture,
        blind_holdout, conclusions, appendix_budgets, appendix_references,
    ):
        section(r)
    r.doc.save(path)


if __name__ == '__main__':
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'cpu_experiment_report.docx'
    build(path)
    print(path)
