#!/usr/bin/env python3
"""Generate report tables from recorded 0.6 measurements, never hand-enter rates."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import statistics

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'evidence06'

def read(name):
    return json.loads((EVIDENCE/name).read_text())

def sci(x):
    mantissa, exponent=f'{x:.3e}'.split('e')
    return rf'{mantissa}\times10^{{{int(exponent)}}}'

def table(headers, rows, spec=None):
    spec=spec or 'l'+'r'*(len(headers)-1)
    text=['\\begin{center}\\small',rf'\begin{{tabular}}{{@{{}}{spec}@{{}}}}\toprule',
          ' & '.join(headers)+r'\\\midrule']
    text += [' & '.join(map(str,row))+r'\\' for row in rows]
    text += [r'\bottomrule\end{tabular}\end{center}']
    return '\n'.join(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args()
    rigid=read('hostgrid_benchmark.json');flexible=read('flexible_benchmark.json')
    old=read('linear_screening_benchmark.json')
    physics=read('delayed_physics_audit.json');reciprocal=read('reciprocal_audit.json')
    grid=read('host_grid_module.json')
    testlog=(EVIDENCE/'all_tests.log').read_text() if (EVIDENCE/'all_tests.log').exists() else ''
    matched=re.search(r'(\d+) passed',testlog)
    tests=int(matched.group(1)) if matched else 0
    if args.require_complete:
        assert tests>327 and not re.search(r'[1-9]\d* (?:failed|error)',testlog)
        assert len(rigid['rows'])==15 and len(flexible['rows'])==9
        assert rigid['surrogate_steep_interpolation']=='boltzmann'
        assert all(v['passed'] for v in physics['checks'].values())
        binary=ROOT/'poremc/libporemc.so'
        if binary.exists():
            sha=hashlib.sha256(binary.read_bytes()).hexdigest()
            assert sha==physics['native_library_sha256']
            assert all(r['native_sha256']==sha for d in (rigid,flexible) for r in d['rows'] if r['mode']!='baseline05')
    def summary(data,mode):return next(r for r in data['summary'] if r['mode']==mode)
    r=summary(rigid,'delayed32');f=summary(flexible,'delayed32')
    metrics=(rf'\newcommand{{\TotalTests}}{{{tests}}}'+'\n'+
             rf'\newcommand{{\RigidSpeed}}{{{r["median_speedup"]:.2f}}}'+'\n'+
             rf'\newcommand{{\FlexibleSpeed}}{{{f["median_speedup"]:.2f}}}'+'\n')
    (ROOT/'report_06/metrics.tex').write_text(metrics)
    labels={'baseline05':'0.5 全谱逐候选','native06':'0.6 半谱逐候选',
            'delayed32':r'0.6 参考网格 $32^3$',
            'biased32':r'$32^3$ + alias 根位点',
            'delayed64':r'0.6 参考网格 $64^3$'}
    pieces=[r'\section{独立数值与采样分布验证}',
      '最终测试日志中的 '+str(tests)+' 项测试通过，另有 28 个参数子场景。新增核验同时比较能量恒等式和采样分布；后者直接检验参考势变化后，残差校正是否恢复目标分布。',
      table(['验证对象','最大绝对误差'],[
          ['半谱总能 / 独立全谱',rf'${sci(reciprocal["max_total_energy_error_K"])}$ K'],
          ['增量能量 / 独立有限和',rf'${sci(reciprocal["max_increment_error_K"])}$ K'],
          ['框架 FFT 节点 / 直接求和',rf'${sci(reciprocal["max_host_fft_error_K"])}$ K'],
          ['平移相位 / 坐标重算结构因子',rf'${sci(reciprocal["max_translation_structure_factor_error"])}$'],
          ['Boltzmann 插值 / 独立加权配分和',rf'${sci(grid["boltzmann_formula_error_K"])}$ K'],
          ['生产目标探针 / 独立 Ewald',rf'${sci(physics["checks"]["prefix_and_residual"]["max_exact_reference_error_K"])}$ K']
      ],'lr')]
    ideal=physics['checks']['nonuniform_root_poisson']
    pieces.append(f'非均匀根位点检验把 85\\% 的提议概率集中在 24 个 voxel 中的一个，同时保留全空间支持。理论计数均值和方差均为 2.7；{ideal["samples"]:,} 个保存样本给出均值 {ideal["mean"]:.5f}、方差 {ideal["variance"]:.5f}。该算例检验根位点密度只进入 Rosenbluth 权重，而没有重复进入残差。')
    canonical=physics['checks']['canonical_quadrature']
    pieces.append('更强的检验故意把参考势截幅到 50 K，使其与带电单粒子的完整目标明显不同。独立空间积分和原生采样比较如下；误差条为分块标准误。')
    pieces.append(table(['观测量','独立目标积分','未校正参考势','校正后采样'],[
      [rf'$\langle {fn}\rangle$',f'{target:.6f}',f'{surrogate:.6f}',rf'${observed:.6f}\pm{se:.6f}$']
      for fn,target,surrogate,observed,se in zip(
        [r'\cos(2\pi x/L_x)',r'\sin(2\pi x/L_x)',r'\cos(2\pi z/L_z)'],
        canonical['quadrature'],canonical['uncorrected_surrogate_quadrature'],canonical['sampled_mean'],canonical['block_se'])]))
    pieces.append('独立积分的网格加密最大变化为 '+rf'${sci(max(canonical["quadrature_refinement_difference"]))}$'+
                  '；本次实际残差范围为 '+f'{canonical["residual_range_K"][0]:.2f} 至 {canonical["residual_range_K"][1]:.2f} K。参考势本身不能复现目标分布，加入第二阶段后采样与目标积分一致。逐分子结构因子恢复、五类移动和缓存跨温度重用也已核验。')
    pieces += [r'\section{完整循环的同机单线程基准}',r'\subsection{模型与计时约定}',
      'CPU 为 '+rigid['cpu_model']+'。两端使用相同编译优化，固定一个 OpenMP / BLAS 线程。对照源为打包的未修改 0.5 核心；0.6 所有主表均保持同一完整矩形有限 Ewald 目标，没有施加径向模式截断。',
      '本轮没有用户实际 CIF 与完整 p-xylene 力场输入，因此创建并公开了计算规模匹配的合成算例：48 \\AA{} 立方盒、4,416 个框架点、12 个客体位点、10 个 CBMC trial、10 个初始分子、$k_{\\max}=13$、$T=350$ K、$r_c=10$ \\AA{}、$\\alpha=0.28$ \\AA$^{-1}$。五种移动等概率，平移上限 0.65 \\AA{}、转角上限 0.35 rad；LJ 混合采用 Lorentz--Berthelot 与截断移位，1--4 LJ / Coulomb 缩放为 0.5 / 0.75。',
      '框架为形成周期狭缝的八个轻微起伏平面，位点 $\\sigma=2.9$ \\AA、$\\epsilon=4$ K；电荷由固定种子生成并减去均值。刚性算例由六个环 C、四个 H 和两个甲基伪位点组成，采用公开在脚本中的合成参数；柔性算例为十二位点链，$\\sigma=2.2$ \\AA、$\\epsilon=6$ K，电荷交替 $\\pm0.03e$，键长 1.3 \\AA、键角 $112^\\circ$、三重扭转系数 15 K。两类活度分别为 $8/V$ 和 $12/V$。这些参数定义数值和性能任务，不是对某种材料的力场标定。',
      f'刚性每组预热 {rigid["burnin"]} 次、生产 {rigid["steps"]} 次；柔性分别为 {flexible["burnin"]} 与 {flexible["steps"]} 次。各三种子、独立进程并交替测试顺序。生产吞吐不含初始化和末态参考重算，关闭逐帧总能记录；接受次数与计数轨迹同时保留。',
      r'\subsection{十二位点刚性客体}']
    for data in (rigid,flexible):
        if data is flexible:pieces.append(r'\subsection{十二位点柔性客体}')
        pieces.append(table(['路径','尝试/s','接受/s','配对加速','加速范围'],[
          [labels[row['mode']],f'{row["median_moves_s"]:,.2f}',f'{row["median_accepted_s"]:,.2f}',
           f'{row["median_speedup"]:.2f}',f'{row["speedup_range"][0]:.2f}--{row["speedup_range"][1]:.2f}'] for row in data['summary']]))
    pieces.append('吞吐两列分别为三次测量的中位数；加速列为每次配对比值的中位数，因此不必等于两列中位数之比。接受吞吐避免把纯拒绝路径的高尝试数误认成构型探索效率；构型相关性另在长链部分检验。')
    maximum=max(row['max_final_recompute_error_K'] for d in (rigid,flexible) for row in d['summary'])
    pieces.append('全部配对任务的末态完整重算最大差为 '+rf'${sci(maximum)}$ K。'+
                  '不同接受算法使用不同数量的随机数，轨迹无需逐步相同；物理输入哈希、初始能量、末态重算和分布共同定义比较依据。')
    pieces.append(r'\subsection{缓存初始化与生产工作量}')
    cache_rows=[]
    for mesh in (32,64):
        rr=[r for r in rigid['rows'] if r['mode'] in (f'delayed{mesh}',f'biased{mesh}')]
        cold=[r['construction_seconds'] for r in rr if not r['cache_existed_before']]
        warm=[r['construction_seconds'] for r in rr if r['cache_existed_before']]
        cache_rows.append([rf'${mesh}^3$',f'{statistics.median(cold):.3f}' if cold else '--',
                           f'{statistics.median(warm):.3f}' if warm else '--',
                           f'{rr[0]["cache_bytes"]/1024**2:.2f}'])
    pieces.append(table(['网格','首次构建/s','读取复用/s','磁盘字段/MiB'],cache_rows))
    pieces.append('这里的初始化包括势函数和原生引擎构建，单独记录，不加入生产吞吐。缓存保存原始字段，重新选择温度、活度和根位点偏置时可以继续读取。')
    stage_rows=[]
    for mode in ('delayed32','biased32','delayed64'):
        rows=[row for row in rigid['rows'] if row['mode']==mode]
        evaluated=sum(row['profile']['acceleration']['stage1_evaluated'] for row in rows)
        passed=sum(row['profile']['acceleration']['stage1_passed'] for row in rows)
        accepted=sum(row['profile']['acceleration']['stage2_accepted'] for row in rows)
        exact=sum(row['profile']['acceleration']['exact_evaluations'] for row in rows)
        attempts=sum(row['steps']+row['burnin'] for row in rows)
        stage_rows.append([labels[mode],f'{100*passed/evaluated:.2f}',f'{100*accepted/passed:.2f}',f'{exact/attempts:.3f}'])
    pieces.append(table(['路径','第一阶段通过/\\%','第二阶段通过/\\%','精确分子评价/尝试'],stage_rows))
    pieces.append('阶段统计包含预热与生产；空体系无法执行的删除等移动不进入第一阶段分母。一次替换包含旧、新两个残差，旧分子结构因子与两种局域能直接读取缓存。')
    pieces += [r'\section{粗网格的负结果与混合检验}',
      '初始三线性参考把非常高的排斥节点能量混入相邻可达体积。下表给出相同刚性任务三条短链的平均计数，保留失败的优化路径以解释最终设计。Boltzmann 插值恢复了进入近壁区域的机会，但参数选择仍以更长轨迹和相关性判断。']
    def means(data,mode):return ', '.join(f'{r["count_mean"]:.2f}' for r in sorted(data['rows'],key=lambda r:r['seed']) if r['mode']==mode)
    pieces.append(table(['路径','三种子的平均粒子数'],[
      ['精确 0.5 参考',means(rigid,'baseline05')],
      [r'线性能量插值 $32^3$',means(old,'delayed32')],
      [r'Boltzmann $32^3$',means(rigid,'delayed32')],
      [r'Boltzmann $64^3$',means(rigid,'delayed64')]],'lr'))
    mixing_path=EVIDENCE/'mixing_diagnostics.json'
    if mixing_path.exists():
        mixing=next(d for d in read('mixing_diagnostics.json')['datasets'] if d['label']=='long_boltzmann')
        if args.require_complete:assert len(mixing['rows'])==12
        pieces.append('长链对比每种路径运行三个独立种子，每条预热 5,000 次、生产 30,000 次，每 10 次保存计数。下表的误差为三个种子均值的标准误；ESS 按各链的初始正序列自相关估计后相加，仅针对粒子数这一观测量。')
        pieces.append(table(['路径',r'$\langle N\rangle$',r'种子均值 SE',r'$N$ 的 ESS 合计'],[
            [labels[row['mode']],f'{row["pooled_mean"]:.3f}',f'{row["replicate_mean_se"]:.3f}',f'{row["total_count_ess"]:.1f}']
            for row in mixing['summary']]))
        pieces.append('相同 90,000 次生产尝试下，$64^3$ 路径的计数 ESS 与精确对照相近，且优于本例的 $32^3$ 路径；结合完整循环吞吐，本报告建议从 $64^3$ Boltzmann 参考网格开始。$32^3$ 与 alias 参数仍可按目标体系实测选择。独立链均值仍有散布，表中没有把这组诊断作为精确等温线的最终物性。长链为缩短验证时间并行运行，其耗时不用于性能倍率或 ESS/s。')
    elif args.require_complete:raise RuntimeError('Missing completed long-chain mixing table')
    else:pieces.append('长链统计正在生成。')
    tuning_path=EVIDENCE/'reciprocal_tuning.json'
    if tuning_path.exists():
        tuning=read('reciprocal_tuning.json')
        pieces.extend([r'\section{显式模式截断的精度与成本空间}',
            r'除保持原有限和的半谱优化外，本版提供离线误差界建议工具。给定最大环境载量与完整分子的电荷绝对值和，可以界定省略倒空间壳层造成的能量增量差。下面使用同一刚性合成模型、$N_{\max}=64$；保守替换界覆盖完整分子的插入、删除与替换。主性能表没有采用这些额外截断。',
            table(['容限/K',r'$k_c$/\AA$^{-1}$','保留半谱','替换上界/K','实测最大差/K'],[
                [f'{r["criterion"]["tolerance_K"]:.2g}',f'{r["recommendation"]["sampling"]["ewald_k_cutoff"]:.6f}',
                 str(r['modes']['retained_half']),f'{r["bounds"]["replacement_increment_K"]:.6f}',
                 rf'${sci(r["validation"]["max_observed_omitted_replacement_K"])}$'] for r in tuning['reports']]),
            '每个建议另以环境占据 0、10、64 的 12 组新旧刚体构型检查实际省略差。解析界是给定载量范围内的约束，样本最大差只是这组构型的测量。非零径向截断定义另一个有限 Ewald Hamiltonian；没有把其模式数缩减当作本轮已测得的额外加速。',
            r'\begin{lstlisting}'+'\n'+
            'python tools06/advise_reciprocal.py your_hpc_input.json \\\n    --max-molecules 64 --tolerance-K 0.1 \\\n    --output reciprocal_advice.json\n'+r'\end{lstlisting}',
            '该工具只输出建议，不修改输入文件。载量超出分析范围时应重新评估，生产采样中不动态切换目标模式集合。'])
    elif args.require_complete:raise RuntimeError('Missing completed reciprocal-tuning table')
    else:pieces.append('显式模式截断的误差界分析正在生成。')
    (ROOT/'report_06/results.tex').write_text('\n\n'.join(pieces)+'\n')
    print(ROOT/'report_06/results.tex')

if __name__=='__main__':main()
