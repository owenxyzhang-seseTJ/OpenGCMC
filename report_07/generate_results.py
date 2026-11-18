"""Render 0.7 report tables from the recorded 0.7 data files."""
from pathlib import Path
import argparse
import json
import re

REPORT=Path(__file__).resolve().parent
DATA=REPORT.parent/'evidence07'

def read(name):
    return json.loads((DATA/(name+'.json')).read_text())

def write(name, content):
    (REPORT/(name+'.tex')).write_text(content+'\n',encoding='utf-8')

def scientific(x):
    mantissa,exp=f'{x:.2e}'.split('e')
    return rf'{mantissa}\times 10^{{{int(exp)}}}'

def table(headers,rows,formatting=None):
    align=formatting or 'l'+'r'*(len(headers)-1)
    return '\n'.join([r'\begin{center}\small\begin{tabular}{@{}'+align+r'@{}}\toprule',
                      ' & '.join(headers)+r'\\\midrule',*[' & '.join(row)+r'\\' for row in rows],
                      r'\bottomrule\end{tabular}\end{center}'])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft',action='store_true',help='Allow unfinished medium/dense benchmark records.')
    args=parser.parse_args()
    hpc=read('hpc_user_reported')['low']
    raspa=hpc['raspa2_cycles_per_second']*hpc['attempts_per_cycle']
    hpc_rows=[]
    for name,value in [('0.5 exact',hpc['baseline05_exact']),('0.6 exact 半谱',hpc['v06_exact']),
                       ('0.6 delayed，$32^3$',hpc['v06_delayed32']),('0.6 delayed，$64^3$',hpc['v06_delayed64']),
                       (r'RASPA2，$\langle N\rangle=5.53$',raspa)]:
        rate=f'{value:,.1f}'.removesuffix('.0')
        hpc_rows.append([name,rate,f'{value/hpc["baseline05_exact"]:.2f}',f'{value/raspa:.3f}'])
    write('hpc_table',table(['路径','尝试/s','相对 0.5 exact','相对 RASPA2'],hpc_rows))
    cache=read('host_cache_final')
    assert cache['cache_and_trajectory_match']
    names={'off':'关闭磁盘缓存','cold':'首次写入','warm':'读取复用'}
    write('cache_table',table(['方式','构建数','写入数','加载数','初始化/s','磁盘/MiB'],[
        [names[r['state']],str(r['builds']),str(r['writes']),str(r['loads']),
         f'{r["construction_seconds"]:.3f}',f'{r["file_bytes"]/1024**2:.2f}'] for r in cache['records']]))
    old,new=[r['median_speedup'] for r in read('interpolation')['results']]
    write('interpolation',f'周期查询微测在 350 K 下的配对提速为 {old:.2f} 倍，在关闭热权重预计算的 2 K 下为 {new:.2f} 倍。')
    bench=read('cpu_fourier')
    rows=[]
    for key,label in [('rigid_10_candidates_us','10 个刚体候选'),('single_site_10_trials_with_prefix_us','10 个含前缀的单位点试点'),('selected_molecule_structure_us','所选分子的结构因子'),('replacement_energy_us','完整分子替换能量差'),('translated_structure_us','平移后的结构因子'),('removal_energy_us','删除旧分子的能量')]:
        v=bench['summary'][key]
        times=v['median_us']
        rows.append([label,f'{times["5"]:,.1f}',f'{times["6"]:,.1f}',f'{times["7"]:,.1f}',f'{v["median_paired_speedup_07_vs_06"]:.2f}'])
    write('cpu_table',table(['工作项 / $\\mu$s','0.5','0.6','0.7','0.7/0.6 速度比'],rows))
    gpu=read('gpu_tensor_oracle');compile=read('gpu_tensor_compile')
    assert not compile['gpu_executed'] and not gpu['gpu_executed']
    m=gpu['workspace_model']
    e=max(r['max_energy_error_K'] for r in gpu['primitive']['cases'])
    write('gpu_status',r'\shead{编译、算术与工作存储}'+
          f'完整 CUDA 插件用 NVCC 编译为 {compile["architecture"].replace("_",chr(92)+"_")} 设备代码并链接 CUDA 运行库。CPU 上复用同一张量算术和接口测试替身，三组独立配置的最大能量差为 ${scientific(e)}$ K。此环境没有可执行核函数的物理 GPU。'+
          f'对上述 10 候选任务，能量中间数组按维度计算从 {m["direct_energy_intermediate_bytes"]:,} 降至 {m["tensor_energy_intermediate_bytes"]:,} 字节，缩小 {m["intermediate_reduction_factor"]:.2f} 倍。单个入选分子的结构因子回传 {m["selected_return_bytes"]:,} 字节。')
    all_rows=[]
    lengths=[]
    count_pairs=matched_pairs=0
    for label in ['low','mid','dense','high']:
        if args.draft and not (DATA/(label+'.json')).exists():continue
        case=read(label)
        desc={'low':'低','mid':'中','dense':'较高','high':'高'}[label]
        lengths.append(f'{desc}载量 {case["burnin"]:,}/{case["production_attempts_per_rep"]:,}')
        pair_keys=sorted({(r['seed'],r['mode']) for r in case['rows']})
        for seed,mode in pair_keys:
            old,new=[r for r in case['rows'] if r['seed']==seed and r['mode']==mode]
            count_pairs+=1
            matched_pairs+=int(all(old[key]==new[key] for key in ('counts','counters','final_positions_sha256')))
        for s in case['summary']:
            all_rows.append([desc,s['mode'],f'{s["new_mean_N"]:.2f}',
                             f'{s["old_attempts_s"]:,.0f}',f'{s["new_attempts_s"]:,.0f}',
                             f'{s["old_accepted_s"]:,.0f}',f'{s["new_accepted_s"]:,.0f}',f'{s["median_speedup"]:.2f}'])
    write('chains',table(['条件','模式',r'$\langle N\rangle$','0.6 尝试/s','0.7 尝试/s','0.6 接受/s','0.7 接受/s','倍数'],all_rows)+
          '\n各耗时只计生产。每组预热/生产次数分别为'+ '，'.join(lengths)+'。每组 3 个种子，吞吐列取各自中位数，最右列为配对吞吐比的中位数。0.6 禁用写盘，0.7 开启原始字段缓存。完整初始化时间单列于记录。'+f'{count_pairs} 对轨迹中有 {matched_pairs} 对得到相同的采样粒子数序列、移动计数和末态坐标指纹。')
    cp=read('cpu_fourier_audit'); er=read('endpoint_high')['error_K']
    ts=(DATA/'tests.log').read_text()
    passed=re.search(r'(\d+) passed, (\d+) subtests passed',ts)
    if not passed and not args.draft:raise RuntimeError('Need completed CPU regression log')
    test_phrase=(f'最终回归通过 {passed.group(1)} 项测试及 {passed.group(2)} 个参数化子测试。' if passed else '回归测试正在生成。')
    text=(f'{cp["configurations"]} 组全谱、半谱、三斜晶胞及径向子集配置覆盖 {cp["single_site_candidates"]} 个单位点候选。'+
          f'与独立全模式逐项求和相比，单位点批处理的最大误差为 ${scientific(cp["max_single_site_error_K"])}$ K，'+
          f'改变总电荷的分子替换误差为 ${scientific(cp["max_replacement_error_K"])}$ K。'+
          f'独立高载量整链的末态能量重算差为 ${scientific(er)}$ K。{test_phrase}')
    write('validation',text)

if __name__=='__main__':
    main()
