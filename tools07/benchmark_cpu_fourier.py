"""Build, pin, and interleave the same C++ harness across three releases."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline06',type=Path,default=ROOT/'baseline_06',help='Unmodified 0.6 source root')
    parser.add_argument('--cpu',type=int)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--output',type=Path,default=ROOT/'evidence07/cpu_fourier.json')
    args=parser.parse_args()
    if args.cpu is not None:
        os.sched_setaffinity(0,{args.cpu})
    compiler=os.environ.get('CXX','g++')
    flags=['-std=c++17','-O3','-fno-fast-math','-ffp-contract=off','-march=native','-fopenmp']
    systems={5:ROOT/'baseline_05/native',6:args.baseline06/'native',7:ROOT/'native'}
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    rows=[]
    with tempfile.TemporaryDirectory() as temp:
        for version,src in systems.items():
            exe=Path(temp)/str(version)
            subprocess.run([compiler,*flags,f'-DPOREMC_CPU_VERSION={version}','-I',str(src),str(ROOT/'tools07/benchmark_cpu_fourier.cpp'),str(src/'ewald_state.cpp'),'-ldl','-o',str(exe)],check=True,env=env)
        for replicate in range(args.repeats):
            order=[5,6,7]
            order=order[replicate%3:]+order[:replicate%3]
            if replicate%2: order.reverse()
            for version in order:
                result=json.loads(subprocess.check_output([str(Path(temp)/str(version))],env=env,text=True))
                rows.append(dict(replicate=replicate,**result))
    observed=[k for k in rows[0] if k.endswith('_us')]
    summary={}
    for k in observed:
        medians={str(v):statistics.median(r[k] for r in rows if r['version']==v) for v in systems}
        ratios=[]
        for i in range(args.repeats):
            paired={r['version']:r[k] for r in rows if r['replicate']==i}
            ratios.append(paired[6]/paired[7])
        summary[k]={'median_us':medians,'median_paired_speedup_07_vs_06':statistics.median(ratios),'paired_ratios':ratios}
    result={'scope':'CPU subroutines only; synthetic fixed inputs, including 5 resident molecules; single thread; kmax=13, rectangular target, 4416 host sites, 12 molecular sites of which eight charged, ten trials. Full physical model unchanged. Times exclude the Python driver.','model':platform.platform(),'compiler':subprocess.check_output([compiler,'--version'],text=True).splitlines()[0],'flags':flags,'affinity':sorted(os.sched_getaffinity(0)),'repeats':args.repeats,'harness_sha256':sha(ROOT/'tools07/benchmark_cpu_fourier.cpp'),'source_sha256':{v:sha(p/'ewald_state.cpp') for v,p in systems.items()},'rows':rows,'summary':summary}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
