"""Check and time identical periodic mixed cubic/Boltzmann grid queries in 0.6/0.7."""
from pathlib import Path
import argparse,json,os,statistics,subprocess,tempfile

ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'evidence07/interpolation.json')
    args=p.parse_args()
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    summary=[]
    with tempfile.TemporaryDirectory() as tmp:
        for tag,src in [('06',ROOT/'baseline_06/native'),('07',ROOT/'native')]:
            subprocess.run([os.environ.get('CXX','g++'),'-std=c++17','-O3','-march=native','-ffp-contract=off','-fno-fast-math','-fopenmp',
                '-I',str(src),str(ROOT/'tools07/benchmark_interpolation.cpp'),str(src/'host_grid.cpp'),str(src/'kernels.cpp'),'-o',str(Path(tmp)/tag)],check=True)
        for T in (350.,2.):
            ratios=[];rows=[];maximum=0.
            for rep in range(3):
                pair={}
                for tag in (('06','07') if rep%2==0 else ('07','06')):
                    pair[tag]=json.loads(subprocess.check_output([str(Path(tmp)/tag),str(T)],text=True))
                maximum=max(maximum,max(abs(x-y) for x,y in zip(pair['06'].pop('probes'),pair['07'].pop('probes'))))
                for k in ('cubic','boltzmann'):assert pair['06'][k]==pair['07'][k] and pair['07'][k]>0
                ratios.append(pair['07']['queries_s']/pair['06']['queries_s']);rows.append(pair)
            assert maximum<2e-8
            summary.append(dict(T_K=T,max_abs_difference_K=maximum,paired_speedups=ratios,median_speedup=statistics.median(ratios),rows=rows))
    args.output.write_text(json.dumps(dict(scope='isolated single-core periodic interpolation, no Markov chain',results=summary),indent=2)+'\n')
if __name__=='__main__':main()
