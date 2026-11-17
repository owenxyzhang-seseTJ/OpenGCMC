"""Compare direct, cell and cached algorithms in the current implementation."""
from pathlib import Path
import argparse,json,os,subprocess,sys

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--section",choices=["all","kernels","mc"],default="kernels")
    p.add_argument("--case",choices=["flex_host","rigid_host","dense_guest"],default="flex_host")
    p.add_argument("--steps",type=int,default=40000)
    p.add_argument("--skip-build",action="store_true")
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    out=root/"benchmarks/results";out.mkdir(parents=True,exist_ok=True)
    env={**os.environ,"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"}
    if not a.skip_build:subprocess.run([sys.executable,str(root/"native/build.py")],cwd=root,env=env,check=True)
    rows=[]
    def worker(*options):
        result=subprocess.run([sys.executable,str(root/"benchmarks/worker.py"),"--source",str(root),*options],env=env,capture_output=True,text=True,check=True)
        return json.loads(result.stdout)
    if a.section in ("all","kernels"):
        for i in range(12,16):
            values={algorithm:worker("--kind","kernel","--fixture",str(root/f"tests/data/fixtures/{i:02d}.bin"),"--algorithm",algorithm) for algorithm in ["direct","cell","cached"]}
            rows.append({"fixture":i,"algorithms":values})
            print("fixture",i,{k:v["median_ms"] for k,v in values.items()},flush=True)
    if a.section in ("all","mc"):
        for seed in [37,43,59]:
            values={algorithm:worker("--kind","mc","--case",a.case,"--seed",str(seed),"--algorithm",algorithm,"--steps",str(a.steps)) for algorithm in ["cell","cached"]}
            rows.append({"seed":seed,"case":a.case,"algorithms":values})
            print("MC",a.case,seed,flush=True)
    (out/"current_cpu_algorithms.json").write_text(json.dumps(rows,indent=2)+"\n")
if __name__=="__main__":main()
