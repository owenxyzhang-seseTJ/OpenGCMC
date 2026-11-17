"""Compare cached and cell-list native trajectories, states and acceptances."""
from pathlib import Path
import json,subprocess,sys,os
root=Path(__file__).resolve().parents[1]
env={**os.environ,"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1"}
records=[]
for case in ["flex_host","rigid_host","dense_guest"]:
    for seed in [37,43,59]:
        values=[]
        for algorithm in ["cell","cached"]:
            p=subprocess.run([sys.executable,str(root/"benchmarks/worker.py"),"--source",str(root),"--kind","mc","--case",case,"--seed",str(seed),"--steps","10000","--algorithm",algorithm],env=env,capture_output=True,text=True,check=True)
            values.append(json.loads(p.stdout))
        a,b=values
        row={"case":case,"seed":seed,"counts_identical":a["counts"]==b["counts"],"counters_identical":a["acceptance"]==b["acceptance"],"positions_identical":a["final_positions"]==b["final_positions"]}
        records.append(row);print(row,flush=True)
output=root/"benchmarks/results/state_replays.json";output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps(records,indent=2)+"\n")
assert all(r[k] for r in records for k in ["counts_identical","counters_identical","positions_identical"])
