"""Index the stored benchmark records without rerunning experiments."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[1]
rows=[]
for p in sorted((root/"benchmarks/results").glob("*.json")):
    if p.name=="index.json":continue
    data=json.loads(p.read_text())
    rows.append({"record":p.name,"summary":data.get("summary") if isinstance(data,dict) else {"entries":len(data)}})
(root/"benchmarks/results/index.json").write_text(json.dumps(rows,indent=2)+"\n")
print(json.dumps(rows,indent=2))
