from pathlib import Path
import sys,tempfile,time,json
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from tools.fixtures import read_fixture
from poremc.topology import Molecule
from poremc.potentials import ClassicalPotential
from poremc.proposals import build_host_proposal
from poremc.accelerators import GridHostSurrogate
f=root/'tests/data/fixtures/14.bin';d=read_fixture(f)
m=Molecule('probe',[2.9],[65.]);p=ClassicalPotential([m],d['cell'],d['cutoff'],d['host'],d['hs'],d['he'],d['hq'],algorithm='cached')
records=[]
for rep in range(3):
 with tempfile.TemporaryDirectory() as directory:
  one=GridHostSurrogate(p,shape=(24,24,24),cache_dir=directory);two=GridHostSurrogate(p,shape=(24,24,24),cache_dir=directory)
  first=build_host_proposal(p,m,300,shape=(24,24,24),cache_dir=directory);nextT=build_host_proposal(p,m,320,shape=(24,24,24),cache_dir=directory)
  records.append({'grid_cold':one.metadata,'grid_disk_hit':two.metadata,'root_cold':first.metadata,'root_reweight_other_T':nextT.metadata})
  assert two.metadata['cache_hit'] and nextT.metadata['cache_hit']
(root/'benchmarks/results/persistence_benchmark.json').write_text(json.dumps(records,indent=2))
print([(r['grid_cold']['setup_seconds'],r['grid_disk_hit']['setup_seconds']) for r in records])
