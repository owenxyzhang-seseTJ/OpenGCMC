"""Host-C++ syntax check of CUDA source, NOT a CUDA compiler or GPU execution test."""
from pathlib import Path
import subprocess,tempfile,json
root=Path(__file__).resolve().parents[1]
header='''#include <cmath>
#define __global__
#define __shared__ static
struct Dim {unsigned x=0,y=0,z=0;};
Dim blockIdx,blockDim,threadIdx;
inline void __syncthreads() {}
'''
with tempfile.TemporaryDirectory() as d:
    source=Path(d)/'cuda_syntax.cpp';source.write_text(header+(root/'poremc/cached_cuda.cu').read_text())
    command=['g++','-std=c++17','-fsyntax-only',str(source)]
    p=subprocess.run(command,capture_output=True,text=True)
    record={'kind':'host-C++ CUDA-source syntax only','passed':p.returncode==0,'cuda_compiled':False,'gpu_executed':False,'stderr':p.stderr}
    (root/'benchmarks/cuda_syntax.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record));raise SystemExit(p.returncode)
