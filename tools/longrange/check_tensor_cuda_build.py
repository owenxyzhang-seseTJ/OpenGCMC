#!/usr/bin/env python3
"""Compile the complete CUDA plug-in including sm_86 machine code; probe driver availability."""
import argparse
import ctypes as C
import json
from pathlib import Path
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cuda-root',type=Path,required=True)
    p.add_argument('--arch',default='sm_86')
    p.add_argument('--output',type=Path,default=Path('tensor_build_validation.json'))
    args=p.parse_args();toolkit=args.cuda_root.resolve();nvcc=str(toolkit/'bin/nvcc')
    libdir=next(d for d in (toolkit/'lib64',toolkit/'lib') if list(d.glob('libcudart.so.*')))
    runtime=sorted(libdir.glob('libcudart.so.*'))[0]
    source=ROOT/'native/ewald_cuda.cu'
    with tempfile.TemporaryDirectory() as temp:
        obj=Path(temp)/'tensor.o';plugin=Path(temp)/'poremc_cuda_tensor.so'
        compile_args=[nvcc,'-O3','--fmad=false','-std=c++17','-Xcompiler','-fPIC','-Xptxas=-v','-c','-arch='+args.arch,str(source),'-o',str(obj)]
        compile_log=subprocess.run(compile_args,capture_output=True,text=True,check=True)
        subprocess.run(['c++','-shared',str(obj),str(runtime),'-o',str(plugin)],check=True)
        cudart=C.CDLL(str(runtime),mode=C.RTLD_GLOBAL)
        library=C.CDLL(str(plugin));symbols=[n for n in ('create','batch','set_lattice','structure','device_name','kernel_name','destroy','last_error') if getattr(library,'poremc_ewald_cuda_'+n)]
        devices=C.c_int();status=cudart.cudaGetDeviceCount(C.byref(devices));cudart.cudaGetErrorString.restype=C.c_char_p
        proof=dict(scope='Full production CUDA translation unit compiled to device machine code and linked. No GPU kernel executed by this compile audit.',
            gpu_executed=False,architecture=args.arch,nvcc=subprocess.check_output([nvcc,'--version'],text=True).strip(),flags=compile_args[1:9],
            ptxas=compile_log.stderr.splitlines(),abi_symbols=symbols,runtime=runtime.name,device_query_status=status,
            device_query_diagnostic=cudart.cudaGetErrorString(status).decode(),available_device_count=devices.value)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({key:proof[key] for key in ('architecture','device_query_diagnostic','available_device_count','gpu_executed')}))

if __name__=='__main__':main()
