#!/usr/bin/env python3
"""Prepare immutable plans on CPU; compile and compare CUDA kernels without CuPy.

All output records describe actual local execution. This tool never substitutes
CPU timing when a GPU or CUDA compiler is unavailable.
"""
from pathlib import Path
import argparse,json,os,shutil,struct,subprocess,sys,tempfile,time,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from poremc.kernels import NativeEnvironment,COULOMB_K
from tools.fixtures import read_fixture,env_args,env_kwargs

def prepare(d,path):
    t=time.perf_counter()
    with NativeEnvironment(*env_args(d),**env_kwargs(d),algorithm='cached') as env:
        plan=env.export_spatial_plan();order=plan['order'];fs=np.ascontiguousarray((d['host']@np.linalg.inv(d['cell']))%1.0)
        params=np.column_stack([d['ts'],d['te'],d['tq']]);types,ids=np.unique(params,axis=0,return_inverse=True)
        hs,he,hq=[d[k][order] for k in ['hs','he','hq']];table=np.empty((len(types),4,len(hs)))
        for i,(s,e,q) in enumerate(types):
            s2=(.5*(s+hs))**2;e4=4*np.sqrt(e)*np.sqrt(he);c3=(s2/d['cutoff']**2)**3
            table[i]=[s2,e4,e4*c3*(c3-1) if d['shifted'] else np.zeros(len(hs)),COULOMB_K*q*hq]
        with open(path,'wb') as f:
            f.write(struct.pack('<8s6i2Q',b'PCACHE02',*plan['bins'],int(plan['indexed']),int(np.count_nonzero(d['cell']-np.diag(np.diag(d['cell'])))==0),len(types),len(plan['offsets']),len(plan['indices'])))
            for array,dtype in [(fs[order].T,'<f8'),(table,'<f8'),(ids,'<i4'),(plan['offsets'],'<u8'),(plan['indices'],'<u4')]:np.asarray(array,dtype=dtype).tofile(f)
        return {'plan_setup_seconds':time.perf_counter()-t,'profile':env.profile,'plan_bytes':Path(path).stat().st_size}

def device_state():
    p=subprocess.run(['nvidia-smi','--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    return p.stdout.strip()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arch',default='sm_86');p.add_argument('--repeats',type=int,default=7);p.add_argument('--nvcc');p.add_argument('--output',default='benchmarks/results/gpu');p.add_argument('--max-temperature',type=float,default=87.);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    out=ROOT/a.output;out.mkdir(parents=True,exist_ok=True)
    fixture=ROOT/'tests/data/fixtures';records=[]
    if not a.prepare_only:
        nvcc=a.nvcc or shutil.which('nvcc')
        if not nvcc and os.environ.get('POREMC_CUDA_ROOT'):nvcc=str(Path(os.environ['POREMC_CUDA_ROOT'])/'bin/nvcc')
        if not nvcc:raise SystemExit('nvcc not found; set --nvcc or POREMC_CUDA_ROOT. No GPU timings were produced.')
        command=[nvcc,'-O3','-std=c++17','--fmad=false','-arch='+a.arch,str(ROOT/'tools/gpu_cached_driver.cu'),'-o',str(out/'driver')]
        subprocess.run(command,check=True);(out/'compile.json').write_text(json.dumps({'command':command,'compiler':subprocess.check_output([nvcc,'--version'],text=True)},indent=2))
    for i in range(16):
        f=fixture/f'{i:02d}.bin';d=read_fixture(f);planfile=out/f'{i:02d}.plan';r={'fixture':i,**prepare(d,planfile)}
        if not a.prepare_only:
            r['device_before']=device_state();temperature=float(r['device_before'].splitlines()[0].split(',')[1])
            if temperature>=a.max_temperature:r['status']='SKIPPED_TEMPERATURE'
            else:
                prefix=out/f'{i:02d}';subprocess.run([str(out/'driver'),str(f),str(planfile),str(prefix),str(a.repeats if i>=12 else 0)],check=True)
                r['timing']=json.loads(prefix.with_suffix('.json').read_text());expected=np.fromfile(f.with_suffix('.gpu'),dtype='<f8')
                r['comparisons']={}
                for name in ['direct','cached']:
                    v=np.fromfile(str(prefix)+'_'+name+'.bin',dtype='<f8');finite=np.isfinite(v)&np.isfinite(expected)
                    ok=np.allclose(v,expected,rtol=1e-9 if i>=12 else 2e-11,atol=1e-6 if i>=12 else 1e-7)
                    r['comparisons'][name]={'passed':bool(ok),'count':len(v),'max_scaled_error':float(np.max(np.abs(v[finite]-expected[finite])/np.maximum(1,np.abs(expected[finite])),initial=0))}
                r['status']='PASS' if all(v['passed'] for v in r['comparisons'].values()) else 'FAIL'
                r['device_after']=device_state()
        else:r['status']='PREPARED_CPU_ONLY'
        records.append(r);(out/'results.json').write_text(json.dumps(records,indent=2));print(i,r['status'],flush=True)
    return int(any(r['status']=='FAIL' for r in records))
if __name__=='__main__':raise SystemExit(main())
