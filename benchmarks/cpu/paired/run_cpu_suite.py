"""Run matched native and RASPA3 CPU benchmarks on one allocated worker group."""

import argparse
from pathlib import Path
import subprocess
import sys

from benchmark_precision import read_model, setup

HERE=Path(__file__).resolve().parent
GROUPS=('controls','xylene_low','xylene_mid','xylene_high','ton_butane',
        'irmof_mixture','nist','cfcmc','ton_butane_dilute','xylene_low_confirm',
        'holdout_same_candidates')
NATIVE_ONLY={'controls','nist','xylene_low_confirm','holdout_same_candidates'}


def command(script,*args):
    print(script,*args,flush=True)
    subprocess.run([sys.executable,'-B',str(HERE/script),*map(str,args)],check=True)


def run(args):
    models=HERE/'inputs'
    output=args.output.resolve()
    cache=args.cache.resolve()
    output.mkdir(parents=True,exist_ok=True)

    def native(case,mode,budgets,warmup,mesh=32,guest='incremental',label=None,seed_offset=0):
        command('benchmark_precision.py',models/f'{case}.json','--output',output/f'{label or case+"-"+mode}.json',
                '--cache',cache,'--mode',mode,'--budgets',*budgets,'--warmup',warmup,
                '--mesh',mesh,'--guest-update',guest,'--workers',4,'--seed-offset',seed_offset)

    def raspa(case,budgets,warmup,stride=5,mode='cbmc'):
        command('benchmark_raspa3.py',models/f'{case}.json','--binary',args.raspa3,
                '--output',output/f'{case}-raspa-{mode}.json','--mode',mode,
                '--budgets',*budgets,'--warmup',warmup,'--stride',stride,'--workers',4)

    case=args.group
    if case=='xylene_low_confirm':
        path=models/'xylene_low.json'
        with setup(read_model(path),path,1,'delayed',cache,32,'incremental'):
            pass
        native('xylene_low','delayed',[120000,360000,720000,1440000],30000,
               label='xylene_low-delayed-long',seed_offset=150000)
        return
    if case=='holdout_same_candidates':
        path=models/'holdout_co2_uio66.json'
        with setup(read_model(path),path,1,'delayed',cache,32,'incremental'):
            pass
        for mode in ('reference-mh','delayed'):
            native(path.stem,mode,[240000,480000,960000,1440000,1920000],20000,
                   label=f'holdout-warm-{mode}',seed_offset=928000)
        return
    if case=='nist':
        for model in ('nist_co2_zif8','nist_n2_co2_zif8'):
            for mode in ('delayed','reference-mh'):
                native(model,mode,[20000,60000,120000,240000],20000)
        return
    if case=='controls':
        path=models/'irmof_mixture.json'
        for mesh in (32,8):
            with setup(read_model(path),path,1,'delayed',cache,mesh,'incremental'):
                pass
        for mode in ('reference-mh','delayed','exact'):
            native('irmof_mixture',mode,[20000,60000,120000,240000],20000,
                   label=f'control-{mode}')
        native('irmof_mixture','delayed',[20000,60000,120000,240000],20000,
               mesh=8,label='control-delayed-mesh8')
        native('irmof_mixture','delayed',[20000,60000,120000,240000],20000,
               guest='rebuild',label='control-delayed-rebuild')
        return
    if case=='cfcmc':
        raspa('irmof_mixture',[5000,10000,15000],15000,stride=10,mode='cfcmc')
        return
    if case.startswith('xylene'):
        native(case,'delayed',[40000,120000,240000,480000],30000)
        if case=='xylene_mid':
            native(case,'reference-mh',[40000,120000,240000,480000],30000)
            native(case,'exact',[8000,20000,40000],10000,label='xylene_mid-exact-cost')
        budget=[400,1000,2000,3600] if case=='xylene_high' else [800,2400,4800,8000]
        raspa(case,budget,1000,8)
    elif case=='ton_butane_dilute':
        native(case,'delayed',[60000,180000,400000],20000)
        native('ton_butane_dilute_suffix','delayed',[60000,180000,400000],20000)
        raspa(case,[800,1600,4000,8000],1000,8)
    elif case=='ton_butane':
        native('butane_harmonic_ideal','delayed',[20000,60000,120000],6000)
        raspa('butane_harmonic_ideal',[1000,2500,5000],1000,5)
        native('ton_butane_harmonic','delayed',[100000,240000,480000,840000],35000)
        native('ton_butane_harmonic_suffix','delayed',[100000,240000,480000,840000],35000)
        raspa('ton_butane_harmonic',[2000,6000,12000,21000],2500,10)
    else:
        native(case,'delayed',[20000,60000,120000,240000],20000)
        raspa(case,[1000,3000,6000,12000],1000,5)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('group',choices=GROUPS)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--cache',required=True,type=Path)
    p.add_argument('--raspa3',type=Path)
    a=p.parse_args()
    if a.group not in NATIVE_ONLY and (not a.raspa3 or not a.raspa3.is_file()):
        p.error('this group needs a working RASPA3 binary')
    run(a)
