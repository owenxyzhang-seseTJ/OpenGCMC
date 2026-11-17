#!/usr/bin/env python3
"""Compare completed RASPA3 CLI runs with the current native cached sampler.

Input directories contain simulation.json, model.json at the parent level,
and complete RASPA output. Sampling runs are serial and use actual move counts.
"""
from datetime import datetime, timezone
import argparse
import itertools
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from poremc.kernels import NativeEnvironment, ReferenceEnvironment
from poremc.statistics import summarize
from native_benchmark import NativePointSampler


def energy(host,positions,sigma=3.,epsilon=5.):
    p=np.asarray(positions,dtype=float).reshape(-1,3)
    value=float(np.sum(host.energy(p[:,None,:],[sigma],[epsilon],[0.])))
    inv=np.linalg.inv(host.cell)
    for i in range(len(p)):
        d=(p[i]-p[:i])@inv;d-=np.rint(d)
        r2=np.sum((d@host.cell)**2,axis=1);r2=r2[r2<host.cutoff**2]
        t=(sigma*sigma/r2)**3;c=(sigma/host.cutoff)**6
        value+=float(np.sum(4*epsilon*(t*(t-1)-c*(c-1))))
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();model=json.loads((args.directory/'model.json').read_text())
    unit=np.asarray(model['unit_cell']);cell=np.diag(model['replication'])@unit
    shifts=np.array(list(itertools.product(*(range(n) for n in model['replication']))))@unit
    hostpos=np.concatenate([np.asarray(model['fractional'])@unit+s for s in shifts])
    hs=np.tile(model['sigma'],len(shifts));he=np.tile(model['epsilon'],len(shifts));hq=np.zeros(len(hs))
    result={'date_UTC':datetime.now(timezone.utc).isoformat(),'software':'OpenGCMC current source',
            'platform':platform.platform(),'python':sys.version,'numpy':np.__version__,
            'model':model,'framework_sites':len(hostpos),'replicates':[],
            'timing_scope':'Native production time. Initialization excluded. RASPA property defaults retained. OpenGCMC samples molecule counts every 20 attempts; energy recomputation occurs outside timing.',
            'native_interface':'Production C ABI through tools/native_benchmark.py',
            'move_family':'Conventional single-site insertion/deletion with equal probabilities; one trial per insertion, shifted LJ, neutral, no tail corrections.'}
    kw=dict(coulomb='none',shifted=True,alpha=.25)
    with NativeEnvironment(hostpos,hs,he,hq,cell,12.,algorithm='cached',threads=1,**kw) as host, ReferenceEnvironment(hostpos,hs,he,hq,cell,12.,algorithm='direct',**kw) as reference:
        for seed in model['seeds']:
            directory=args.directory/f'seed_{seed}'
            upstream=json.loads(next((directory/'output').glob('output*.json')).read_text())
            config=json.loads((directory/'simulation.json').read_text())
            stats=upstream['output']['MCMoveStatistics']['system']['probe']
            moves=int(stats['Swap']['all']);assert moves==model['production_cycles']*20
            assert stats['Swap (CBMC)']['all']==0
            rtime=float(upstream['output']['cpuTimings']['production'])
            beta=float(upstream['initialization']['initialConditions']['beta'])
            units=upstream['initialization']['units']
            temperature=units['Boltzmann constant [-]']*model['temperature_K']*units['Energy to Kelvin [-]']
            activity=model['pressure_Pa']/units['pressure [Pa]']*beta
            assert math.isclose(activity,model['activity_per_A3'],rel_tol=1e-10)
            rst=json.loads(next((directory/'output').glob('restart*.json')).read_text())
            positions=np.asarray(rst['probe']).reshape(-1,3)
            u_r=float(upstream['output']['recomputedEnergies']['Total potential energy [K]'])
            u_n=energy(reference,positions)
            error=abs(u_r-u_n)
            assert np.isclose(u_r,u_n,rtol=1e-10,atol=2e-7),(u_r,u_n)
            with NativePointSampler(host,temperature,activity,seed) as sim:
                # Both programs start empty; 500 initialization cycles correspond
                # to 10,000 attempts in this low-loading single-component setup.
                before=sim.counters()
                trace,timing=sim.run(moves,burnin=model['initialization_cycles']*20,sample_every=20)
                final=sim.positions();u_fast=sim.energy();u_full=energy(reference,final)
                assert np.isclose(u_fast,u_full,rtol=1e-10,atol=2e-7),(u_fast,u_full)
                assert int((sim.counters()-before)[:,0].sum())==moves+model['initialization_cycles']*20
                profile=host.profile
            seconds=float(timing[0]);ratio=rtime/seconds
            row={'seed':seed,'actual_attempts':moves,'raspa_version':upstream['version'],
                 'raspa_production_seconds':rtime,'opengcmc_production_seconds':seconds,'throughput_ratio':ratio,
                 'raspa_moves_per_second':moves/rtime,'opengcmc_moves_per_second':moves/seconds,
                 'raspa_final_positions_A':positions.tolist(),'raspa_final_energy_K':u_r,
                 'opengcmc_on_raspa_snapshot_energy_K':u_n,'cross_engine_energy_error_K':error,
                 'opengcmc_final_positions_A':final.tolist(),'opengcmc_final_energy_K':u_fast,
                 'opengcmc_final_recomputed_energy_K':u_full,'opengcmc_count_trace':trace.tolist(),
                 'opengcmc_count_statistics':summarize(trace,seconds),'opengcmc_host_profile':profile,
                 'opengcmc_temperature_matched_K':temperature,'matched_activity_per_A3':activity,
                 'upstream_move_counts':{'conventional':int(stats['Swap']['all']),'CBMC':int(stats['Swap (CBMC)']['all'])}}
            result['replicates'].append(row)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
            print('PAIRED',seed,'ratio',ratio,'energy_error_K',error,flush=True)
        ratios=[x['throughput_ratio'] for x in result['replicates']]
        result['summary']={'median_throughput_ratio':statistics.median(ratios),'min_throughput_ratio':min(ratios),'max_throughput_ratio':max(ratios),
                           'all_energy_checks_passed':True,'max_cross_engine_energy_error_K':max(x['cross_engine_energy_error_K'] for x in result['replicates']),
                           'sample_count_per_engine_per_repeat':model['production_cycles'],
                           'nonempty_upstream_snapshots':sum(bool(x['raspa_final_positions_A']) for x in result['replicates'])}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('SUMMARY',result['summary'],flush=True)


if __name__=='__main__':main()
