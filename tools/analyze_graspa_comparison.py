#!/usr/bin/env python3
"""Compare completed short gRASPA runs to the current native cached sampler."""
import argparse
from datetime import datetime,timezone
import itertools
import json
from pathlib import Path
import platform
import re
import statistics
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from poremc.kernels import NativeEnvironment,ReferenceEnvironment
from native_benchmark import NativePointSampler
from analyze_raspa3_comparison import energy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.directory
    model=json.loads((root/'inputs/model.json').read_text())
    records=json.loads((root/'graspa_timings.json').read_text())
    execution_path=root/'execution.json'
    execution=json.loads(execution_path.read_text()) if execution_path.exists() else {
        'CUDA_LAUNCH_BLOCKING':'1','shared_device_existing_task':True,'device_temperature_limit_C':87}
    unit=np.array(model['unit_cell']);cell=np.diag(model['replication'])@unit
    shifts=np.array(list(itertools.product(*(range(n) for n in model['replication']))))@unit
    hostpos=np.concatenate([np.array(model['fractional'])@unit+s for s in shifts])
    sigma=np.tile(model['sigma'],len(shifts));epsilon=np.tile(model['epsilon'],len(shifts));q=np.zeros(len(sigma))
    result={'software':'OpenGCMC current source','date_UTC':datetime.now(timezone.utc).isoformat(),
            'platform':platform.platform(),'python':sys.version,'numpy':np.__version__,
            'model':model,'framework_sites':len(hostpos),'replicates':[],
            'sampling_protocol':'Both start empty, no warm-up, 1000 scheduled insertion/deletion attempts, K=1, equal selection probabilities. gRASPA UseMaxStep=yes and MaxStepPerCycle=1.',
            'timing_scope':'gRASPA stock Work took timer: after preparation and before final energy checks; includes native standard production statistics. OpenGCMC native production loop includes count sampling every 20 attempts. Setup excluded.',
            'gRASPA_source':'https://github.com/snurr-group/gRASPA/tree/e6da4e376f19047df6cb85b9d20685d1bf0777e8',
            'execution_conditions':execution,
            'measure':'Short-run scheduled attempts per second; not an equilibrium ESS comparison.'}
    kw=dict(coulomb='none',shifted=True,alpha=.25)
    with NativeEnvironment(hostpos,sigma,epsilon,q,cell,12.,algorithm='cached',threads=1,**kw) as host,ReferenceEnvironment(hostpos,sigma,epsilon,q,cell,12.,algorithm='direct',**kw) as ref:
        for base in records:
            seed=base['seed'];d=root/f'timed_seed_{seed}';log=(d/'console.log').read_text()
            assert 'END OF PROGRAM' in log and 'Max Step = 1' in log
            assert 'CBMC Uses 1 trial positions and 1 trial orientations' in log
            config=(d/'simulation.input').read_text();assert 'NumberOfInitializationCycles 0' in config
            steps=int(re.search(r'NumberOfProductionCycles\s+(\d+)',config)[1]);assert steps==base['steps']==1000
            stage=re.search(r'\*\*\* FINAL STAGE \*\*\*(.*?)\*\*\* RUNNING DELTA_E',log,re.S)[1]
            final_logged_energy=float(re.search(r'Total Energy:\s+[-+0-9.eE]+\s+\(([-+0-9.eE]+) \[K\]',stage)[1])
            # The stock executable writes its restart at PrintEvery. With the
            # sparse logging used here, that snapshot is cycle 0, not the final
            # production state. Compare it to the matching cycle-0 energy.
            snapshot_internal=float(re.search(r'PRODUCTION Cycle:\s+0,.*?Total Energy:\s+([-+0-9.eE]+)',log)[1])
            upstream_energy=snapshot_internal*1.20272430057
            restart=(d/'Restart/System_0/restartfile').read_text()
            matches=re.findall(r'Adsorbate-atom-position:\s+\d+\s+\d+\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)',restart)
            points=np.asarray(matches,dtype=float).reshape(-1,3)
            full=energy(ref,points);err=abs(full-upstream_energy)
            assert err<=2e-5,(upstream_energy,full)
            with NativePointSampler(host,300.,model['activity_per_A3'],seed) as sim:
                trace,timing=sim.run(steps,sample_every=20)
                own=sim.positions();running=sim.energy();recomputed=energy(ref,own)
                assert np.isclose(running,recomputed,rtol=1e-10,atol=2e-7)
                assert int(sim.counters()[:,0].sum())==steps
            seconds=float(timing[0]);ratio=base['gRASPA_sampling_seconds']/seconds
            insertions=int(re.findall(r'Insertion Performed:\s+(\d+)',log)[-1])
            deletions=int(re.findall(r'Deletion Performed:\s+(\d+)',log)[-1])
            row={**base,'opengcmc_sampling_seconds':seconds,'throughput_ratio':ratio,
                 'gRASPA_scheduled_attempts':steps,'gRASPA_reported_insertions':insertions,'gRASPA_reported_deletions':deletions,
                 'opengcmc_scheduled_attempts':steps,'gRASPA_snapshot_cycle':0,'gRASPA_snapshot_positions_A':points.tolist(),
                 'gRASPA_snapshot_logged_energy_K':upstream_energy,'gRASPA_final_logged_energy_K':final_logged_energy,
                 'opengcmc_on_gRASPA_snapshot_energy_K':full,
                 'cross_engine_energy_error_K':err,'gRASPA_logged_energy_decimal_places':5,
                 'opengcmc_final_positions_A':own.tolist(),'opengcmc_final_energy_K':running,
                 'opengcmc_recomputed_final_energy_K':recomputed,'opengcmc_count_trace':trace.tolist()}
            result['replicates'].append(row);print('PAIRED_GRASPA',seed,'ratio',ratio,'energy_error',err,flush=True)
    ratios=[r['throughput_ratio'] for r in result['replicates']]
    result['summary']={'median_throughput_ratio':statistics.median(ratios),'min_throughput_ratio':min(ratios),'max_throughput_ratio':max(ratios),
        'all_energy_checks_passed':True,'maximum_logged_energy_error_K':max(r['cross_engine_energy_error_K'] for r in result['replicates']),
        'scheduled_attempts_per_repeat':1000,'repeats':len(ratios)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print('SUMMARY',result['summary'],flush=True)


if __name__=='__main__':main()
