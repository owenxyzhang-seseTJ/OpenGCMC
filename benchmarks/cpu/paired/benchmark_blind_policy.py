"""Lock a precision-cost decision on a development system before the holdout."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from benchmark_precision import write_json


HERE=Path(__file__).resolve().parent
HOLDOUT='holdout_co2_uio66'
MODES=('exact','delayed')


def cost_estimate(report, index):
    stage=report['checkpoints'][index]
    obs=stage['observables']
    goals=report['precision_goals']
    longest_production=max(ch['profiles'][index]['production_wall_s'] for ch in report['chains'])
    fixed=max(0.,stage['batch_wall_s']-longest_production)
    ready=all(x['mcse'] is not None and x['rank_split_rhat'] is not None for x in obs.values())
    slowdown=max(1.,*( (obs[k]['mcse']/tol)**2 for k,tol in goals.items())) if ready else None
    delta_triggers=[]
    delta_valid=[]
    for ch in report['chains']:
        final=ch['profiles'][index]['acceleration']
        warm=ch['acceleration_after_warmup']
        delta_triggers.append(final['stage1_passed']-warm['stage1_passed'])
        delta_valid.append(final['stage1_evaluated']-warm['stage1_evaluated'])
    return {
        'predicted_s':None if slowdown is None else fixed+longest_production*slowdown,
        'fixed_wall_s':fixed,
        'max_prepare_s':max(ch['preparation_s'] for ch in report['chains']),
        'seconds_per_attempt_batch':longest_production/stage['budget'],
        'worst_rhat':max(x['rank_split_rhat'] for x in obs.values()) if ready else None,
        'ESS_penalty':{k:stage['draws_per_chain']*len(report['chains'])/v['ess'] if v.get('ess') else None
                       for k,v in obs.items()},
        'fraction_of_valid_candidates_reaching_exact':sum(delta_triggers)/sum(delta_valid) if sum(delta_valid) else None,
    }


def decide(scores, ratio, max_rhat):
    ex,da=(scores[m] for m in MODES)
    if (ex['predicted_s'] is not None and da['predicted_s'] is not None and
        da['worst_rhat']<=max_rhat and da['predicted_s']<ratio*ex['predicted_s']):
        return 'delayed'
    return 'exact'


def lock_policy(development, path):
    scores={}
    for mode in MODES:
        p=development/f'control-{mode}.json'
        r=json.loads(p.read_text())
        scores[mode]=cost_estimate(r,0)
    protocol={
        'locked_utc':datetime.now(timezone.utc).isoformat(),
        'holdout':HOLDOUT,
        'holdout_condition':{'temperature_K':350.,'fugacity_Pa':60000.,
                             'framework':'UiO-66','mesh':32},
        'development':'charged IRMOF-1 CO2/CH4 at 2 bar',
        'development_pilot':scores,
        'required_predicted_ratio':.85,
        'pilot_max_rhat':1.10,
        'pilot_budget':20000,
        'pilot_burnin':20000,
        'pilot_seed_offset':731000,
        'confirmation_seed_offset':928000,
        'confirmation_budgets':[20000,60000,120000,240000],
    }
    protocol['development_decision']=decide(scores,.85,1.10)
    with path.open('x') as f:
        json.dump(protocol,f,indent=2)
        f.write('\n')


def execute(path, output, cache):
    policy=json.loads(path.read_text())
    if policy['holdout']!=HOLDOUT:
        raise ValueError('the model is not the registered holdout')
    output.mkdir(parents=True,exist_ok=True)
    model=HERE/'inputs'/f'{HOLDOUT}.json'
    def run(mode, label, budgets, offset):
        target=output/f'blind-{label}-{mode}.json'
        cmd=[sys.executable,'-B',str(HERE/'benchmark_precision.py'),str(model),'--output',str(target),
             '--mode',mode,'--cache',str(cache/label/mode),'--warmup',str(policy['pilot_burnin']),
             '--seed-offset',str(offset),'--mesh',str(policy['holdout_condition']['mesh']),
             '--budgets',*map(str,budgets)]
        subprocess.run(cmd,check=True)
        return json.loads(target.read_text())

    pilot={mode:cost_estimate(run(mode,'pilot',[policy['pilot_budget']],policy['pilot_seed_offset']),0)
           for mode in MODES}
    decision=decide(pilot,policy['required_predicted_ratio'],policy['pilot_max_rhat'])
    with (output/'blind-decision.json').open('x') as f:
        json.dump({'utc':datetime.now(timezone.utc).isoformat(),'predicted_choice':decision,'pilot':pilot},f,indent=2)
    confirmation={mode:run(mode,'confirmation',policy['confirmation_budgets'],
                           policy['confirmation_seed_offset'])['qualified_wall_s'] for mode in MODES}
    write_json(output/'blind-result.json',{
        'model':HOLDOUT,'predicted_choice':decision,'time_to_precision_s':confirmation,
        'faster_qualified_mode':min(confirmation,key=confirmation.get) if all(
            v is not None for v in confirmation.values()) else None,
    })


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True,type=Path)
    parser.add_argument('--lock-from',type=Path)
    parser.add_argument('--execute-to',type=Path)
    parser.add_argument('--cache',type=Path)
    args=parser.parse_args()
    if args.lock_from:
        lock_policy(args.lock_from,args.protocol)
    elif args.execute_to and args.cache:
        execute(args.protocol,args.execute_to,args.cache)
    else:
        parser.error('use --lock-from or --execute-to with --cache')
