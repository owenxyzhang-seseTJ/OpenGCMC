"""Export benchmark precision checkpoints and verify their retained samples."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from benchmark_precision import diagnostics


def records(study):
    for folder, machine in (('m5', 'Apple M5'), ('hpc', 'HPC Xeon Max 9468')):
        directory = study / folder
        for source in sorted([*directory.glob('*.json'), *directory.glob('*/*.json')]):
            run = json.loads(source.read_text())
            if 'checkpoints' not in run:
                continue
            case = run['case']
            model = json.loads((study / 'inputs' / f'{case}.json').read_text())
            metadata = model['metadata']
            chain_counts, trans_counts = [], []
            chains = run['chains']
            for chain in chains:
                with np.load(source.parent / chain['data'], allow_pickle=False) as data:
                    chain_counts.append(data['counts'])
                    trans_counts.append(data['trans_count'])
            counts, trans = np.asarray(chain_counts), np.asarray(trans_counts)
            groups = np.asarray([chain['start'] for chain in chains])
            if counts.shape[0] != 4 or list(groups).count('empty') != 2:
                raise ValueError(f'Expected two chains per starting state in {source}')
            raspa = run.get('engine') == 'RASPA3'
            valid_budgets = [step['budget'] for step in run['checkpoints'] if step['met']]
            for index, step in enumerate(run['checkpoints']):
                n = step['draws_per_chain']
                saved = counts[:, :n]
                if saved.shape[1] != n:
                    raise ValueError(f'Missing sample at checkpoint {step["budget"]} in {source}')
                trans_prefix = trans[:, :n]
                computed = diagnostics(saved, trans_prefix, metadata)
                passing = all(
                    value['rank_split_rhat'] is not None and value['rank_split_rhat'] < 1.01
                    and value['mcse'] is not None and value['mcse'] <= run['precision_goals'][key]
                    for key, value in computed.items()
                )
                batch_wall = max(chain['elapsed_at_prefix_s'][index] for chain in chains)
                if passing != step['met'] or batch_wall != step['batch_wall_s']:
                    raise ValueError(f'{source}: inconsistent checkpoint qualification or timing')
                for key, value in step['observables'].items():
                    for metric in ('mean', 'mcse', 'rank_split_rhat', 'ess'):
                        actual, stored = computed[key].get(metric), value.get(metric)
                        if actual is None or stored is None:
                            matches = actual is stored
                        else:
                            matches = np.isclose(actual, stored, rtol=1e-9, atol=1e-12)
                        if not matches:
                            raise ValueError(f'{source}: {key} {metric} differs from saved draws')

                    def group_mean(group):
                        mask = groups == group
                        if key == 'joint_probability':
                            return np.all(saved[mask] >= metadata['joint_threshold'], axis=2).mean()
                        if key == 'trans_fraction':
                            total = saved[mask, :, 0].sum()
                            return trans_prefix[mask].sum() / total if total else None
                        return saved[mask, :, int(key.split('_')[1])].mean()

                    yield {
                        'machine': machine,
                        'case': case,
                        'engine': run.get('engine', 'OpenGCMC'),
                        'mode': run['mode'],
                        'budget_unit': 'cycle' if raspa else 'attempt',
                        'warmup': run['warmup_cycles'] if raspa else run['warmup'],
                        'production_per_chain': step['budget'],
                        'draws_per_chain': n,
                        'batch_wall_s': step['batch_wall_s'],
                        'checkpoint_passed': step['met'],
                        'first_pass': bool(valid_budgets and step['budget'] == valid_budgets[0]),
                        'right_censored_at_final_budget': not valid_budgets,
                        'observable': key,
                        'mean': value['mean'],
                        'mcse': value['mcse'],
                        'mcse_goal': run['precision_goals'][key],
                        'rank_split_rhat': value['rank_split_rhat'],
                        'ess': value.get('ess'),
                        'mean_from_empty': group_mean('empty'),
                        'mean_from_preloaded': group_mean('preloaded'),
                        'source': source.relative_to(study),
                    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    rows = list(records(args.study))
    with args.output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f'Verified and exported {len(rows)} checkpoint observables.')
