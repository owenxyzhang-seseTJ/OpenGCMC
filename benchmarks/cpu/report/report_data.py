"""Read archived OpenGCMC benchmark records used in the experiment report."""

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
STUDY = HERE.parent / 'paired'


def read(relative):
    return json.loads((STUDY / relative).read_text())


def real_run(platform, group, name):
    relative = Path(platform) / (group if platform == 'hpc' else '')
    return read(relative / f'{name}.json')


def timing(run):
    first = run['qualified_wall_s']
    return (first, False) if first is not None else (run['checkpoints'][-1]['batch_wall_s'], True)


def seconds(run):
    elapsed, censored = timing(run)
    return f'{"＞" if censored else ""}{elapsed:.2f}'


def checkpoint(run, first=False):
    return next(x for x in run['checkpoints'] if x['met']) if first else run['checkpoints'][-1]


# Display order, storage group, OpenGCMC record, RASPA3 record.
REAL_CASES = (
    ('IRMOF-1／CO₂+CH₄', 'irmof_mixture', 'irmof_mixture-delayed', 'irmof_mixture-raspa-cbmc'),
    ('TON／正丁烷 2 Pa', 'ton_butane_dilute', 'ton_butane_dilute-delayed', 'ton_butane_dilute-raspa-cbmc'),
    ('TON／正丁烷 100 Pa', 'ton_butane', 'ton_butane_harmonic-delayed', 'ton_butane_harmonic-raspa-cbmc'),
    ('对二甲苯 0.02 Pa', 'xylene_low', 'xylene_low-delayed-long', 'xylene_low-raspa-cbmc'),
    ('对二甲苯 0.50 Pa', 'xylene_mid', 'xylene_mid-delayed', 'xylene_mid-raspa-cbmc'),
    ('对二甲苯 40 Pa', 'xylene_high', 'xylene_high-delayed', 'xylene_high-raspa-cbmc'),
)


def timing_runs(platform, case):
    _, group, native, raspa = case
    native_group = 'xylene_low_confirm' if group == 'xylene_low' else group
    return real_run(platform, native_group, native), real_run(platform, group, raspa)
