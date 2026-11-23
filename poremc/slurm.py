"""Slurm array execution for OpenGCMC sample blocks."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import uuid

from .scheduling import _integer, _publish, _read, run_block


def run_array_item(manifest):
    wave = _read(manifest)
    index = int(os.environ['SLURM_ARRAY_TASK_ID'])
    if not 0 <= index < len(wave['tickets']):
        raise ValueError('array index is outside the manifest')
    ticket = wave['tickets'][index]
    outcome = Path(ticket['attempt'])/'outcome.json'
    try:
        run_block(ticket)
    except Exception as exc:
        _publish(outcome, {'error': f'{type(exc).__name__}: {exc}'})
        raise
    _publish(outcome, {'error': None})


def _command(*args):
    return subprocess.run(args, check=True, text=True, capture_output=True).stdout.strip()


def _known_job(name):
    active = _command('squeue','-h','-n',name,'-o','%A').splitlines()
    history = _command('sacct','-n','-X','--name',name,'-o','JobIDRaw','-P').splitlines()
    bases = (row.strip().rstrip('|').split('_')[0] for row in active+history)
    matches = {base for base in bases if base.isdigit()}
    if len(matches)>1:
        raise RuntimeError(f'ambiguous Slurm submission for {name}')
    return next(iter(matches),None)


def _reconcile(store, manifest, poll_seconds):
    wave = _read(manifest)
    identifier = manifest.with_name('job.json')
    job = _read(identifier)['job'] if identifier.exists() else _known_job(wave['name'])
    if job is None:
        raise RuntimeError(f'Slurm has not confirmed submission of {wave["name"]}')
    if not identifier.exists():
        _publish(identifier, {'job': job})
    unresolved = set(range(len(wave['tickets'])))
    while unresolved:
        if store.status(wave['tickets'][0]['run'])['status']=='cancelled':
            _command('scancel',job)
            return
        for i in sorted(unresolved):
            ticket = wave['tickets'][i]
            outcome = Path(ticket['attempt'])/'outcome.json'
            if outcome.exists():
                store.finish(ticket,error=_read(outcome)['error'])
                unresolved.remove(i)
        if unresolved:
            terminal = {'BOOT_FAIL','CANCELLED','COMPLETED','DEADLINE','FAILED','NODE_FAIL',
                        'OUT_OF_MEMORY','PREEMPTED','REVOKED','TIMEOUT'}
            for row in _command('sacct','-n','-X','-j',job,'-o','JobID%50,State','-P').splitlines():
                fields = row.strip().split('|')
                if len(fields)<2 or not fields[1].strip():
                    continue
                item,state = fields[:2]
                if not item.startswith(f'{job}_') or not item.split('_')[1].isdigit():
                    continue
                i = int(item.split('_')[1])
                state = state.split()[0].rstrip('+')
                if i in unresolved and state in terminal:
                    # A marker can appear between the file check and accounting query.
                    ticket = wave['tickets'][i]
                    outcome = Path(ticket['attempt'])/'outcome.json'
                    error = _read(outcome)['error'] if outcome.exists() else f'Slurm {state} without a committed block'
                    store.finish(ticket,error=error)
                    unresolved.remove(i)
        if unresolved:
            time.sleep(poll_seconds)


def run_slurm(store, run, workers=1, wave_size=128, poll_seconds=10, partition=None):
    """Run Slurm arrays from a controller with a ledger and a shared run directory."""
    import fcntl

    _integer(workers,'workers')
    _integer(wave_size,'wave_size')
    _integer(poll_seconds,'poll_seconds')
    store.status(run)
    directory = store.root/'runs'/run/'slurm'
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        waves = sorted(directory.glob('*/manifest.json'))
        known = {p.parent.name for p in waves}
        with store._db(True) as db:
            rows = db.execute("SELECT DISTINCT owner FROM chains WHERE run=? AND owner LIKE 'opengcmc-%'",
                              (run,)).fetchall()
            for row in rows:
                if row['owner'] not in known:
                    db.execute("UPDATE chains SET status='queued',owner=NULL,lease=0 WHERE run=? AND owner=?",
                               (run,row['owner']))
        for wave in waves:
            if not wave.with_name('reconciled.json').exists():
                _reconcile(store,wave,poll_seconds)
                _publish(wave.with_name('reconciled.json'),{})
        while store.status(run)['status']!='cancelled':
            name = 'opengcmc-'+uuid.uuid4().hex[:20]
            tickets = []
            for _ in range(wave_size):
                ticket = store.claim(run,owner=name)
                if ticket is None:
                    break
                tickets.append(ticket)
            if not tickets:
                break
            wave_dir = directory/name
            manifest = wave_dir/'manifest.json'
            _publish(manifest,{'name':name,'tickets':tickets})
            cfg = tickets[0]['plan']['config']
            threads = _integer(cfg.get('potential',{}).get('threads',1),'threads')
            args = ['--parsable',f'--job-name={name}',f'--array=0-{len(tickets)-1}%{workers}',
                    f'--cpus-per-task={threads}',f'--output={wave_dir}/%A_%a.log']
            if partition:
                args.append(f'--partition={partition}')
            if cfg['sampling'].get('reciprocal_backend') in ('cuda','cuda-tensor','cuda-resident'):
                args.append('--gres=gpu:1')
            wrap = shlex.join([sys.executable,'-m','poremc.mcp','array-item',str(manifest)])
            job = _command('sbatch',*args,'--wrap',wrap).split(';')[0]
            if not job.isdigit():
                raise RuntimeError('Slurm returned an invalid allocation number')
            _publish(manifest.with_name('job.json'),{'job':job})
            _reconcile(store,manifest,poll_seconds)
            _publish(manifest.with_name('reconciled.json'),{})
    return store.status(run)
