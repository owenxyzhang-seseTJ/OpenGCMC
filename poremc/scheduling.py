"""Durable molecular simulation plans, independent chains and sample blocks."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import uuid

import numpy as np

from .cache import content_key
from .cli import clean_json, construct_case
from .statistics import chain_mean_interval
from .units import uptake_mmol_g


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _publish(path, data):
    """Make a complete artifact visible before exposing it in the job ledger."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(_encode(clean_json(data)))
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
        if os.name == 'posix':
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _input_snapshot(config, base_directory):
    config = copy.deepcopy(config)
    base = Path(base_directory).resolve()

    def input_file(relative):
        path = (base / relative).resolve()
        if not path.is_relative_to(base):
            raise ValueError('planned model files must be inside the input directory')
        return path

    config['species'] = [_read(input_file(m['file'])) if set(m) == {'file'} else m
                         for m in config['species']]
    host = config.get('host', {})
    cif = None
    if 'cif' in host:
        cif = input_file(host['cif']).read_text(encoding='utf-8')
        host['cif'] = 'framework.cif'
        if isinstance(host.get('force_field'), str):
            host['force_field'] = _read(input_file(host['force_field']))
    sampling = config.setdefault('sampling', {})
    if sampling.get('cuda_library'):
        raise ValueError('scheduled runs use the installed reciprocal backend')
    for key in ('host_grid_cache', 'host_field_cache'):
        sampling.pop(key, None)
    if config.get('root_bias'):
        config['root_bias'].pop('cache_dir', None)
    return config, cif


def _integer(value, name, minimum=1, maximum=2**63-1):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f'{name} must be an integer in [{minimum}, {maximum}]')
    return value


def make_plan(config, *, base_directory='.', pressure_pa=None, reservoirs=None,
              count_windows=None,replicas=4, block_steps=10000, count_mcse=None, log_selectivity_mcse=None):
    """Freeze molecular input and a fixed production budget for each reservoir."""
    config, cif = _input_snapshot(config, base_directory)
    sampling = config['sampling']
    if count_windows is not None:
        if pressure_pa is not None or reservoirs is not None or 'count_bias' in sampling:
            raise ValueError('count windows use one reservoir and supply their own biases')
        if sampling.get('driver','python')!='python':
            raise ValueError('count windows require the Python GCMC driver')
        from .free_energy import HarmonicCountBias
        if not isinstance(count_windows,list) or len(count_windows)<2:
            raise ValueError('provide at least two count windows')
        biases=[HarmonicCountBias(**x) for x in count_windows]
        coefficient=np.asarray(biases[0].coefficients)
        if (coefficient.shape!=(len(config['species']),) or not coefficient.any()
                or any(not np.array_equal(b.coefficients,coefficient) for b in biases)):
            raise ValueError('all windows must share one nonzero count coordinate')
        reservoirs=[config['reservoir'] for _ in biases]
    if pressure_pa is not None and reservoirs is not None:
        raise ValueError('give pressure_pa or reservoirs')
    if pressure_pa is not None:
        reservoir = config['reservoir']
        if 'mole_fractions' not in reservoir:
            raise ValueError('pressure points require fixed reservoir mole fractions')
        reservoirs = [{**reservoir, 'pressure_Pa': p} for p in pressure_pa]
    reservoirs = [config['reservoir']] if reservoirs is None else copy.deepcopy(reservoirs)
    if not reservoirs or not all(isinstance(r, dict) for r in reservoirs):
        raise ValueError('provide at least one reservoir')
    replicas = _integer(replicas, 'replicas')
    steps = _integer(sampling.get('steps', 10000), 'steps')
    burnin = _integer(sampling.get('burnin', 1000), 'burnin', 0)
    every = _integer(sampling.get('sample_every', 10), 'sample_every')
    block_steps = min(steps, _integer(block_steps, 'block_steps'))
    if steps % every or block_steps % every:
        raise ValueError('steps and block_steps must be multiples of sample_every')
    for goal in (count_mcse, log_selectivity_mcse):
        if goal is not None and (not math.isfinite(goal) or goal <= 0):
            raise ValueError('precision targets must be positive and finite')
    if log_selectivity_mcse is not None and (
        len(config['species']) != 2
        or any(len(r.get('mole_fractions', []))!=2 or min(r['mole_fractions'])<=0 for r in reservoirs)
    ):
        raise ValueError('a selectivity target requires two positive bulk mole fractions')
    sampling.update(steps=steps, burnin=burnin, sample_every=every)
    return {'config': config, 'framework': cif, 'reservoirs': reservoirs, 'replicas': replicas,
            'block_steps': block_steps, 'count_mcse': count_mcse,
            'log_selectivity_mcse': log_selectivity_mcse,'count_windows':count_windows}


class RunStore:
    """Coordinates attempts and publishes one result for each logical block."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / 'runs.sqlite3'
        with self._db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS plans(id TEXT PRIMARY KEY, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS runs(
                    id TEXT PRIMARY KEY, idempotency_key TEXT UNIQUE NOT NULL,
                    plan_id TEXT NOT NULL REFERENCES plans(id), seed TEXT NOT NULL,
                    created REAL NOT NULL, cancelled INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS chains(
                    run TEXT NOT NULL REFERENCES runs(id), point INTEGER NOT NULL,
                    replica INTEGER NOT NULL, seed TEXT NOT NULL, next_block INTEGER NOT NULL DEFAULT 0,
                    generation INTEGER NOT NULL DEFAULT 0, lease REAL NOT NULL DEFAULT 0, owner TEXT,
                    status TEXT NOT NULL DEFAULT 'queued', error TEXT,
                    PRIMARY KEY (run,point,replica)
                );
                CREATE TABLE IF NOT EXISTS blocks(
                    run TEXT NOT NULL, point INTEGER NOT NULL, replica INTEGER NOT NULL,
                    block INTEGER NOT NULL, path TEXT NOT NULL, last_step INTEGER NOT NULL,
                    committed REAL NOT NULL,
                    PRIMARY KEY (run,point,replica,block),
                    FOREIGN KEY (run,point,replica) REFERENCES chains(run,point,replica)
                );
            """)

    @contextmanager
    def _db(self, write=False):
        db = sqlite3.connect(self.database, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            with db:
                if write:
                    db.execute('BEGIN IMMEDIATE')
                yield db
        finally:
            db.close()

    def plan(self, config, **options):
        plan = make_plan(config, **options)
        plan_id = content_key({'plan': plan}, {})
        with self._db(True) as db:
            db.execute('INSERT OR IGNORE INTO plans VALUES (?,?)', (plan_id, _encode(plan)))
        sample = plan['config']['sampling']
        chains = len(plan['reservoirs'])*plan['replicas']
        return {'plan_id': plan_id, 'points': len(plan['reservoirs']), 'chains': chains,
                'attempts': chains*(sample['burnin']+sample['steps']), 'block_steps': plan['block_steps']}

    def submit(self, plan_id, seed, idempotency_key):
        seed = _integer(seed, 'seed', 0, 2**64-1)
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise ValueError('a nonempty idempotency key is required')
        with self._db(True) as db:
            existing = db.execute('SELECT * FROM runs WHERE idempotency_key=?', (idempotency_key,)).fetchone()
            if existing:
                if (existing['plan_id'], existing['seed'])!=(plan_id, str(seed)):
                    raise ValueError('idempotency key already identifies a different calculation')
                return {'run_id': existing['id'], 'created': False}
            row = db.execute('SELECT body FROM plans WHERE id=?', (plan_id,)).fetchone()
            if row is None:
                raise ValueError('unknown plan')
            plan = json.loads(row['body'])
            run = uuid.uuid4().hex
            db.execute('INSERT INTO runs(id,idempotency_key,plan_id,seed,created) VALUES (?,?,?,?,?)',
                       (run,idempotency_key,plan_id,str(seed),time.time()))
            rows = []
            for point in range(len(plan['reservoirs'])):
                for replica in range(plan['replicas']):
                    chain_seed = np.random.SeedSequence([seed,point,replica]).generate_state(1,dtype=np.uint64)[0]
                    rows.append((run,point,replica,str(int(chain_seed))))
            db.executemany('INSERT INTO chains(run,point,replica,seed) VALUES (?,?,?,?)', rows)
        return {'run_id': run, 'created': True}

    @staticmethod
    def _run(db, run):
        row = db.execute('SELECT runs.*,plans.body FROM runs JOIN plans ON plan_id=plans.id WHERE runs.id=?',
                         (run,)).fetchone()
        if row is None:
            raise ValueError('unknown run')
        return row

    def claim(self, run, *, owner='local', lease_seconds=120):
        now = time.time()
        with self._db(True) as db:
            meta = self._run(db,run)
            if meta['cancelled']:
                return None
            chain = db.execute("""
                SELECT * FROM chains WHERE run=? AND (status='queued' OR
                    (status='running' AND owner='local' AND lease<=?))
                ORDER BY next_block,point,replica LIMIT 1
            """, (run, now)).fetchone()
            if chain is None:
                return None
            key = (run,chain['point'],chain['replica'])
            idx = chain['next_block']
            previous = db.execute('SELECT path FROM blocks WHERE run=? AND point=? AND replica=? AND block=?',
                                  (*key,idx-1)).fetchone() if idx else None
            if idx and previous is None:
                raise RuntimeError('missing committed predecessor block')
            token = chain['generation']+1
            db.execute("UPDATE chains SET status='running',generation=?,lease=?,owner=? WHERE run=? AND point=? AND replica=?",
                       (token,now+lease_seconds,owner,*key))
        attempt = self.root/'runs'/run/f"p{key[1]:04d}-r{key[2]:04d}"/f'b{idx:06d}-a{token:06d}'
        return {'run': run, 'point': key[1], 'replica': key[2], 'generation': token, 'block': idx,
                'seed': int(chain['seed']), 'plan': json.loads(meta['body']),
                'previous': previous['path'] if previous else None, 'attempt': str(attempt),
                'cache': str(self.root/'cache'/'fields')}

    @staticmethod
    def _ticket_key(ticket):
        return (ticket['run'],ticket['point'],ticket['replica'],ticket['generation'],ticket['block'])

    def extend(self,ticket,seconds):
        with self._db(True) as db:
            db.execute("""UPDATE chains SET lease=? WHERE run=? AND point=? AND replica=?
                          AND generation=? AND next_block=? AND status='running'""",
                       (time.time()+seconds,*self._ticket_key(ticket)))

    def finish(self,ticket,error=None):
        """Publish a block only while its lease generation owns the logical chain."""
        path = Path(ticket['attempt'])/'samples.json'
        block = ticket['block']
        plan = ticket['plan']
        sampling = plan['config']['sampling']
        start = min(block*plan['block_steps'],sampling['steps'])
        end = min(start+plan['block_steps'],sampling['steps'])
        last_step = sampling['burnin']+end
        if error is None:
            payload = _read(path)
            expected = list(range(sampling['burnin']+start+sampling['sample_every'],
                                  last_step+1,sampling['sample_every']))
            if payload['sample_steps']!=expected or payload['steps']!=end-start:
                raise ValueError('work block has incorrect global sample steps')
            if _read(Path(ticket['attempt'])/'checkpoint.json')['step_count']!=last_step:
                raise ValueError('work block checkpoint does not match sample steps')
        with self._db(True) as db:
            if self._run(db,ticket['run'])['cancelled']:
                return False
            key = self._ticket_key(ticket)
            row = db.execute("""SELECT status FROM chains WHERE run=? AND point=? AND replica=?
                                AND generation=? AND next_block=?""", key).fetchone()
            if row is None or row['status']!='running':
                return False
            if error is None:
                db.execute('INSERT INTO blocks VALUES (?,?,?,?,?,?,?)',
                           (*key[:3],block,str(path),last_step,time.time()))
                status = 'completed' if end==sampling['steps'] else 'queued'
                next_block = block+1
            else:
                status,next_block = 'failed',block
            db.execute("""UPDATE chains SET status=?,next_block=?,owner=NULL,lease=0,error=?
                          WHERE run=? AND point=? AND replica=?""", (status,next_block,error,*key[:3]))
        return True

    def cancel(self, run):
        with self._db(True) as db:
            self._run(db,run)
            db.execute('UPDATE runs SET cancelled=1 WHERE id=?', (run,))
        return self.status(run)

    def resume(self, run):
        with self._db(True) as db:
            row = self._run(db,run)
            db.execute('UPDATE runs SET cancelled=0 WHERE id=?', (run,))
            states = ('failed','running') if row['cancelled'] else ('failed',)
            for state in states:
                db.execute("""UPDATE chains SET status='queued',generation=generation+1,lease=0,
                              owner=NULL,error=NULL WHERE run=? AND status=?""", (run,state))
        return self.status(run)

    def status(self, run):
        with self._db() as db:
            meta = self._run(db,run)
            totals = dict(db.execute('SELECT status,count(*) FROM chains WHERE run=? GROUP BY status',(run,)))
            progress = db.execute('SELECT count(*),max(last_step) FROM blocks WHERE run=?',(run,)).fetchone()
            failures = [dict(r) for r in db.execute(
                'SELECT point,replica,error FROM chains WHERE run=? AND error IS NOT NULL',(run,))]
        if meta['cancelled']:
            state = 'cancelled'
        else:
            state = next((s for s in ('running','queued','failed','completed') if totals.get(s)), 'queued')
        return {'run_id':run,'status':state,'chains':totals,'blocks':progress[0],
                'latest_checkpoint_step':progress[1],'errors':failures}

    def run_local(self,run,workers=1):
        import fcntl

        _integer(workers,'workers')
        self.status(run)

        def loop():
            while True:
                ticket=self.claim(run)
                if ticket is None:
                    with self._db() as db:
                        live=db.execute("""SELECT 1 FROM chains WHERE run=? AND owner='local'
                                          AND status='running' LIMIT 1""",(run,)).fetchone()
                    if live and self.status(run)['status']!='cancelled':
                        time.sleep(.2)
                        continue
                    return
                stop = threading.Event()

                def heartbeat():
                    while not stop.wait(30):
                        self.extend(ticket,120)

                thread = threading.Thread(target=heartbeat,daemon=True)
                thread.start()
                try:
                    run_block(ticket)
                    self.finish(ticket)
                except Exception as exc:
                    self.finish(ticket, error=f'{type(exc).__name__}: {exc}')
                finally:
                    stop.set()
                    thread.join()

        directory=self.root/'runs'/run
        directory.mkdir(parents=True,exist_ok=True)
        with (directory/'local-controller.lock').open('a') as lock:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                return self.status(run)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(lambda _: loop(),range(workers)))
        return self.status(run)

    def result(self,run):
        status = self.status(run)
        if status['status']!='completed':
            return status
        with self._db() as db:
            meta = self._run(db,run)
            plan = json.loads(meta['body'])
            rows = list(db.execute('SELECT * FROM blocks WHERE run=? ORDER BY point,replica,block',(run,)))
        grouped = [[[] for _ in range(plan['replicas'])] for _ in plan['reservoirs']]
        for row in rows:
            grouped[row['point']][row['replica']].append(_read(row['path']))
        points = [_point_summary(reservoir,chains,plan) for reservoir,chains in zip(plan['reservoirs'],grouped)]
        summary={**status,'points':points,
                 'elapsed_since_submission_s':max(row['committed'] for row in rows)-meta['created'],
                 'artifacts':[row['path'] for row in rows]}
        if plan.get('count_windows'):
            for point,bias in zip(points,plan['count_windows']):
                point['count_bias']=bias
            summary['free_energy_kT']=_window_summary(plan['count_windows'],grouped)
        return clean_json(summary)


def run_block(ticket):
    """Compute one fixed-length block. Works on a worker without the run database."""
    plan = ticket['plan']
    config = copy.deepcopy(plan['config'])
    config['reservoir'] = plan['reservoirs'][ticket['point']]
    sampling = config['sampling']
    sampling['seed'] = ticket['seed']
    if plan.get('count_windows'):
        sampling['count_bias']=plan['count_windows'][ticket['point']]
    if sampling.get('driver')=='cpp' and sampling.get('sampling_mode')=='delayed':
        sampling['host_field_cache'] = ticket['cache']
    directory = Path(ticket['attempt'])
    directory.mkdir(parents=True,exist_ok=True)
    if plan['framework'] is not None:
        (directory/'framework.cif').write_text(plan['framework'],encoding='utf-8')
    predecessor = ticket['previous']
    checkpoint = _read(Path(predecessor).with_name('checkpoint.json')) if predecessor else None
    start = time.perf_counter()
    sim,options,host,identity,_ = construct_case(config,directory,checkpoint=checkpoint)
    setup = time.perf_counter()-start
    try:
        block = ticket['block']
        options['steps'] = min(plan['block_steps'],sampling['steps']-block*plan['block_steps'])
        if block:
            options.update(burnin=0,adapt=False)
        result = sim.run(**options)
        snapshot = sim.checkpoint()
        snapshot.update(count_trace=[],energy_trace=[],sample_steps=[],physics_id=identity)
        result.update(setup_seconds=setup,species_order=[m.name for m in sim.species],
                      framework_mass_amu=host.mass_amu)
        if hasattr(sim,'profile'):
            result['kernel_profile']=sim.profile
        _publish(directory/'checkpoint.json',snapshot)
        _publish(directory/'samples.json',result)
    finally:
        if hasattr(sim,'close'):
            sim.close()


def _point_summary(reservoir,chains,plan):
    first = chains[0][0]
    x = np.asarray([[row for b in chain for row in b['counts']] for chain in chains], dtype=float)
    names = first['species_order']
    metrics = {name:chain_mean_interval(x[:,:,i]) for i,name in enumerate(names)}
    for i,name in enumerate(names):
        metrics[name]['nonzero_draws'] = int(np.count_nonzero(x[:,:,i]))
    mass = first['framework_mass_amu']
    uptake = None if not mass else {name:uptake_mmol_g(m['mean'],mass) for name,m in metrics.items()}
    uptake_intervals = None if not mass else {
        name:[uptake_mmol_g(bound,mass) for bound in value['ci95']] for name,value in metrics.items()}
    selectivity = None
    fractions = reservoir.get('mole_fractions',[])
    if len(names)==len(fractions)==2 and min(fractions)>0:
        a,b=x.mean(axis=(0,1))
        if a>0 and b>0:
            log_s=math.log(a/b)-math.log(fractions[0]/fractions[1])
            # Delta method keeps the two counts from each draw together.
            result=chain_mean_interval(x[:,:,0]/a-x[:,:,1]/b+log_s)
            selectivity={'direction':names,'estimate':math.exp(log_s),'log':result}
            if all(math.isfinite(z) for z in result['ci95']):
                selectivity['ci95']=[math.exp(z) for z in result['ci95']]
    goals = []
    if plan['count_mcse'] is not None:
        goals.extend(m['mcse']<=plan['count_mcse'] and m['rank_split_rhat']<1.01 for m in metrics.values())
    if plan['log_selectivity_mcse'] is not None:
        goals.append(bool(selectivity and selectivity['log']['mcse']<=plan['log_selectivity_mcse']
                          and selectivity['log']['rank_split_rhat']<1.01))
    return {'reservoir':reservoir,'counts_per_cell':metrics,'uptake_mmol_g':uptake,
            'uptake_ci95_mmol_g':uptake_intervals,
            'selectivity':selectivity,'precision_met':bool(all(goals)) if goals else None,
            'preparation_seconds':sum(b['setup_seconds'] for chain in chains for b in chain),
            'production_seconds':sum(b['production_seconds'] for chain in chains for b in chain)}


def _window_summary(windows,grouped):
    from .free_energy import HarmonicCountBias, MBAR

    biases=[HarmonicCountBias(**window) for window in windows]
    samples=[np.asarray([row for chain in chains for b in chain for row in b['counts']])
             for chains in grouped]
    counts=np.concatenate(samples)
    reduced=np.stack([b.values(counts) for b in biases])
    try:
        fit=MBAR(reduced,[len(s) for s in samples])
    except (ValueError,RuntimeError) as exc:
        return {'status':'unresolved','reason':str(exc)}
    weights=np.exp(fit.log_weights(np.zeros(len(counts))))
    x,indices=np.unique(counts@np.asarray(biases[0].coefficients),return_inverse=True)
    mass=np.bincount(indices,weights=weights)
    positive=mass>0
    f=-np.log(mass[positive]/mass[positive].max())
    return {'status':'point_estimate','coordinate':x[positive].tolist(),'relative_free_energy':f.tolist(),
            'probability':mass[positive].tolist(),'overlap':fit.overlap.tolist(),
            'overlap_gap':fit.spectral_gap,'kish_ess':float(1/(weights@weights))}
