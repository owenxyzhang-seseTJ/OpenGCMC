"""Batch independent trial requests against identical immutable snapshots.

This batches energy evaluations, not state commits. Each chain makes its own
accept/reject decision after the returned energies; no concurrent conflicting
moves are accepted using a stale shared state.
"""
from dataclasses import dataclass
import numpy as np

@dataclass
class TrialRequest:
    environment: object
    coordinates: np.ndarray
    sigma: np.ndarray
    epsilon: np.ndarray
    charge: np.ndarray

def evaluate_requests(requests,max_batch_atoms=1_000_000):
    """Coalesce matching host/parameter requests, preserving original ordering."""
    if max_batch_atoms<1:raise ValueError('positive batch capacity required')
    requests=list(requests);groups={};result=[None]*len(requests)
    for index,r in enumerate(requests):
        x=np.asarray(r.coordinates,float)
        if x.ndim!=3 or x.shape[-1]!=3:raise ValueError('coordinates must have shape [B,A,3]')
        params=tuple(np.ascontiguousarray(v,dtype=np.float64) for v in (r.sigma,r.epsilon,r.charge))
        if any(v.shape!=(x.shape[1],) for v in params):raise ValueError('parameter shape mismatch')
        key=(id(r.environment),x.shape[1],*(v.tobytes() for v in params))
        groups.setdefault(key,[]).append((index,r.environment,x,params))
    for items in groups.values():
        chunk=[];atoms=0
        def flush():
            if not chunk:return
            x=np.concatenate([r[2] for r in chunk],axis=0)
            values=chunk[0][1].energy(x,*chunk[0][3]);offset=0
            for index,_,coords,_ in chunk:
                n=len(coords);result[index]=values[offset:offset+n].copy();offset+=n
        for item in items:
            n=int(np.prod(item[2].shape[:2]))
            if chunk and atoms+n>max_batch_atoms:flush();chunk=[];atoms=0
            chunk.append(item);atoms+=n
        flush()
    return result

def run_native_replicas(species,potential,cell,temperature,activities,seeds,
                        run_options,simulation_options=None,max_workers=None):
    """Independent C++ chains share one immutable host cache; ctypes releases GIL.

    Each chain owns its guest environment, RNG and counters. Internal OpenMP
    threads should normally be 1 when parallelizing across replicas.
    """
    from concurrent.futures import ThreadPoolExecutor
    import time
    from .fast import FastGCMC
    seeds=[int(s) for s in seeds]
    if not seeds or len(set(seeds))!=len(seeds):raise ValueError('distinct nonempty seeds required')
    options=dict(simulation_options or {})
    if 'seed' in options:raise ValueError('specify seeds through seeds, not simulation_options')
    def one(seed):
        with FastGCMC(species,potential,cell,temperature,activities,seed=seed,**options) as sim:
            result=sim.run(**run_options);result['seed']=seed;result['cache_profile']=sim.profile
            return result
    start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:results=list(pool.map(one,seeds))
    elapsed=time.perf_counter()-start
    return {'replicas':results,'wall_seconds':elapsed,'production_attempts':sum(r['steps'] for r in results),
            'production_attempts_per_total_wall_second':sum(r['steps'] for r in results)/elapsed,
            'shared_immutable_host':True}
