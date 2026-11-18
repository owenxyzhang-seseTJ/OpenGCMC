#!/usr/bin/env python3
"""Validate and time the required FP64 CUDA tensor plug-in on a real GPU.

Default: native trial batches and delayed full-loop comparisons, including copies and
synchronization. --input reuses a real PoreMC JSON for the paired production loop.
A missing device is recorded and gives a nonzero exit status. All timing is in seconds.
"""
import argparse
import copy
import ctypes as C
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from poremc.fast import FastGCMC
from tools.longrange.fixtures import workload_fixture,make_potential,simulation_arguments


def primitive_checks(plugin):
    """Exercise actual plug-in batch/structure ABI against direct complex exponentials."""
    lib=C.CDLL(str(plugin))
    d=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    i=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
    lib.poremc_ewald_cuda_create.argtypes=[C.c_size_t,d,d,C.c_int]
    lib.poremc_ewald_cuda_create.restype=C.c_void_p
    lib.poremc_ewald_cuda_destroy.argtypes=[C.c_void_p]
    lib.poremc_ewald_cuda_set_lattice.argtypes=[C.c_void_p,d,i,i]
    lib.poremc_ewald_cuda_last_error.restype=C.c_char_p
    lib.poremc_ewald_cuda_device_name.argtypes=[C.c_void_p]
    lib.poremc_ewald_cuda_device_name.restype=C.c_char_p
    lib.poremc_ewald_cuda_structure.argtypes=[C.c_void_p,C.c_size_t,d,d,d]
    lib.poremc_ewald_cuda_batch.argtypes=[C.c_void_p,C.c_size_t,C.c_size_t,d,d,d,C.c_double,C.c_double,C.c_double,d]
    rng=np.random.default_rng(7102)
    cell=np.array([[37.,0.,0.],[4.,33.,0.],[-7.,3.,41.]])
    inverse=np.ascontiguousarray(np.linalg.inv(cell))
    records=[]
    for limits,n,b,half,sparse in [((13,13,13),12,10,True,False),((2,2,24),35,3,False,True),((0,0,1),1,2,False,False)]:
        modes=np.array([(x,y,z) for x in range(-limits[0],limits[0]+1) for y in range(-limits[1],limits[1]+1)
                        for z in range(-limits[2],limits[2]+1) if (x,y,z)!=(0,0,0) and
                        (not half or (x,y,z)>(0,0,0)) and (not sparse or (x+y+z)%11!=0)],dtype=np.int32)
        wave=np.ascontiguousarray(2*np.pi*modes @ inverse.T)
        weights=np.exp(-.5*np.sum(wave**2,axis=1))/(1+np.sum(wave**2,axis=1))
        g=lib.poremc_ewald_cuda_create(len(modes),wave,weights,int(os.environ.get('POREMC_CUDA_DEVICE','0')))
        if not g:raise RuntimeError(lib.poremc_ewald_cuda_last_error().decode())
        try:
            def check(status):
                if status:raise RuntimeError(lib.poremc_ewald_cuda_last_error().decode())
            check(lib.poremc_ewald_cuda_set_lattice(g,inverse,np.asarray(limits,dtype=np.int32),modes))
            xyz=np.ascontiguousarray(rng.uniform(-1.2,2.2,(b,n,3))@cell)
            charges=rng.uniform(-.35,.35,n)
            if n>1:charges[::5]=0.
            sf=(np.exp(1j*(xyz@wave.T))*charges[None,:,None]).sum(axis=1)
            got_sf=np.empty(2*len(modes));check(lib.poremc_ewald_cuda_structure(g,n,xyz[0],charges,got_sf))
            sf_error=float(np.max(np.abs(got_sf.view(np.complex128)-sf[0])))
            if sf_error>2e-11:raise AssertionError(sf_error)
            max_error=0.
            for baseq in (0.,-.27,.19):
                base=np.ascontiguousarray(rng.normal(size=len(modes))+1j*rng.normal(size=len(modes)))
                self_coeff,bg_coeff=-13.,-.15
                reference=((2*(base.conj()*sf).real+abs(sf)**2)*weights).sum(axis=1)
                reference+=self_coeff*np.dot(charges,charges)+bg_coeff*(2*baseq*sum(charges)+sum(charges)**2)
                got=np.empty(b)
                check(lib.poremc_ewald_cuda_batch(g,b,n,xyz,charges,base.view(np.float64),baseq,self_coeff,bg_coeff,got))
                np.testing.assert_allclose(got,reference,rtol=2e-11,atol=2e-9)
                max_error=max(max_error,float(np.max(np.abs(got-reference))))
            records.append(dict(kmax=limits,atoms=n,batches=b,retained_modes=len(modes),half=half,sparse=sparse,
                                max_energy_error_K=max_error,max_structure_error=sf_error))
            device=lib.poremc_ewald_cuda_device_name(g).decode()
        finally:lib.poremc_ewald_cuda_destroy(g)
    return {'device':device,'cases':records}


def fixture_engine(backend,mode,plugin,seed,kmax=13,host_atoms=4416):
    m,physics,entries=workload_fixture(host_atoms=host_atoms,kmax=kmax)
    potential=make_potential(m,physics)
    sim=FastGCMC(m,potential,potential.cell,physics['temperature'],[physics['activity']],seed=seed,
        reciprocal_backend=backend,cuda_library=str(plugin),sampling_mode=mode,host_grid_mesh=64,
        **simulation_arguments(physics))
    for xyz in entries:sim.add_particle(0,xyz)
    return sim


def device_probe_timings(plugin,repeats):
    rng=np.random.default_rng(7009);m,_,entries=workload_fixture()
    engines={}
    try:
        for backend in ('cpu','cuda','cuda-tensor'):
            engines[backend]=fixture_engine(backend,'exact',plugin,7031)
        rows=[]
        for b in (1,10,64):
            coords=np.ascontiguousarray(entries[0][None,:,:]+rng.normal(0.,.08,size=(b,len(m.charge),3)))
            energies={key:sim.trial_molecule_energies(coords,exclude=0) for key,sim in engines.items()}
            for key,numbers in energies.items():np.testing.assert_allclose(numbers,energies['cpu'],rtol=5e-11,atol=5e-8)
            elapsed={key:[] for key in engines}
            for repeat in range(repeats):
                for key in (list(engines) if repeat%2==0 else reversed(engines)):
                    start=time.perf_counter();engines[key].trial_molecule_energies(coords,exclude=0)
                    elapsed[key].append(time.perf_counter()-start)
            rows.append(dict(batch=b,seconds=elapsed,median_call_s={k:statistics.median(v) for k,v in elapsed.items()},
                             max_abs_tensor_cpu_K=float(abs(energies['cuda-tensor']-energies['cpu']).max())))
        return rows
    finally:
        for sim in engines.values():sim.close()


def production(plugin,args):
    from poremc.cli import construct_case
    rows=[]
    for r in range(args.repeats):
        for backend in (('cpu','cuda-tensor') if r%2==0 else ('cuda-tensor','cpu')):
            seed=7417+r
            if args.input:
                original=json.loads(args.input.read_text());cfg=copy.deepcopy(original)
                cfg.setdefault('potential',{}).update(backend='native',threads=1)
                cfg.setdefault('sampling',{}).update(driver='cpp',sampling_mode='delayed',reciprocal_method='direct',
                    reciprocal_backend=backend,cuda_library=str(plugin),seed=seed)
                sim=construct_case(cfg,args.input.resolve().parent)[0]
            else:sim=fixture_engine(backend,'delayed',plugin,seed)
            with sim:
                sim.run(args.burnin,sample_every=max(1,args.burnin),record_energy=False)
                before=sim.long_range_profile['gpu_selected_structures']
                result=sim.run(args.steps,sample_every=10,record_energy=False)
                profile=sim.long_range_profile
                if backend=='cuda-tensor' and not profile['gpu_selected_structures']>before:
                    raise RuntimeError('No selected GPU structures ran in production; increase steps or enable insertion, rotation, regrow.')
                # Rebuild accepted-energy state on CPU using the same target; initialization is outside timing.
                entries=sim.entries();final_energy=sim.total_energy()
                if args.input:
                    cfg['sampling']['reciprocal_backend']='cpu';cfg.pop('initial_particles',None)
                    audit=construct_case(cfg,args.input.resolve().parent)[0]
                else:
                    m,ph,_=workload_fixture();pot=make_potential(m,ph)
                    audit=FastGCMC(m,pot,pot.cell,ph['temperature'],[ph['activity']])
                with audit:
                    for species,coords in entries:audit.add_particle(species,coords)
                    error=abs(audit.total_energy()-final_energy)
                if error>1e-5:raise AssertionError(f'Final CPU rebuild disagrees by {error} K')
                rows.append(dict(seed=seed,backend=backend,attempts_per_s=result['attempted_moves_per_second'],
                    mean_count=float(np.mean(result['counts'])),acceptance=result['acceptance'],profile=profile,
                    production_selected_gpu=profile['gpu_selected_structures']-before,final_cpu_rebuild_error_K=error))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plugin',type=Path,default=ROOT/'poremc/libporemc_ewald_cuda.so')
    parser.add_argument('--input',type=Path,help='optional complete real PoreMC input for production pairs')
    parser.add_argument('--steps',type=int,default=3000);parser.add_argument('--burnin',type=int,default=500)
    parser.add_argument('--repeats',type=int,default=3);parser.add_argument('--probe-repeats',type=int,default=10)
    parser.add_argument('--output',type=Path,default=Path('tensor_gpu_results.json'))
    args=parser.parse_args();plugin=args.plugin.resolve();args.output.parent.mkdir(parents=True,exist_ok=True)
    evidence={'gpu_executed':False,'timing_scope':'Public native probes and complete single-chain loops, including CPU/device transfers and synchronization.'}
    if args.input:evidence['input_name']=args.input.name
    try:
        os.environ['POREMC_CUDA_KERNEL']='direct' # pin the legacy comparator; explicit cuda-tensor overrides this
        evidence['plugin']=plugin.name
        evidence['primitive']=primitive_checks(plugin)
        if 'test double' in evidence['primitive']['device']:
            raise RuntimeError('The real-device runner requires a physical CUDA device.')
        evidence['gpu_executed']=True
        evidence['batch_timings']=device_probe_timings(plugin,args.probe_repeats)
        evidence['delayed_production']=production(plugin,args)
        evidence['status']='passed'
    except Exception as exc:
        evidence.update(status='failed',diagnostic=str(exc));raise
    finally:args.output.write_text(json.dumps(evidence,indent=2)+'\n')


if __name__=='__main__':main()
