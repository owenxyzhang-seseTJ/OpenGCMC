from pathlib import Path

import numpy as np
import pytest

from poremc.cli import construct_case
from poremc.scheduling import RunStore, _read, run_block


def _ideal():
    return {
        'temperature_K':298.,'cell_A':[20,20,20],
        'species':[{'name':m,'sigma':[0.],'epsilon':[0.]} for m in ('A','B')],
        'potential':{'kind':'classical','cutoff_A':8.},
        'reservoir':{'pressure_Pa':1e6,'mole_fractions':[.4,.6]},
        'sampling':{'driver':'cpp','steps':1200,'burnin':150,'sample_every':10,'trials':1,
                    'record_energy':False}
    }


def test_scheduler_resumes_committed_blocks_without_duplicate_samples(tmp_path):
    store=RunStore(tmp_path)
    case=_ideal()
    plan=store.plan(case,pressure_pa=[2e5,4e5,6e5,8e5,1e6,1.2e6,1.4e6,1.6e6],
                    replicas=4,block_steps=400)
    first=store.submit(plan['plan_id'],81531,'request')
    run=first['run_id']
    assert not store.submit(plan['plan_id'],81531,'request')['created']
    with pytest.raises(ValueError,match='different calculation'):
        store.submit(plan['plan_id'],81532,'request')
    old=store.claim(run,lease_seconds=0)
    new=store.claim(run)
    assert new['generation']!=old['generation']
    run_block(new)
    run_block(old)
    assert store.finish(new) and not store.finish(old) and not store.finish(new)
    in_flight=store.claim(run)
    assert store.cancel(run)['status']=='cancelled'
    assert store.claim(run) is None
    run_block(in_flight)
    assert not store.finish(in_flight)
    store.resume(run)
    assert store.run_local(run,workers=4)['status']=='completed'
    summary=store.result(run)
    assert summary['chains']=={'completed':32}
    assert len(summary['artifacts'])==summary['blocks']==96
    assert summary['latest_checkpoint_step']==1350
    rows=[_read(p) for p in summary['artifacts'][:3]]
    assert rows[0]['sample_steps']+rows[1]['sample_steps']+rows[2]['sample_steps']==list(range(160,1351,10))
    case['sampling']['seed']=new['seed']
    case['reservoir']['pressure_Pa']=2e5
    sim,options,*_=construct_case(case)
    with sim:
        direct=sim.run(**options)
    np.testing.assert_array_equal(sum((r['counts'] for r in rows),[]),direct['counts'])


def test_slurm_array_records_the_same_durable_blocks(tmp_path,monkeypatch):
    from poremc import slurm

    store=RunStore(tmp_path/'runs')
    plan=store.plan(_ideal(),replicas=2,block_steps=600)
    run=store.submit(plan['plan_id'],71,'slurm_test')['run_id']
    submissions=[]

    def fake_slurm(*command):
        assert command[0]=='sbatch'
        import shlex
        job_name=next(x.split('=',1)[1] for x in command if x.startswith('--job-name='))
        manifest=Path(shlex.split(command[-1])[-1])
        wave=_read(manifest)
        assert wave['name']==job_name
        assert '--array=0-1%2' in command
        if not submissions:
            # Lose the command response after Slurm accepted the array.
            for i in range(2):
                monkeypatch.setenv('SLURM_ARRAY_TASK_ID',str(i))
                slurm.run_array_item(manifest)
            submissions.append('3210')
            raise RuntimeError('lost Slurm receipt')
        for i in range(2):
            monkeypatch.setenv('SLURM_ARRAY_TASK_ID',str(i))
            slurm.run_array_item(manifest)
        submissions.append('3211')
        return '3211'

    monkeypatch.setattr(slurm,'_command',fake_slurm)
    monkeypatch.setattr(slurm,'_known_job',lambda _: '3210')
    with pytest.raises(RuntimeError,match='lost Slurm receipt'):
        slurm.run_slurm(store,run,workers=2)
    assert slurm.run_slurm(store,run,workers=2)['status']=='completed'
    assert store.result(run)['blocks']==4
    assert submissions==['3210','3211']


def test_count_window_checkpoint_and_reweighting(tmp_path):
    case=_ideal()
    case['potential']={'kind':'zero'}
    case['reservoir']={'activities_A3_inverse':[2/8000,0.]}
    case['sampling'].update(driver='python',steps=16000,burnin=400)
    store=RunStore(tmp_path)
    windows=[{'coefficients':[1,0],'center':x,'stiffness':.3} for x in (1.,4.)]
    plan=store.plan(case,replicas=2,block_steps=8000,count_windows=windows)
    run=store.submit(plan['plan_id'],906,'windows')['run_id']
    assert store.run_local(run,workers=2)['status']=='completed'
    result=store.result(run)['free_energy_kT']
    assert result['status']=='point_estimate' and result['overlap_gap']>.01
    mean=np.asarray(result['coordinate'])@np.asarray(result['probability'])
    assert mean==pytest.approx(2.,abs=.15)


def test_mcp_stdio_submits_and_reads_mixture_results(tmp_path):
    import asyncio
    import sys

    pytest.importorskip('mcp')
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def scenario():
        server=StdioServerParameters(command=sys.executable,args=['-m','poremc.mcp','--root',
                                   str(tmp_path),'serve','--workers','2'])
        async with stdio_client(server) as (read,write), ClientSession(read,write) as session:
            await session.initialize()
            tools={tool.name for tool in (await session.list_tools()).tools}
            assert {'opengcmc_submit','opengcmc_resume','opengcmc_cancel'} <= tools

            async def call(name,**args):
                result=await session.call_tool(f'opengcmc_{name}',args)
                assert not result.isError,result.content
                return result.structuredContent

            plan=await call('plan',config=_ideal(),replicas=2,block_steps=600)
            submitted=await call('submit',plan_id=plan['plan_id'],seed=935,idempotency_key='stdio')
            run=submitted['run_id']
            while True:
                progress=await call('status',run_id=run)
                if progress['status']=='completed':
                    break
                assert progress['status'] in ('queued','running'),progress
                await asyncio.sleep(.05)
            result=await call('result',run_id=run)
            assert result['blocks']==4 and len(result['points'])==1
            assert set(result['points'][0]['counts_per_cell'])=={'A','B'}

    asyncio.run(asyncio.wait_for(scenario(),timeout=30))
