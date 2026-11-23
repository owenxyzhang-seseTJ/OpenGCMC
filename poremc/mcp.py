"""MCP stdio tools for planning, executing and inspecting OpenGCMC runs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from .scheduling import RunStore, _integer, _read


def _launch(store,run,workers,backend):
    args = [sys.executable,'-m','poremc.mcp','--root',str(store.root),
            'work',run,'--workers',str(workers),'--backend',backend]
    logs = store.root/'runs'/run
    logs.mkdir(parents=True,exist_ok=True)
    with (logs/'controller.log').open('ab') as log:
        subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
                         start_new_session=True,close_fds=True)


def create_server(store,input_directory='.',backend='local',workers=1):
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError('Install opengcmc[mcp] to serve MCP tools') from exc
    server = FastMCP('OpenGCMC')
    _integer(workers,'workers')
    if backend not in ('local','slurm'):
        raise ValueError('backend must be local or slurm')

    @server.tool()
    def opengcmc_plan(config: dict, pressure_pa: list[float] | None = None,
                      reservoirs: list[dict] | None = None, count_windows: list[dict] | None = None,
                      replicas: int = 4,
                      block_steps: int = 10000, count_mcse: float | None = None,
                      log_selectivity_mcse: float | None = None) -> dict[str, Any]:
        """Prepare an isotherm or harmonic count windows for a free-energy profile."""
        return store.plan(config,base_directory=input_directory,pressure_pa=pressure_pa,reservoirs=reservoirs,
                          count_windows=count_windows,replicas=replicas,block_steps=block_steps,count_mcse=count_mcse,
                          log_selectivity_mcse=log_selectivity_mcse)

    @server.tool()
    def opengcmc_submit(plan_id: str, seed: int, idempotency_key: str) -> dict[str, Any]:
        """Submit an immutable calculation plan and return its durable run ID."""
        submission = store.submit(plan_id,seed,idempotency_key)
        if submission['created'] or store.status(submission['run_id'])['status']=='queued':
            _launch(store,submission['run_id'],workers,backend)
        return submission

    @server.tool()
    def opengcmc_status(run_id: str) -> dict[str, Any]:
        """Report committed steps, completed blocks and chain states."""
        return store.status(run_id)

    @server.tool()
    def opengcmc_result(run_id: str) -> dict[str, Any]:
        """Read aggregated observables and the contributing sample artifacts."""
        return store.result(run_id)

    @server.tool()
    def opengcmc_cancel(run_id: str) -> dict[str, Any]:
        """Cancel further work and retain committed data."""
        return store.cancel(run_id)

    @server.tool()
    def opengcmc_resume(run_id: str) -> dict[str, Any]:
        """Restart missing work from the last committed checkpoints."""
        state = store.resume(run_id)
        if state['status'] in ('queued','running'):
            _launch(store,run_id,workers,backend)
        return state

    return server


def main(argv=None):
    import json

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default=os.environ.get('OPENGCMC_RUN_DIRECTORY','opengcmc-runs'))
    sub = p.add_subparsers(dest='action',required=True)
    serve = sub.add_parser('serve')
    serve.add_argument('--input-directory',default='.')
    serve.add_argument('--backend',choices=('local','slurm'),default='local')
    serve.add_argument('--workers',type=int,default=1)
    work = sub.add_parser('work')
    work.add_argument('run_id')
    work.add_argument('--backend',choices=('local','slurm'),default='local')
    work.add_argument('--workers',type=int,default=1)
    work.add_argument('--partition')
    array = sub.add_parser('array-item')
    array.add_argument('manifest')
    submit = sub.add_parser('submit')
    submit.add_argument('input')
    submit.add_argument('--seed',type=int,required=True)
    submit.add_argument('--key',required=True)
    submit.add_argument('--pressure-pa',type=float,nargs='+')
    submit.add_argument('--replicas',type=int,default=4)
    submit.add_argument('--block-steps',type=int,default=10000)
    for action in ('status','result','cancel','resume'):
        sub.add_parser(action).add_argument('run_id')
    args = p.parse_args(argv)
    if args.action=='array-item':
        from .slurm import run_array_item
        run_array_item(args.manifest)
        return 0
    store = RunStore(args.root)
    if args.action=='serve':
        create_server(store,args.input_directory,args.backend,args.workers).run(transport='stdio')
        return 0
    if args.action=='work':
        start=time.perf_counter()
        if args.backend=='slurm':
            from .slurm import run_slurm
            answer=run_slurm(store,args.run_id,args.workers,partition=args.partition)
        else:
            answer=store.run_local(args.run_id,args.workers)
        answer['controller_wall_seconds']=time.perf_counter()-start
    elif args.action=='submit':
        path=Path(args.input).resolve()
        plan=store.plan(_read(path),base_directory=path.parent,pressure_pa=args.pressure_pa,
                        replicas=args.replicas,block_steps=args.block_steps)
        answer=store.submit(plan['plan_id'],args.seed,args.key)
    else:
        answer=getattr(store,args.action)(args.run_id)
    print(json.dumps(answer,indent=2,allow_nan=False))
    return 0 if answer.get('status')!='failed' else 1


if __name__=='__main__':
    raise SystemExit(main())
