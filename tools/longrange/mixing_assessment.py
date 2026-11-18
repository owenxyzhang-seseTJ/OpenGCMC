#!/usr/bin/env python3
"""Count-observable mixing diagnostics from recorded complete-loop traces.

These are observable-specific autocorrelation estimates, not a universal
effective-sample speedup. Block errors and replicate means are retained so
that a fast chain trapped in the wrong occupancy region is visible.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import statistics
import numpy as np


def count_diagnostics(trace,seconds,sample_every):
    x=np.asarray(trace,float);n=len(x)
    mean=float(np.mean(x));variance=float(np.var(x))
    blocks=min(30,n//10)
    if blocks>=2:
        block_size=n//blocks
        means=x[:blocks*block_size].reshape(blocks,block_size).mean(1)
        block_se=float(np.std(means,ddof=1)/math.sqrt(blocks))
    else:
        means=np.array([]);block_size=0;block_se=None
    if variance>0 and n>3:
        size=1<<(2*n-1).bit_length();centered=x-mean
        transformed=np.fft.rfft(centered,n=size)
        correlation=np.fft.irfft(transformed*transformed.conj(),n=size)[:n]
        correlation/=np.arange(n,0,-1);correlation/=correlation[0]
        summed=0.;last=float('inf');used=0
        for lag in range(1,n-1,2):
            pair=float(correlation[lag]+correlation[lag+1])
            if pair<=0:break
            pair=min(last,pair);summed+=pair;last=pair;used=lag+1
        tau=max(1.,1+2*summed);ess=n/tau
    else:tau=None;ess=None;used=0
    return dict(samples=n,mean=mean,variance=variance,count_range=[int(x.min()),int(x.max())],
                block_means=means.tolist(),block_size_samples=block_size,block_se=block_se,
                integrated_autocorrelation_samples=tau,
                integrated_autocorrelation_moves=None if tau is None else tau*sample_every,
                positive_pair_last_lag_samples=used,count_ess=ess,
                count_ess_per_second=None if ess is None or seconds is None else ess/seconds)


def analyze(label,path):
    data=json.loads(Path(path).read_text())
    rows=[]
    for source in data['rows']:
        seconds=source['production_seconds'] if data.get('performance_measurement_valid',True) else None
        row=count_diagnostics(source['count_trace'],seconds,source['sample_every'])
        row.update(label=label,case=data['case'],mode=source['mode'],seed=source['seed'],
                   steps=source['steps'],burnin=source['burnin'])
        rows.append(row)
    summary=[]
    for mode in data['modes']:
        group=[r for r in rows if r['mode']==mode]
        if not group:continue
        means=[r['mean'] for r in group]
        sample_counts=[r['samples'] for r in group]
        pooled=float(np.average(means,weights=sample_counts))
        variance_mean=sum((r['samples']/sum(sample_counts))**2*(r['block_se'] or 0.)**2 for r in group)
        ess_rates=[r['count_ess_per_second'] for r in group if r['count_ess_per_second'] is not None]
        summary.append(dict(label=label,mode=mode,replicate_means=means,pooled_mean=pooled,
            combined_within_chain_block_se=math.sqrt(variance_mean),
            replicate_mean_se=statistics.stdev(means)/math.sqrt(len(means)) if len(means)>1 else None,
            total_count_ess=sum(r['count_ess'] or 0. for r in group),
            median_count_ess_per_second=statistics.median(ess_rates) if ess_rates else None))
    return dict(input_path=str(Path(path)),label=label,case=data['case'],rows=rows,summary=summary)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',action='append',required=True,help='LABEL=benchmark.json; repeat for comparisons')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    results=[]
    for item in args.input:
        label,path=item.split('=',1);results.append(analyze(label,path))
    data=dict(observable='molecule count N',
        autocorrelation_estimator='FFT autocorrelation, initial positive adjacent pairs, monotone pairs, tau>=1',
        block_error='Up to30 contiguous blocks; error of block means; finite-chain estimate',
        interpretation='Count ESS is observable-specific. Inspect replicate agreement and mixing before using ESS/s.',
        datasets=results)
    Path(args.output).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    for result in results:
        for row in result['summary']:print(json.dumps(row))


if __name__=='__main__':main()
