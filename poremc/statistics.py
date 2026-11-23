"""Autocorrelation-aware simulation summaries; no independent-sample SEM shortcut."""
from __future__ import annotations
import math
import numpy as np


def integrated_autocorrelation_time(values):
    """FFT autocovariance and Geyer initial-positive paired-sum estimate.

    Returns NaN for a constant or fewer-than-four-sample observable: its
    decorrelation cannot be diagnosed from that trajectory.
    """
    x = np.asarray(values, float)
    if x.ndim != 1 or len(x) < 4 or not np.isfinite(x).all():
        return math.nan
    x = x - x.mean()
    n = len(x)
    var = np.dot(x, x) / n
    if var <= 0:
        return math.nan
    f = np.fft.rfft(x, n=1 << (2*n-1).bit_length())
    acov = np.fft.irfft(f*np.conj(f))[:n] / n
    rho = acov / acov[0]
    # Pair rho(0)+rho(1), rho(2)+rho(3), etc.; enforce monotone pairs.
    total = 0.0
    previous = math.inf
    for i in range(0, n-1, 2):
        pair = float(rho[i] + rho[i+1])
        if pair <= 0:
            break
        pair = min(pair, previous)
        total += pair
        previous = pair
    return max(1.0, -1.0 + 2.0 * total)


def effective_sample_size(values):
    tau = integrated_autocorrelation_time(values)
    return len(values) / tau if math.isfinite(tau) else math.nan


def block_standard_error(values, blocks=10):
    x = np.asarray(values, float)
    if len(x) < 4:
        return math.nan
    blocks = min(int(blocks), len(x)//2)
    if blocks < 2:
        return math.nan
    size = len(x)//blocks
    means = x[:blocks*size].reshape(blocks, size).mean(axis=1)
    return float(means.std(ddof=1)/math.sqrt(blocks))


def summarize(values, elapsed_seconds=None):
    x = np.asarray(values, float)
    if x.ndim != 1 or len(x) == 0:
        raise ValueError("summarize requires a nonempty one-dimensional series")
    tau = integrated_autocorrelation_time(x)
    ess = len(x)/tau if math.isfinite(tau) else math.nan
    std = float(x.std(ddof=1)) if len(x)>1 else math.nan
    ans = {"samples": len(x), "mean": float(x.mean()), "std": std,
           "tau_int_saved_samples": tau, "ess": ess,
           "sem_autocorrelation": std/math.sqrt(ess) if ess>0 else math.nan,
           "sem_blocks": block_standard_error(x)}
    if elapsed_seconds is not None:
        ans["effective_samples_per_second"] = ess/elapsed_seconds if elapsed_seconds>0 else math.nan
    return ans


def split_rhat(chains):
    """Classical split R-hat for equal-length chains, not rank-normalized R-hat."""
    x = np.asarray(chains, float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 4:
        raise ValueError("need at least two chains with four saved draws each")
    n = x.shape[1]//2
    y = np.concatenate([x[:, :n], x[:, -n:]], axis=0)
    w = np.mean(np.var(y, axis=1, ddof=1))
    b = n*np.var(y.mean(axis=1), ddof=1)
    if w == 0:
        return 1.0 if b == 0 else math.inf
    varplus = ((n-1)*w+b)/n
    return float(math.sqrt(varplus/w))


def rank_split_rhat(chains):
    """Maximum of rank-normalized and folded split R-hat (Vehtari et al.)."""
    from scipy.special import ndtri
    from scipy.stats import rankdata

    x = np.asarray(chains, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 4 or not np.isfinite(x).all():
        raise ValueError("need at least two equal finite chains with four or more draws")
    half = x.shape[1] // 2
    x = np.concatenate((x[:, :half], x[:, -half:]), axis=1)
    if np.ptp(x) == 0:
        return math.nan

    def normal_scores(values):
        ranks = rankdata(values).reshape(values.shape)
        return ndtri((ranks - 3/8) / (values.size + 1/4))

    return max(split_rhat(normal_scores(x)),
               split_rhat(normal_scores(np.abs(x-np.median(x)))))


def chain_mean_interval(chains):
    """Mean and approximate 95% interval over independent equal-length chains.

    Use the larger of the between-chain and within-chain correlation-aware
    errors. A constant series has no empirical information about rare visits.
    """
    from scipy.stats import t

    x = np.asarray(chains, dtype=float)
    if x.ndim != 2 or not x.size or not np.isfinite(x).all():
        raise ValueError("need equal-length nonempty finite chains")
    n, length = x.shape
    mean = float(x.mean())
    sem = rhat = ess = math.nan
    if n >= 2 and length >= 4 and np.ptp(x) > 0:
        rhat = rank_split_rhat(x)
        stats = [summarize(chain) for chain in x]
        if all(math.isfinite(v['sem_autocorrelation']) for v in stats):
            between = float(np.std(x.mean(axis=1),ddof=1)/math.sqrt(n))
            within = math.sqrt(sum(v['sem_autocorrelation']**2 for v in stats))/n
            sem = max(between,within)
            ess = min(float(n*length),float(x.var(ddof=1)/sem**2))
    delta = float(t.ppf(.975,n-1)*sem) if math.isfinite(sem) else math.nan
    return {'mean':mean,'mcse':sem,'ci95':[mean-delta,mean+delta],
            'rank_split_rhat':rhat,'ess':ess,'chains':n,'draws_per_chain':length}
