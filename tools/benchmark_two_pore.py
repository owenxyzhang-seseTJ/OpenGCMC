"""Enumerate an augmented CBMC kernel on two single-occupancy pores."""

from __future__ import annotations

import argparse
from itertools import product
import json
import math
from pathlib import Path

import numpy as np


STATES = tuple(product(range(4), repeat=2))  # empty, trans, gauche+, gauche-
INDEX = {state: i for i, state in enumerate(STATES)}
N = np.count_nonzero(STATES, axis=1)
ACTIVITY_VOLUME = 0.51
MOVES = (0.31, 0.24, 0.30, 0.15)  # insert, delete, full, suffix
TRIALS = 2
DRAW = np.array([.29, .06, .14, .18, .22, .11])
SITE_ENERGY = np.array([[0., -3.8, -1.4, -1.0], [0., -0.9, -2.7, -1.9]])
CROSS_ENERGY = np.array([[.65, -.8, .4], [.3, -.3, -.5], [-.6, .5, .1]])
MODES = ("reference_only", "reference_mh", "delayed")


def with_pose(state, pose):
    pore, conformation = divmod(pose, 3)
    return (conformation + 1, state[1]) if pore == 0 else (state[0], conformation + 1)


def energy(state):
    a, b = state
    both = CROSS_ENERGY[a-1, b-1] if a and b else 0.0
    return float(SITE_ENERGY[0, a] + SITE_ENERGY[1, b] + both)


def probabilities(energies):
    weights = (ACTIVITY_VOLUME / 6) ** N * np.exp(-energies)
    return weights / weights.sum()


def growth_paths(bath, ref, poses, forced=None):
    """Yield final pose, mean reference weight, and augmented path probability."""
    ids = np.asarray(poses)
    density = DRAW[ids] / DRAW[ids].sum()
    w = np.array([
        0.0 if bath[p//3] else
        math.exp(ref[INDEX[bath]] - ref[INDEX[with_pose(bath, p)]]) / (len(poses) * q)
        for p, q in zip(ids, density)
    ])
    choices = range(len(poses))
    random_count = TRIALS - (forced is not None)
    for random_trials in product(choices, repeat=random_count):
        ids_in_trials = random_trials if forced is None else (poses.index(forced), *random_trials)
        trial_weights = w[list(ids_in_trials)]
        sumw = trial_weights.sum()
        if not sumw:
            continue
        chance = float(np.prod(density[list(random_trials)]))
        if forced is not None:
            yield forced, sumw/TRIALS, chance
            continue
        for trial, weight in zip(ids_in_trials, trial_weights):
            if weight:
                yield poses[trial], sumw/TRIALS, chance*weight/sumw


def kernel(target, ref):
    p = {mode: np.zeros((len(STATES), len(STATES))) for mode in MODES}
    triggers = {mode: np.zeros(len(STATES)) for mode in MODES}
    max_alpha_difference = 0.0

    def accumulate(i, j, chance, ratio):
        nonlocal max_alpha_difference
        log_ref = math.log(ratio)
        log_residual = -(target[j]-target[i]) + (ref[j]-ref[i])
        log_single = min(0., log_ref + log_residual)
        log_delayed = min(0., log_ref) + min(0., log_residual)
        max_alpha_difference = max(max_alpha_difference, math.exp(log_delayed)-math.exp(log_single))
        p["reference_only"][i,j] += chance*math.exp(min(0., log_ref))
        p["reference_mh"][i,j] += chance*math.exp(log_single)
        p["delayed"][i,j] += chance*math.exp(log_delayed)
        triggers["reference_mh"][i] += chance
        triggers["delayed"][i] += chance*math.exp(min(0., log_ref))

    all_poses = tuple(range(6))
    for i, state in enumerate(STATES):
        n = N[i]
        for pose, w, chance in growth_paths(state, ref, all_poses):
            new = INDEX[with_pose(state, pose)]
            ratio = ACTIVITY_VOLUME*w/(n+1)*MOVES[1]/MOVES[0]
            accumulate(i, new, MOVES[0]*chance, ratio)
        for pore, c in enumerate(state):
            if not c:
                continue
            bath = (0,state[1]) if pore == 0 else (state[0],0)
            old_pose = 3*pore+c-1
            for _, w, chance in growth_paths(bath, ref, all_poses, forced=old_pose):
                ratio = n/ACTIVITY_VOLUME/w*MOVES[0]/MOVES[1]
                accumulate(i, INDEX[bath], MOVES[1]*chance/n, ratio)
            for poses, move_chance in ((all_poses, MOVES[2]), (tuple(range(3*pore,3*pore+3)), MOVES[3])):
                for _, old_w, old_chance in growth_paths(bath, ref, poses, forced=old_pose):
                    for pose, new_w, new_chance in growth_paths(bath, ref, poses):
                        j = INDEX[with_pose(bath, pose)]
                        accumulate(i, j, move_chance*old_chance*new_chance/n, new_w/old_w)
    for mat in p.values():
        mat[np.diag_indices_from(mat)] += 1-mat.sum(axis=1)
    return p, triggers, max_alpha_difference


def observables(prob):
    states = np.asarray(STATES)
    occupancy = prob @ (states != 0)
    conformation = np.array([prob @ (states == k) for k in (1,2,3)]).T
    return {
        "P_N": np.bincount(N, weights=prob, minlength=3).tolist(),
        "mean_N": float(prob@N),
        "occupancy_A_over_B": float(occupancy[0]/occupancy[1]),
        "mean_pore_occupancy": occupancy.tolist(),
        "conformation_per_pore": (conformation/occupancy[:,None]).tolist(),
        "conformation_pooled": (conformation.sum(axis=0)/occupancy.sum()).tolist(),
    }


def correlation_time(mat, prob, f):
    """Variance factor including all lags, solved by the Markov Poisson equation."""
    centered = f-prob@f
    h = np.linalg.solve(np.eye(len(prob))-mat+prob[None,:], centered)
    variance = prob@(centered**2)
    return float(2*(prob@(centered*h))/variance-1)


def check_model():
    exact = np.array([energy(s) for s in STATES])
    # Permute pores and alias conformations in the reference. This intentionally
    # preserves P(N) exactly while changing both spatial and conformational weights.
    perm = (0,2,1,3)
    coarse = np.array([energy((perm[b],perm[a])) for a,b in STATES])
    truth = probabilities(exact)
    S = np.asarray(STATES)
    f = {"N":N, "both_occupied":N==2,
         "pore_A_minus_B":(S[:,0]!=0).astype(int)-(S[:,1]!=0),
         "trans_count":(S==1).sum(axis=1)}
    rows = []
    for strength in (0., .25, .5, 1.):
        ref = exact + strength*(coarse-exact)
        matrices, calls, alpha_difference = kernel(exact, ref)
        assert alpha_difference <= 1e-14
        row = {"distortion":strength, "maximum_delayed_minus_single_acceptance":alpha_difference, "modes":{}}
        for mode, mat in matrices.items():
            expected = probabilities(ref) if mode=="reference_only" else truth
            flux = expected[:,None]*mat
            imbalance = float(np.max(np.abs(flux-flux.T)))
            assert imbalance < 3e-15
            np.testing.assert_allclose(mat.sum(axis=1), 1., atol=2e-15)
            assert np.min(mat)>=0
            a = mat.T-np.eye(len(mat))
            a[-1] = 1.
            rhs = np.zeros(len(mat))
            rhs[-1] = 1.
            stationary = np.linalg.solve(a,rhs)
            np.testing.assert_allclose(stationary,expected,rtol=0,atol=8e-15)
            row["modes"][mode] = {
                **observables(stationary),
                "maximum_detailed_balance_error":imbalance,
                "potential_evaluations_per_attempt":float(expected@calls[mode]),
                "state_changes_per_attempt":float(expected@(1-mat.diagonal())),
                "integrated_correlation_factors":{name:correlation_time(mat,expected,v) for name,v in f.items()},
                "transition_matrix":mat.tolist(),
            }
        for name in f:
            single = row["modes"]["reference_mh"]["integrated_correlation_factors"][name]
            delayed = row["modes"]["delayed"]["integrated_correlation_factors"][name]
            assert delayed+1e-12 >= single
        rows.append(row)
    return {"states":STATES, "target_energy_over_kBT":exact.tolist(),
            "coarse_energy_over_kBT":coarse.tolist(), "activity_times_volume":ACTIVITY_VOLUME,
            "moves":MOVES, "trials":TRIALS, "trial_probabilities":DRAW.tolist(),
            "exact":observables(truth), "runs":rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(check_model(), indent=2) + "\n")
