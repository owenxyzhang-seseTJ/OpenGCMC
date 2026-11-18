"""Deterministic numerical fixtures, not a calibrated p-xylene/MOF model.

The large fixture matches the *work dimensions* supplied by the user:
4416 fixed sites, 12 guest sites, 10 trials, 10 initial molecules and kmax 13.
No original CIF, p-xylene force field or HPC input was supplied in this turn.
All five native move types have equal probability in the benchmark.
"""
from __future__ import annotations

import hashlib
import json
import math
import numpy as np


def workload_fixture(kind="aromatic12", host_atoms=4416, kmax=13):
    from poremc.topology import Molecule

    rng = np.random.default_rng(620613)
    cell = np.diag([48., 48., 48.])
    if host_atoms == 4416:
        # Eight dense corrugated planes form two periodic, accessible slits.
        # 24 * 23 * 8 = 4416 fixed sites, with no coincident positions.
        a, b, c = np.meshgrid(np.arange(24), np.arange(23), np.arange(8), indexing="ij")
        planes = np.array([.4, 1.5, 2.6, 3.7, 24.4, 25.5, 26.6, 27.7])
        hp = np.column_stack(((a.ravel()+.23)*48/24,
                              (b.ravel()+.39)*48/23, planes[c.ravel()]))
        hp += rng.uniform(-.13, .13, hp.shape)
    else:
        hp = rng.uniform(0, 48, (host_atoms, 3))
    hq = rng.normal(0., .08, host_atoms)
    if host_atoms:
        hq -= hq.mean()
    if kind == "aromatic12":
        phi = np.arange(6)*math.pi/3
        ring = np.column_stack((1.4*np.cos(phi), 1.4*np.sin(phi), np.zeros(6)))
        hid = np.array([1, 2, 4, 5])
        hydrogens = ring[hid]*2.48/1.4
        methyl = ring[[0, 3]]*2.9/1.4
        local = np.concatenate((ring, hydrogens, methyl))
        molecule = Molecule("synthetic_aromatic_12", [3.35]*6+[2.40]*4+[3.70]*2,
            [24.]*6+[5.]*4+[45.]*2,
            charge=[0., -.12, -.12, 0., -.12, -.12]+[.12]*4+[0., 0.],
            rigid_positions=local)
        local = molecule.rigid_positions.copy()
    elif kind == "flexible12":
        molecule = Molecule("synthetic_flexible_12", [2.2]*12, [6.]*12,
            charge=np.tile([.03, -.03], 6), bond_lengths=[0.]+[1.3]*11,
            angles=[math.radians(112.)]*12,
            torsions=[{"indices":[i, i+1, i+2, i+3], "k":15., "periodicity":3}
                      for i in range(9)])
        local = None
    else:
        raise ValueError(kind)
    initial = []
    for i in range(10):
        root = np.array([6.+9.*(i%5), 10.+21.*(i//5), 13.+24.*(i%2)])
        if local is not None:
            angle = .17*i
            rot = np.array([[math.cos(angle), -math.sin(angle), 0.],
                            [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]])
            xyz = local@rot.T + root
        else:
            # A planar trans zig-zag avoids packing or internal overlaps in
            # the starting state.  The subsequent CBMC explores all torsions.
            xyz = np.zeros((12, 3))
            half_turn=math.radians((180.-112.)/2)
            for j in range(1,12):
                xyz[j]=xyz[j-1]+1.3*np.array([math.cos(half_turn),(-1.)**j*math.sin(half_turn),0.])
            center=np.array([7.+16.*(i%3),8.+16.*((i//3)%3),13.+24.*(i//9)])
            xyz += center-xyz.mean(axis=0)
        initial.append(xyz)
    physics = dict(cell=cell, cutoff=10., alpha=.28, kmax=(kmax,)*3,
        host_positions=hp, host_sigma=np.full(host_atoms, 2.9),
        host_epsilon=np.full(host_atoms, 4.), host_charge=hq,
        temperature=350., activity=(12. if kind=='flexible12' else 8.)/float(np.linalg.det(cell)),
        trials=10, max_translation=.65, max_rotation=.35,
        scale14_lj=.5, scale14_coulomb=.75,
        weights={name:1. for name in ("insert", "delete", "translate", "rotate", "regrow")})
    return molecule, physics, initial


def make_potential(molecule, physics):
    from poremc.potentials import ClassicalPotential
    return ClassicalPotential([molecule], physics["cell"], physics["cutoff"],
        host_positions=physics["host_positions"], host_sigma=physics["host_sigma"],
        host_epsilon=physics["host_epsilon"], host_charge=physics["host_charge"],
        coulomb="ewald", alpha=physics["alpha"], kmax=physics["kmax"],
        algorithm="cached", threads=1, scale14_lj=physics["scale14_lj"],
        scale14_coulomb=physics["scale14_coulomb"], allow_charge_background=True)


def input_hash(molecule, physics, initial):
    def serial(x):
        if isinstance(x, np.ndarray):return x.tolist()
        if isinstance(x, (np.integer, np.floating)):return x.item()
        raise TypeError(type(x).__name__)
    data=dict(molecule=molecule.to_dict(), physics=physics, initial=initial)
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=serial).encode()).hexdigest()


def simulation_arguments(physics):
    return dict(trials=physics["trials"], move_weights=physics["weights"],
                max_translation=physics["max_translation"], max_rotation=physics["max_rotation"])
