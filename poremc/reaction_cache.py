"""Exact local quadratic charge response for frozen-geometry reaction trials.

For identical LJ parameters, U(q+dq)-U(q)=g.dq+1/2*dq.H.dq.
H contains Ewald, intramolecular exclusions, slab and the constrained electrode
Schur complement. A cache is valid ONLY at its accepted-state version.
"""
from __future__ import annotations
import numpy as np
from scipy.special import erfc
from .cell import cutoff_image


class LocalChargeResponse:
    def __init__(self, molecular_hamiltonian, uid):
        self.h = molecular_hamiltonian
        h = self.h.atomic
        self.version = h.version
        self.uid = uid
        i = self.h.molecule_index(uid)
        m = self.h.molecules[i]
        template = self.h.templates[m.template]
        state = template.states[m.state]
        self.template = m.template
        self.q = np.asarray(state.charges)
        self.sigma, self.epsilon = tuple(state.sigma), tuple(state.epsilon)
        p = m.positions
        n = len(p)
        self.gradient = np.zeros(n)
        self.hessian = np.zeros((n, n))
        if h.ewald is None:
            return
        offsets = np.r_[0, np.cumsum([len(x.positions) for x in self.h.molecules])]
        mask = np.ones(len(h.types), bool)
        mask[offsets[i]:offsets[i+1]] = False
        envp = np.r_[h.host_positions, h.positions[mask]]
        envq = np.r_[h.host_charge, h.valences[h.types[mask]]]
        e = h.ewald
        phase = np.exp(1j * (p @ e.k.T))
        own_structure = self.q @ phase
        env_structure = h.structure - own_structure
        G = 2 * np.real((phase * e.coefficient) @ phase.conj().T)
        G -= np.pi / (e.alpha**2 * e.volume)
        G[np.diag_indices(n)] -= 2 * e.alpha / np.sqrt(np.pi)
        for a in range(n):
            for b in range(a):
                r = float(np.linalg.norm(p[a]-p[b]))
                # Molecules are unwrapped and all primary bond distances <rc.
                G[a,b] += (erfc(e.alpha*r) if r<e.cutoff else 0.)/r - 1/r
                G[b,a] = G[a,b]
        linear = 2 * np.real((phase * e.coefficient) @ env_structure.conj())
        linear -= np.pi * envq.sum() / (e.alpha**2 * e.volume)
        if len(envp):
            d = cutoff_image(p[:,None]-envp[None], h.cell)
            r = np.linalg.norm(d, axis=2)
            active = (r < e.cutoff) & (r > 1e-12)
            safe = np.where(active, r, 1.)
            linear += np.sum(np.where(active, erfc(e.alpha*safe)/safe, 0.) * envq[None], axis=1)
        if h.slab:
            G += 4*np.pi/e.volume * np.outer(p[:,2],p[:,2])
            linear += 4*np.pi/e.volume * p[:,2] * (envq @ envp[:,2])
        G *= h.K
        linear *= h.K
        self.hessian = G
        self.gradient = linear + G @ self.q
        if h.electrodes:
            response = h.electrodes.response
            F = h.electrodes.fields(p, np.ones(n))
            one = np.ones(n)
            v = F @ response.t
            self.hessian += (-F @ response.P @ F.T - np.outer(one,v) - np.outer(v,one)
                             + np.outer(one,one)/response.den)
            self.gradient += F @ h.electrode_state['q'] + h.electrode_state['lagrange']*one
        self.hessian = .5*(self.hessian+self.hessian.T)

    def deltas(self, states):
        if self.h.atomic.version != self.version:
            raise ValueError('stale local charge-response cache')
        ts = self.h.templates[self.template].states
        targets = [ts[s] for s in states]
        if any(tuple(s.sigma) != self.sigma or tuple(s.epsilon) != self.epsilon for s in targets):
            raise ValueError('local quadratic response requires identical LJ parameters')
        dq = np.array([s.charges for s in targets])-self.q
        return dq @ self.gradient + .5*np.sum((dq @ self.hessian)*dq,axis=1)


def state_energy_differences(hamiltonian, uid, states):
    """Batch state energy changes; fall back explicitly for LJ-changing states."""
    i = hamiltonian.molecule_index(uid)
    m = hamiltonian.molecules[i]
    template = hamiltonian.templates[m.template]
    source = template.states[m.state]
    states = tuple(states)
    same_lj = all(tuple(template.states[s].sigma)==tuple(source.sigma) and
                  tuple(template.states[s].epsilon)==tuple(source.epsilon) for s in states)
    if same_lj:
        return LocalChargeResponse(hamiltonian, uid).deltas(states)
    result = []
    for s in states:
        mol = m.clone(); mol.state = s
        result.append(hamiltonian.propose([i],[mol])['delta_energy'])
    return np.asarray(result)
