#!/usr/bin/env python3
"""Advise a fixed physical reciprocal cutoff for a PoreMC JSON input.

The reference is the input's complete finite rectangular Ewald mode set.
Only an explicit recommendation is written: this program never edits the
input or adjusts a running sampler.  A nonzero cutoff defines a different
finite Ewald Hamiltonian, with the reported analytic bound on the omitted
complete-molecule insertion/deletion/replacement energy increment.

Example:
  python tools06/advise_reciprocal.py input.json --max-molecules 64 \
      --tolerance-K 0.1 --output reciprocal_advice.json

The tolerance applies to the conservative replacement bound, twice the
insertion bound, covering translation, rotation and complete regrowth too.
No sampler, host grid, or initial-particle energy calculation is constructed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poremc.electrostatics import Ewald
from poremc.units import COULOMB_KA


def host_structure(ewald, positions, charges, block_atoms=64):
    """Bounded-memory Fourier transform; exploit the full set's conjugacy."""
    xyz = np.asarray(positions, float).reshape(-1, 3)
    q = np.asarray(charges, float).reshape(-1)
    if len(xyz) != len(q) or not np.isfinite(xyz).all() or not np.isfinite(q).all():
        raise ValueError("Host positions and charges must be finite and aligned.")
    if not isinstance(block_atoms, int) or block_atoms < 1:
        raise ValueError("block_atoms must be a positive integer.")
    count = len(ewald.k)
    paired = count % 2 == 0 and np.array_equal(ewald.k, -ewald.k[::-1])
    evaluated = count // 2 if paired else count
    result = np.zeros(evaluated, dtype=np.complex128)
    # Fractional wrapping bounds arguments without changing periodic phases.
    fraction = xyz @ ewald.inverse
    xyz = (fraction - np.floor(fraction)) @ ewald.cell
    active = q != 0.
    xyz, q = xyz[active], q[active]
    for begin in range(0, len(q), block_atoms):
        phase = np.einsum("ai,ki->ak", xyz[begin:begin+block_atoms],
                          ewald.k[:evaluated], optimize=False)
        values = np.exp(1j * phase)
        values *= q[begin:begin+block_atoms, None]
        result += values.sum(axis=0)
    return np.concatenate((result, result[::-1].conj())) if paired else result


def advise(ewald, host_s, molecule_qabs, max_molecules, tolerance_K,
           host_qabs=0., host_atoms=0):
    """Choose the most aggressive complete-shell cutoff satisfying the bound.

    In exact arithmetic, for discarded D and Qabs=sum |q_i|:
      B_ins = sum_D A_k [2 (|S_H(k)|+Nmax Qabs) Qabs + Qabs^2].
    The replacement bound is 2 B_ins, with the same excluded-molecule bath.
    Self, background, real-space and intramolecular terms cancel between the
    two cutoff choices and are not included in this bound.
    """
    if isinstance(max_molecules, bool) or int(max_molecules) != max_molecules or max_molecules < 0:
        raise ValueError("max_molecules must be a nonnegative integer.")
    nmax = int(max_molecules)
    qabs, tolerance = float(molecule_qabs), float(tolerance_K)
    if not math.isfinite(qabs) or qabs < 0 or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Molecular Qabs and tolerance must be finite and nonnegative.")
    sh = np.asarray(host_s, complex).reshape(-1)
    if len(sh) != len(ewald.k) or not np.isfinite(sh).all():
        raise ValueError("One finite host structure factor per full Ewald mode is required.")
    # Analytic inequality is exact arithmetic.  The numerical implementation
    # evaluates it in float64 with conservative transform/summation padding;
    # it is not an interval-arithmetic certification of libm trigonometry.
    padding = (64 * np.finfo(float).eps * max(0., float(host_qabs))
               * (int(host_atoms) + 1 + 2*math.pi*float(np.sum(ewald.kmax))))
    host_upper = np.abs(sh) + padding
    ak = np.nextafter(COULOMB_KA * np.asarray(ewald.coefficient), np.inf)
    terms = ak * (2*(host_upper+nmax*qabs)*qabs + qabs*qabs)
    terms = np.nextafter(terms, np.inf) if qabs else np.zeros_like(terms)
    radius = np.linalg.norm(ewald.k, axis=1)
    order = np.argsort(radius, kind="stable")
    r = radius[order]
    # Degenerate shells can differ by roundoff in triclinic arithmetic.
    gap = 2e-12*np.maximum(1., r[1:])
    ends = np.r_[np.flatnonzero(np.diff(r) > gap)+1, len(r)]
    suffix = np.zeros(len(r)+1, dtype=np.longdouble)
    suffix[:-1] = np.cumsum(terms[order][::-1], dtype=np.longdouble)[::-1]
    def tail(kept):
        value = float(suffix[kept])
        return math.nextafter(value, math.inf) if value else 0.
    kept = len(r)
    for end in ends if tolerance > 0 else []:
        # Keep at least the first shell: the native direct engine requires
        # one conjugate pair even for an electrically neutral zero-charge guest.
        if 2*tail(int(end)) <= tolerance:
            kept = int(end)
            break
    cutoff = 0. if kept == len(r) else float((r[kept-1]+r[kept])/2)
    discarded = radius > cutoff if cutoff else np.zeros(len(r), dtype=bool)
    if int((~discarded).sum()) != kept:
        raise RuntimeError("Shell midpoint did not reproduce the selected mode set.")
    if kept % 2 or not np.array_equal(discarded, discarded[::-1]):
        raise RuntimeError("The recommended radial set must retain conjugate pairs.")
    insertion_bound = tail(kept)
    return {
        "schema": "poremc-reciprocal-advice-v1",
        "reference": {"method": "complete rectangular finite direct Ewald",
                      "kmax": ewald.kmax.tolist(), "alpha_A_inverse": ewald.alpha,
                      "cell_A": ewald.cell.tolist(), "cutoff_real_A": ewald.cutoff},
        "assumptions": {
            "max_molecules": nmax, "molecular_sum_absolute_charge_e": qabs,
            "bath_charge_bound_e": nmax*qabs,
            "scope": "Complete-molecule moves with no more than Nmax molecules in the excluded-molecule bath.",
            "occupancy_rule": "Nmax is an analysis assumption, not a particle cap. If occupancy exceeds it, reassess offline; never retune the Hamiltonian during sampling.",
        },
        "criterion": {"tolerance_K": tolerance, "move": "replacement (twice insertion bound)"},
        "modes": {"rectangular_full": len(r), "rectangular_half": len(r)//2,
                  "retained_full": kept, "retained_half": kept//2,
                  "discarded_full": len(r)-kept, "radial_shells": len(ends),
                  "max_retained_k_A_inverse": float(r[kept-1]),
                  "min_discarded_k_A_inverse": float(r[kept]) if kept<len(r) else None},
        "bounds": {"insertion_or_deletion_increment_K": insertion_bound,
                   "replacement_increment_K": 2*insertion_bound,
                   "unchanged_terms": ["real space", "self", "charge background", "intramolecular correction"],
                   "formula": "sum_D A_k*(2*(abs(S_host(k))+Nmax*Qabs)*Qabs+Qabs**2)",
                   "host_transform_rounding_allowance_e": padding,
                   "evaluation": "Analytic finite-sum inequality, evaluated in double precision with transform and accumulation padding."},
        "recommendation": {"sampling": {"ewald_k_cutoff": cutoff, "ewald_half_spectrum": True}},
        "changes_finite_hamiltonian": bool(kept<len(r)),
        "interpretation": ("Nonzero radial cutoff defines a different finite Ewald Hamiltonian with the stated move-energy bound; this is not preservation of the original finite-k target."
                           if cutoff else "Retain the original rectangular finite-k target; ewald_k_cutoff=0 means no radial truncation."),
        "input_modified": False,
    }


def load_input(path):
    """Read only geometry/charges from the same poremc-v1 JSON conventions."""
    from poremc.io import read_cif
    from poremc.topology import Molecule
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if data.get("schema", "poremc-v1") != "poremc-v1":
        raise ValueError("Unsupported input schema.")
    potential = data.get("potential", {})
    if potential.get("kind", "classical") != "classical" or potential.get("coulomb") != "ewald":
        raise ValueError("Advice requires a classical Ewald input.")
    if data.get("sampling", {}).get("reciprocal_method", "direct") != "direct":
        raise ValueError("This finite-mode advisor applies to direct Ewald, not a PME mesh.")
    species = data.get("species", [])
    if len(species) != 1:
        raise ValueError("The native FastGCMC advisor expects one molecular species.")
    definition = dict(species[0])
    if "file" in definition:
        if len(definition) != 1:
            raise ValueError("Species-file definitions cannot contain inline overrides.")
        definition = json.loads((path.parent/definition["file"]).read_text())
    definition.pop("internal_partition", None)  # Does not affect charge bounds.
    molecule = Molecule.from_dict(definition)
    cell = np.asarray(data.get("cell_A", np.eye(3)*30), float)
    if cell.shape == (3,):
        cell = np.diag(cell)
    host = data.get("host") or {}
    cutoff = float(potential.get("cutoff_A", 9.))
    if "cif" in host:
        force_field = host.get("force_field")
        if isinstance(force_field, str):
            force_field = json.loads((path.parent/force_field).read_text())
        if not isinstance(force_field, dict):
            raise ValueError("A CIF host requires an explicit force_field map.")
        framework = read_cif(path.parent/host["cif"], force_field,
                             host.get("replicas", [1,1,1]), cutoff,
                             host.get("charge_source", "cif"))
        if "cell_A" in data and not np.allclose(cell, framework.cell):
            raise ValueError("Explicit cell disagrees with CIF supercell.")
        cell, xyz, q = framework.cell, framework.positions, framework.charge
    else:
        xyz = np.asarray(host.get("positions_A", []), float).reshape(-1,3)
        q = np.broadcast_to(np.asarray(host.get("charge_e", 0.), float), (len(xyz),)).copy()
    ewald = Ewald(cell, cutoff, float(potential.get("alpha", .25)), potential.get("kmax", 6))
    return ewald, xyz, q, np.asarray(molecule.charge), {
        "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "molecule": molecule.name, "host_atoms": len(q), "guest_sites": molecule.n_atoms,
        "initial_molecules": len(data.get("initial_particles", [])),
        "existing_radial_cutoff": float(data.get("sampling", {}).get("ewald_k_cutoff", 0.)),
    }


def synthetic_fixture_report(output=None, max_molecules=64, tolerances=(.01, .1, 1.)):
    """One host transform reused for the requested small set of analytic bounds.

    This is an offline mode analysis, not a performance benchmark or a model
    of the user's unsupplied CIF/p-xylene input.
    """
    from tools06.fixtures import workload_fixture, input_hash
    molecule, physics, initial = workload_fixture("aromatic12")
    ewald = Ewald(physics["cell"], physics["cutoff"], physics["alpha"], physics["kmax"])
    host_q = physics["host_charge"]
    sh = host_structure(ewald, physics["host_positions"], host_q)
    reports = [advise(ewald, sh, math.fsum(map(abs, molecule.charge)), max_molecules,
                      tolerance, math.fsum(map(abs, host_q)), len(host_q))
               for tolerance in tolerances]
    # Small direct-exponential check, independent of host_structure's pairing.
    # Each molecular placement is a rigid rotation and translation of the
    # fixture body. This checks a universal Fourier bound, not equilibrated
    # thermodynamic configurations or the cost of the production sampler.
    rng = np.random.default_rng(620619)
    def placement(count):
        molecules = []
        for _ in range(count):
            rotation, _ = np.linalg.qr(rng.normal(size=(3,3)))
            if np.linalg.det(rotation) < 0:
                rotation[:,0] *= -1
            molecules.append(molecule.rigid_positions@rotation+rng.random(3)@physics["cell"])
        return np.asarray(molecules).reshape(-1,3)
    def independent_structure(xyz, charges):
        result = np.zeros(len(ewald.k), dtype=complex)
        for begin in range(0,len(charges),32):
            phase = np.einsum("ai,ki->ak",xyz[begin:begin+32],ewald.k,optimize=False)
            result += (charges[begin:begin+32,None]*(np.cos(phase)+1j*np.sin(phase))).sum(axis=0)
        return result
    occupancy = sorted(set((0,min(10,max_molecules),max_molecules)))
    coefficient = COULOMB_KA*ewald.coefficient
    radius = np.linalg.norm(ewald.k,axis=1)
    masks = []
    for row in reports:
        cutoff = row["recommendation"]["sampling"]["ewald_k_cutoff"]
        masks.append(radius>cutoff if cutoff else np.zeros(len(radius),bool))
        row["validation"] = {"complete_move_pairs":0,"bath_occupancies":occupancy,
            "max_observed_omitted_insertion_K":0.,"max_observed_omitted_replacement_K":0.}
    for count in occupancy:
        bath = independent_structure(placement(count),np.tile(molecule.charge,count))
        base = sh+bath
        for _ in range(4):
            new = independent_structure(placement(1),molecule.charge)
            old = independent_structure(placement(1),molecule.charge)
            new_terms = coefficient*(2*np.real(base.conj()*new)+np.abs(new)**2)
            old_terms = coefficient*(2*np.real(base.conj()*old)+np.abs(old)**2)
            for row,mask in zip(reports,masks):
                insertion = abs(float(np.sum(new_terms[mask])))
                replacement = abs(float(np.sum(new_terms[mask]-old_terms[mask])))
                if insertion>row["bounds"]["insertion_or_deletion_increment_K"]+1e-10 or replacement>row["bounds"]["replacement_increment_K"]+1e-10:
                    raise AssertionError("Independent discarded-mode energy exceeded its analytic bound.")
                val=row["validation"];val["complete_move_pairs"]+=1
                val["max_observed_omitted_insertion_K"]=max(val["max_observed_omitted_insertion_K"],insertion)
                val["max_observed_omitted_replacement_K"]=max(val["max_observed_omitted_replacement_K"],replacement)
    data = {
        "scope": "Analytic finite-mode tail advice for the synthetic 4416/12-site fixture, not the user's material or a timing result.",
        "fixture": "aromatic12", "physical_input_sha256": input_hash(molecule, physics, initial),
        "advisor_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_source_sha256": hashlib.sha256((ROOT/"poremc/electrostatics.py").read_bytes()).hexdigest(),
        "host_atoms": len(host_q), "guest_sites": molecule.n_atoms,
        "max_molecules": max_molecules, "reports": reports,
        "validation_method": "Independent full-mode cos/sin sums for random rigid placements at bath occupancies 0, 10, Nmax; finite-mode differences only, no trajectory timings.",
    }
    out = Path(output or ROOT/"evidence06/reciprocal_tuning.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="PoreMC JSON input; relative molecule/CIF paths resolve beside it")
    parser.add_argument("--max-molecules", type=int, required=True)
    parser.add_argument("--tolerance-K", dest="tolerance_K", type=float, default=.1)
    parser.add_argument("--output", default="reciprocal_advice.json")
    args = parser.parse_args()
    ewald, xyz, q, molecular_q, provenance = load_input(args.input)
    if provenance["initial_molecules"] > args.max_molecules:
        parser.error("The input already contains more molecules than --max-molecules.")
    sh = host_structure(ewald, xyz, q)
    result = advise(ewald, sh, math.fsum(map(abs, molecular_q)), args.max_molecules,
                    args.tolerance_K, math.fsum(map(abs, q)), len(q))
    result["input"] = provenance
    out = Path(args.output).resolve()
    if out == Path(args.input).resolve():
        parser.error("Advice output must differ from the simulation input.")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    print(json.dumps({"output": str(out), "recommendation": result["recommendation"],
                      "modes": result["modes"], "bounds": result["bounds"],
                      "changes_finite_hamiltonian": result["changes_finite_hamiltonian"]}, indent=2))


if __name__ == "__main__":
    main()
