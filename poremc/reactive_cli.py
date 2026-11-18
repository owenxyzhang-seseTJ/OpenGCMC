"""Command-line runner for explicit hydration, constant pH and reaction currents."""
from pathlib import Path
import argparse
import json
import numpy as np
from .hydration import (MolecularState, MolecularTemplate, Molecule,
                        MolecularHamiltonian, spce_water, point_species, rotated_molecule)
from .reactive import (Reservoirs, ReactionEdge, MolecularExchange, ReactiveGCMC,
                       prepare_reactions, acid_site, redox_site)
from .electrode import GaussianElectrodes
from .electrolyte import FractionalRegion
from .kinetics import KineticState, KineticEdge, ReactionNetwork, bin_current
from .calibration import sample_alchemical_windows
from .thermo_cli import jsonable


def build(config):
    cfg = dict(config)
    cell = np.asarray(cfg['cell'], float)
    if cell.shape == (3,):
        cell = np.diag(cell)
    templates = []
    for entry in cfg['templates']:
        item = dict(entry)
        kind = item.pop('kind', 'custom')
        if kind == 'spce':
            templates.append(spce_water())
        elif kind == 'ion':
            templates.append(point_species(**item))
        elif kind == 'acid':
            templates.append(acid_site(**item))
        elif kind == 'redox':
            templates.append(redox_site(**item))
        elif kind == 'custom':
            item['states'] = [MolecularState(**s) for s in item['states']]
            templates.append(MolecularTemplate(**item))
        else:
            raise ValueError(f'unknown template kind {kind}')
    edges = [ReactionEdge(**x) for x in cfg.get('reactions', [])]
    templates, edges = prepare_reactions(templates, edges)
    params = dict(cfg.get('forcefield', {}))
    params.setdefault('cutoff', 6.)
    params.setdefault('alpha', .35)
    params.setdefault('kmax', 4)
    params.setdefault('dielectric', 1.)
    electrodes = None
    if cfg.get('electrodes'):
        e = dict(cfg['electrodes'])
        if 'plane_grid' in e:
            g = e.pop('plane_grid')
            nx, ny = g['shape']
            e['positions'] = [[cell[0, 0] * (i + .5) / nx, cell[1, 1] * (j + .5) / ny, z]
                             for z in g['z_A'] for i in range(nx) for j in range(ny)]
            potentials = e.pop('plane_potentials_V')
            if len(potentials) != len(g['z_A']):
                raise ValueError('one potential per plane required')
            e['potentials'] = np.repeat(potentials, nx * ny)
        electrodes = GaussianElectrodes(cell=cell, **{k: params[k] for k in ('cutoff', 'alpha', 'kmax', 'dielectric')},
                                       slab=params.get('slab', False), **e)
    rng = np.random.default_rng(cfg.get('seed', 123))
    molecules = []
    for item in cfg.get('initial', []):
        d = dict(item)
        tid = d.pop('template')
        state = d.pop('state', 0)
        if 'grid' in d:
            g = d.pop('grid')
            shape = np.asarray(g['shape'], int)
            lower, upper = np.asarray(g['lower_A']), np.asarray(g['upper_A'])
            axes = [lower[j] + (np.arange(shape[j]) + .5) * (upper[j] - lower[j]) / shape[j] for j in range(3)]
            skip = set(g.get('skip_flat_indices', []))
            for k, anchor in enumerate(np.array(np.meshgrid(*axes, indexing='ij')).reshape(3, -1).T):
                if k not in skip:
                    molecules.append(Molecule(tid, state, rotated_molecule(templates[tid], anchor, rng), **d))
        else:
            anchor = np.asarray(d.pop('anchor_A', [0, 0, 0]))
            xyz = np.asarray(d.pop('positions', templates[tid].geometry + anchor))
            molecules.append(Molecule(tid, state, xyz, **d))
    region = FractionalRegion(cell, **cfg.get('region', {}))
    h = MolecularHamiltonian(templates, cell, electrodes=electrodes, **params)
    try:
        h.set_molecules(molecules)
        return ReactiveGCMC(h, Reservoirs(**cfg.get('reservoirs', {})), region=region,
                            log_activities=cfg.get('log_activities'),
                            exchanges=[MolecularExchange(**e) for e in cfg.get('exchanges', [])],
                            reactions=edges, seed=cfg.get('seed', 123), **cfg.get('moves', {}))
    except Exception:
        h.close()
        raise


def build_network(config):
    return ReactionNetwork([KineticState(**s) for s in config['states']],
                           [KineticEdge(**e) for e in config['edges']],
                           **config.get('network', {}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('run', 'kinetics', 'calibrate', 'transport'):
        p = sub.add_parser(name)
        p.add_argument('config')
        p.add_argument('-o', '--output', default='run_04')
        if name == 'run':
            p.add_argument('--restart')
    args = parser.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text(encoding='utf8'))
    out = Path(args.output)
    if args.command != 'calibrate':
        out.mkdir(parents=True, exist_ok=True)
        (out / 'config.json').write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding='utf8')
    if args.command in ('run', 'calibrate'):
        sim = build(cfg)
        try:
            if args.command == 'calibrate':
                cal = dict(cfg['calibration'])
                edge_index = cal.pop('edge_index', 0)
                if type(edge_index) is not int or not 0 <= edge_index < len(sim.reactions):
                    raise ValueError('calibration.edge_index must select a configured reaction')
                edge = sim.reactions[edge_index]
                uid = cal.get('uid')
                if type(uid) is not int or uid < 0:
                    raise ValueError('calibration.uid must be a nonnegative molecule UID')
                try:
                    index = sim.h.molecule_index(uid)
                except StopIteration as err:
                    raise ValueError(f'calibration.uid {uid} does not identify an initial molecule') from err
                molecule = sim.h.molecules[index]
                if molecule.template != edge.template:
                    raise ValueError(f'calibration.uid {uid} belongs to template {molecule.template}; '
                                     f'reaction index {edge_index} requires template {edge.template}')
                r = sample_alchemical_windows(sim, bridge_states=edge.bridge, **cal)
                out.mkdir(parents=True, exist_ok=True)
                (out / 'config.json').write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding='utf8')
                np.savez_compressed(out / 'alchemical_windows.npz', **{k: v for k, v in r.items() if isinstance(v, np.ndarray)})
                (out / 'calibration.json').write_text(json.dumps(jsonable(r), indent=2), encoding='utf8')
            else:
                if args.restart:
                    sim.restore_checkpoint(args.restart)
                r = sim.run(**cfg.get('run', {}))
                fields = ('counts', 'energy_kelvin', 'state_counts', 'electrode_charges')
                np.savez_compressed(out / 'trajectory.npz', **{k: r.pop(k) for k in fields})
                frames = r.pop('frames')
                if frames:
                    # Each record identifies a molecule and its first atom offset.
                    atoms, records = [], []
                    offset = 0
                    for frame, mols in enumerate(frames):
                        for m in mols:
                            atoms.extend(m.positions.tolist())
                            records.append([frame, m.uid, m.template, m.state, offset, len(m.positions)])
                            offset += len(m.positions)
                    np.savez_compressed(out / 'coordinates.npz', positions=np.asarray(atoms), records=np.asarray(records, int))
                sim.save_checkpoint(out / 'checkpoint.json')
                r['model_fingerprint'] = sim.fingerprint()
                r['profile'] = sim.h.atomic.profile
                (out / 'summary.json').write_text(json.dumps(jsonable(r), indent=2), encoding='utf8')
        finally:
            sim.h.close()
    elif args.command == 'transport':
        from .transport import DiffusionRedox
        model = DiffusionRedox(**cfg['transport'])
        r = model.run(**cfg['protocol'])
        np.savez_compressed(out / 'transport.npz', **{k: v for k, v in r.items() if isinstance(v, np.ndarray)})
        (out / 'summary.json').write_text(json.dumps(jsonable({k: v for k, v in r.items() if not isinstance(v, np.ndarray)}), indent=2), encoding='utf8')
    else:
        network = build_network(cfg)
        task = cfg.get('task', 'ssa')
        if task == 'ssa':
            opts = dict(cfg.get('ssa', {}))
            segments = opts.pop('segments')
            r = network.ssa(segments, **opts)
            if 'current_bins' in cfg:
                bins = np.asarray(cfg['current_bins'])
                b = bin_current(r['events'], bins, network.m)
                np.savez_compressed(out / 'current_trace.npz', **b)
            (out / 'events.json').write_text(json.dumps(jsonable(r['events']), indent=2), encoding='utf8')
            r = {k: v for k, v in r.items() if k != 'events'}
        elif task == 'ramp':
            r = network.ramp(**cfg['ramp'])
            np.savez_compressed(out / 'voltammetry.npz', **{k: v for k, v in r.items() if isinstance(v, np.ndarray)})
        elif task == 'steady':
            r = network.steady_state(**cfg['steady'])
        else:
            raise ValueError('kinetic task must be ssa, ramp or steady')
        (out / 'summary.json').write_text(json.dumps(jsonable(r), indent=2), encoding='utf8')
    print(json.dumps(jsonable({k: v for k, v in r.items() if not isinstance(v, np.ndarray)}), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
