"""Explicit JSON and CIF input with symmetry expansion and no silent guessing."""
from pathlib import Path
from dataclasses import dataclass
import ast
import itertools
import json
import re
import shlex
import numpy as np
from .cell import cell_from_parameters, validate_cell

@dataclass
class Framework:
    cell: np.ndarray
    positions: np.ndarray
    sigma: np.ndarray
    epsilon: np.ndarray
    charge: np.ndarray
    labels: list
    mass_amu: float | None = None
    source: str | None = None

    def to_dict(self):
        return {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in vars(self).items()}

def number(token):
    token=str(token)
    if token in ('.','?'): raise ValueError('unknown CIF numerical value')
    return float(re.sub(r'\(\d+\)(?=([eE][+-]?\d+)?$)','',token))

def _tokens(text):
    # Standard quoted CIF values and semicolon-delimited text fields.
    lines=text.splitlines(); clean=[]; i=0
    while i<len(lines):
        if lines[i].startswith(';'):
            body=[lines[i][1:]]; i+=1
            while i<len(lines) and not lines[i].startswith(';'): body.append(lines[i]); i+=1
            if i==len(lines): raise ValueError('unterminated CIF text field')
            clean.append(shlex.quote('\n'.join(body))); i+=1
        else: clean.append(lines[i]); i+=1
    lex=shlex.shlex('\n'.join(clean),posix=True); lex.whitespace_split=True; lex.commenters='#'
    return list(lex)

def parse_cif(text):
    tokens=_tokens(text); data={}; loops=[]; i=0; block=None
    while i<len(tokens):
        token=tokens[i]
        if token.lower().startswith('data_'):
            if block is not None: raise ValueError('multiple CIF data blocks: select one explicitly')
            block=token[5:]; i+=1
        elif token.lower()=='loop_':
            i+=1; keys=[]
            while i<len(tokens) and tokens[i].startswith('_'): keys.append(tokens[i].lower()); i+=1
            if not keys: raise ValueError('empty CIF loop')
            values=[]
            while i<len(tokens):
                t=tokens[i]
                if t.startswith('_') or t.lower() in ('loop_','stop_') or t.lower().startswith(('data_','save_')): break
                values.append(t); i+=1
            if len(values)%len(keys): raise ValueError('CIF loop has incomplete rows')
            loops.append([dict(zip(keys,values[j:j+len(keys)])) for j in range(0,len(values),len(keys))])
        elif token.startswith('_'):
            if i+1==len(tokens): raise ValueError('CIF key missing value')
            data[token.lower()]=tokens[i+1]; i+=2
        elif token.lower()=='stop_': i+=1
        elif token.lower().startswith('save_'): raise ValueError('CIF save frames are not supported')
        else: i+=1
    return data,loops

def _linear_expr(expr,xyz):
    """Restricted AST evaluator for CIF x,y,z affine symmetry operators."""
    def run(n):
        if isinstance(n,ast.Expression): return run(n.body)
        if isinstance(n,ast.Name) and n.id.lower() in ('x','y','z'): return xyz['xyz'.index(n.id.lower())]
        if isinstance(n,ast.Constant) and type(n.value) in (int,float): return float(n.value)
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)):
            return run(n.operand)*(1 if isinstance(n.op,ast.UAdd) else -1)
        if isinstance(n,ast.BinOp) and isinstance(n.op,(ast.Add,ast.Sub,ast.Mult,ast.Div)):
            l,r=run(n.left),run(n.right)
            if isinstance(n.op,ast.Add): return l+r
            if isinstance(n.op,ast.Sub): return l-r
            if isinstance(n.op,ast.Mult): return l*r
            return l/r
        raise ValueError('invalid symmetry expression')
    return run(ast.parse(expr.strip(),mode='eval'))

def symmetry_matrix(operation):
    parts=operation.lower().replace(' ','').split(',')
    if len(parts)!=3: raise ValueError('symmetry operation needs three coordinates')
    shift=np.array([_linear_expr(p,[0,0,0]) for p in parts])
    matrix=np.column_stack([np.array([_linear_expr(p,np.eye(3)[j]) for p in parts])-shift for j in range(3)])
    # Check affine character and integer crystallographic rotation explicitly.
    probe=np.array([0.13,0.27,0.47]); actual=np.array([_linear_expr(p,probe) for p in parts])
    if not np.allclose(actual,matrix@probe+shift) or not np.allclose(matrix,np.rint(matrix)):
        raise ValueError('nonaffine or noncrystallographic symmetry operation')
    if not np.isclose(abs(np.linalg.det(matrix)),1): raise ValueError('singular symmetry operator')
    return matrix,shift

def read_cif(path,force_field,replicas=(1,1,1),cutoff=None,charge_source='cif'):
    """Expand listed symmetry operations, map explicit parameters, replicate.

    force_field maps atom labels or CIF type symbols to sigma/epsilon in A/K,
    optional charge (e) and mass (amu). With charge_source='cif', every atom
    requires a charge tag; use 'force_field' explicitly for uncharged models.
    Fractional occupancy and disorder are rejected, never silently rounded.
    """
    path=Path(path); data,loops=parse_cif(path.read_text())
    cell=cell_from_parameters(*(number(data['_cell_length_'+k]) for k in 'abc'),
            *(number(data.get('_cell_angle_'+k,90)) for k in ('alpha','beta','gamma')))
    site_rows=next((rows for rows in loops if rows and '_atom_site_fract_x' in rows[0]),None)
    if not site_rows: raise ValueError('CIF fractional atom-site loop not found')
    opkeys=('_space_group_symop_operation_xyz','_symmetry_equiv_pos_as_xyz')
    operations=[row[k] for rows in loops for row in rows for k in opkeys if k in row]
    operations += [data[k] for k in opkeys if k in data]
    if not operations:
        group=data.get('_space_group_name_h-m_alt',data.get('_symmetry_space_group_name_h-m','P 1'))
        number_group=data.get('_space_group_it_number',data.get('_symmetry_int_tables_number','1'))
        if group.replace(' ','').lower() not in ('p1',) or number(number_group)!=1:
            raise ValueError('non-P1 CIF must list symmetry operations; export a symmetry-expanded P1 CIF')
        operations=['x,y,z']
    ops=[symmetry_matrix(op) for op in operations]
    # Symmetry matrices must preserve the actual lattice metric.
    metric=cell@cell.T
    for rotation,_ in ops:
        if not np.allclose(rotation.T@metric@rotation,metric,rtol=1e-4,atol=1e-5):
            raise ValueError('CIF symmetry operation inconsistent with cell metric')
    fractions=[]; parameters=[]; labels=[]; mass=0.0; has_mass=True
    for row in site_rows:
        occupancy=number(row.get('_atom_site_occupancy','1'))
        if not np.isclose(occupancy,1): raise ValueError('partial occupancy requires an explicit ordered model')
        label=row.get('_atom_site_label',row.get('_atom_site_type_symbol',''))
        kind=row.get('_atom_site_type_symbol',re.match(r'[A-Za-z]+',label).group(0) if re.match(r'[A-Za-z]+',label) else label)
        if label in force_field: ff=force_field[label]
        elif kind in force_field: ff=force_field[kind]
        else: raise ValueError(f'no explicit force-field parameters for {label} ({kind})')
        sigma,epsilon=float(ff['sigma']),float(ff['epsilon'])
        if charge_source=='cif':
            if '_atom_site_charge' not in row: raise ValueError(f'{label}: missing CIF charge; choose force_field explicitly if desired')
            q=number(row['_atom_site_charge'])
        elif charge_source=='force_field': q=float(ff.get('charge',0.0))
        else: raise ValueError('charge_source must be cif or force_field')
        fractional=np.array([number(row['_atom_site_fract_'+k]) for k in 'xyz'])
        images=[]
        for rot,shift in ops:
            frac=np.mod(rot@fractional+shift,1.0)
            if any(np.linalg.norm((frac-other)-np.rint(frac-other))<1e-7 for other in images): continue
            # Duplicate different sites suggest disorder/duplicate input, not
            # extra physically occupied atoms.
            if any(np.linalg.norm((frac-other)-np.rint(frac-other))<1e-7 for other in fractions):
                raise ValueError('distinct CIF atom sites overlap after symmetry expansion')
            images.append(frac)
        for frac in images:
            fractions.append(frac); parameters.append((sigma,epsilon,q)); labels.append(label)
            if 'mass' in ff: mass+=float(ff['mass'])
            else: has_mass=False
    rep=np.asarray(replicas,dtype=int)
    if rep.shape!=(3,) or (rep<1).any() or not np.array_equal(rep,np.asarray(replicas)):
        raise ValueError('replicas must be three positive integers')
    positions=[]; expanded=[]; expanded_labels=[]
    fractions=np.array(fractions)
    for offset in itertools.product(*(range(int(r)) for r in rep)):
        positions.extend((fractions+offset)@cell); expanded.extend(parameters); expanded_labels.extend(labels)
    final_cell=np.diag(rep)@cell; validate_cell(final_cell,cutoff)
    params=np.array(expanded,float)
    return Framework(final_cell,np.array(positions),params[:,0],params[:,1],params[:,2],expanded_labels,
                     mass*int(np.prod(rep)) if has_mass else None,str(path.resolve()))

def automatic_replicas(cell,cutoff):
    _,_,heights=validate_cell(cell)
    return np.maximum(1,np.floor(2*cutoff/heights).astype(int)+1)

def write_xyz(path,entries,species,comment='PoreMC unwrapped molecular coordinates'):
    rows=[]
    for s,x in entries:
        for p in x: rows.append(f'{species[s].name} {p[0]:.12g} {p[1]:.12g} {p[2]:.12g}')
    Path(path).write_text(str(len(rows))+'\n'+comment+'\n'+'\n'.join(rows)+'\n')

def load_json(path): return json.loads(Path(path).read_text())
