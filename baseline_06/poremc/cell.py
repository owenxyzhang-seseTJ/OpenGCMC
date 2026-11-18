"""Periodic cell geometry with row lattice vectors and explicit cutoff bounds."""
import numpy as np

def validate_cell(cell, cutoff=None):
    h = np.array(cell, dtype=float, copy=True)
    if h.shape != (3,3) or not np.isfinite(h).all() or np.linalg.det(h) <= 0:
        raise ValueError("cell must be finite right-handed nonsingular 3x3")
    inverse = np.linalg.inv(h)
    heights = 1.0 / np.linalg.norm(inverse, axis=0)
    if cutoff is not None and (not np.isfinite(cutoff) or cutoff <= 0 or 2*cutoff >= heights.min()):
        raise ValueError(f"cutoff {cutoff} must be < half minimum face height {heights.min()/2}")
    return h, inverse, heights

def wrap(positions, cell):
    h,inv,_ = validate_cell(cell)
    return np.mod(np.asarray(positions) @ inv, 1.0) @ h

def cutoff_image(displacements, cell):
    """For finite-range interactions subject to strict face-height cutoff.

    Fractional rounding is NOT a general nearest lattice vector solver in a
    skew cell. Under r_cut < min(face heights)/2 every in-cutoff image has all
    fractional coordinates within (-1/2,1/2), so no contributing pair is lost.
    Use closest_image for unrestricted geometric distances.
    """
    h,inv,_ = validate_cell(cell)
    s=np.asarray(displacements)@inv
    return (s-np.rint(s))@h

def closest_image(displacement, cell):
    """Exact closest lattice vector, bounded by reciprocal face heights.

    Mainly for analysis; hot short-range kernels use the cutoff guarantee.
    """
    import itertools
    h,inv,_=validate_cell(cell)
    d=np.asarray(displacement,float)
    if d.shape!=(3,):
        return np.asarray([closest_image(x,h) for x in d.reshape(-1,3)]).reshape(d.shape)
    s=d@inv; n0=np.rint(s); best=(s-n0)@h; r=np.linalg.norm(best)
    bounds=r*np.linalg.norm(inv,axis=0)+1e-12
    lo=np.ceil(s-bounds).astype(int); hi=np.floor(s+bounds).astype(int)
    for n in itertools.product(*(range(lo[i],hi[i]+1) for i in range(3))):
        v=(s-n)@h
        if v@v < best@best: best=v
    return best

def cell_from_parameters(a,b,c,alpha=90,beta=90,gamma=90):
    alpha,beta,gamma=np.deg2rad([alpha,beta,gamma])
    ca,cb,cg=np.cos([alpha,beta,gamma]); sg=np.sin(gamma)
    if min(a,b,c)<=0 or abs(sg)<1e-12: raise ValueError("invalid cell parameters")
    h=np.array([[a,0,0],[b*cg,b*sg,0],[c*cb,c*(ca-cb*cg)/sg,0]],float)
    square=c*c-h[2,0]**2-h[2,1]**2
    if square<=0: raise ValueError("invalid cell angles")
    h[2,2]=np.sqrt(square)
    return validate_cell(h)[0]
