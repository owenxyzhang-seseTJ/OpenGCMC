"""Read the uploaded, byte-preserved CUDA benchmark input format."""
from pathlib import Path
import struct
import numpy as np

def read_fixture(path):
    with open(path,'rb') as f:
        b,a,m,rc,alpha,shifted,mode=struct.unpack('<qqqddii',f.read(48))
        data=np.frombuffer(f.read(),dtype='<f8').copy()
    expected=3*b*a+3*a+6*m+18
    if len(data)!=expected:raise ValueError('fixture length mismatch')
    offset=0
    def take(n,shape=None):
        nonlocal offset
        v=data[offset:offset+n];offset+=n
        return v.reshape(shape) if shape is not None else v
    xyz=take(3*b*a,(b,a,3));ts=take(a);te=take(a);tq=take(a)
    frac=take(3*m,(m,3));hs=take(m);he=take(m);hq=take(m);cell=take(9,(3,3));inv=take(9,(3,3))
    return {'xyz':xyz,'ts':ts,'te':te,'tq':tq,'host':frac@cell,'hs':hs,'he':he,'hq':hq,
            'cell':cell,'cutoff':rc,'alpha':alpha,'shifted':bool(shifted),'coulomb':('none','dsf','ewald-real')[mode]}

def env_args(d):
    return (d['host'],d['hs'],d['he'],d['hq'],d['cell'],d['cutoff'])
def env_kwargs(d):return {k:d[k] for k in ('alpha','shifted','coulomb')}
