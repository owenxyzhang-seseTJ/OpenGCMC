"""Content-addressed preprocessing cache. NumPy arrays only; no pickle/code loading."""
from __future__ import annotations
from pathlib import Path
import zipfile
import hashlib
import json
import os
import tempfile
import numpy as np

SCHEMA = 'poremc-precompute-v2'

def content_key(metadata: dict, arrays: dict[str, np.ndarray]) -> str:
    h=hashlib.sha256(json.dumps({'schema':SCHEMA, **metadata},sort_keys=True,separators=(',',':'),allow_nan=False).encode())
    for name,value in sorted(arrays.items()):
        a=np.ascontiguousarray(value)
        if a.dtype.hasobject: raise TypeError('object arrays cannot be cached')
        h.update(name.encode());h.update(a.dtype.str.encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

class ArrayCache:
    """Atomic, independently verifiable, cross-run immutable array artifacts."""
    def __init__(self,directory):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
    def get(self,key):
        path=self.directory/(key+'.npz')
        if not path.exists(): return None
        try:
            with np.load(path,allow_pickle=False) as f:
                manifest=json.loads(str(f['__manifest__']))
                arrays={k:np.array(f[k],copy=True) for k in f.files if k!='__manifest__'}
            if manifest.get('key')!=key or manifest.get('schema')!=SCHEMA: return None
            if manifest.get('payload')!=content_key({},arrays): return None
            for a in arrays.values():a.flags.writeable=False
            return arrays
        except (OSError,ValueError,KeyError,EOFError,zipfile.BadZipFile):return None
    def put(self,key,arrays):
        manifest={'schema':SCHEMA,'key':key,'payload':content_key({},arrays)}
        fd,tmp=tempfile.mkstemp(prefix=key+'.',suffix='.tmp',dir=self.directory)
        try:
            with os.fdopen(fd,'wb') as f:
                np.savez(f,__manifest__=json.dumps(manifest,sort_keys=True),**arrays)
                f.flush();os.fsync(f.fileno())
            os.replace(tmp,self.directory/(key+'.npz'))
        finally:
            if os.path.exists(tmp):os.unlink(tmp)

def host_key(exact,kind,options):
    return content_key({'kind':kind,'cutoff':exact.cutoff,'alpha':exact.alpha,
                        'coulomb':exact.coulomb,'shifted':exact.shifted,**options},
        {'cell':exact.cell,'host':exact.host_positions,'sigma':exact.host_sigma,
         'epsilon':exact.host_epsilon,'charge':exact.host_charge})
