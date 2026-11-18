"""Validate the optimized direct paths independently of the Python wrapper."""
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]

def test_live_bath_tensor_and_unequal_charge_replacement():
    with tempfile.TemporaryDirectory(prefix='poremc-cpu-audit-') as directory:
        exe=Path(directory)/'audit'
        subprocess.run(['g++','-std=c++17','-O3','-fno-fast-math','-ffp-contract=off',
                        '-I',str(ROOT/'native'),str(ROOT/'tools07/audit_cpu_fourier.cpp'),
                        str(ROOT/'native/ewald_state.cpp'),'-ldl','-o',str(exe)],check=True)
        result=json.loads(subprocess.check_output([str(exe)],text=True))
    assert result['configurations']==8
    assert result['single_site_candidates']==608
    assert result['max_single_site_error_K']<1e-7
    assert result['max_replacement_error_K']<1e-7
    assert result['max_uncharged_error_K']==0
