"""Independent full-mode and host-grid checks, compiled against production C++."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class NativeReciprocal(unittest.TestCase):
    def test_full_half_radial_fft_and_checkpoint(self):
        if shutil.which("g++") is None:
            self.skipTest("C++ compiler is unavailable")
        with tempfile.TemporaryDirectory(prefix="poremc06-reciprocal-") as directory:
            executable = Path(directory) / "audit"
            subprocess.run([
                "g++", "-O3", "-std=c++17", "-fno-fast-math", "-ffp-contract=off",
                "-I", str(ROOT / "native"), str(ROOT / "tools/longrange/audit_reciprocal.cpp"),
                str(ROOT / "native/ewald_state.cpp"), "-ldl", "-o", str(executable),
            ], check=True, capture_output=True, text=True)
            evidence = json.loads(subprocess.check_output([str(executable)], text=True))
            self.assertEqual(evidence["direct_configurations"], 8)
            self.assertEqual(evidence["kmax13_full_modes"], 19682)
            self.assertEqual(evidence["kmax13_stored_modes"], 9841)
            for key, value in evidence.items():
                if "error_K" in key:
                    self.assertLess(value, 2e-8, key)

if __name__ == "__main__":
    unittest.main()
