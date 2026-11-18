#!/usr/bin/env python3
"""Package the PoreMC 0.6 source, compiled report and recorded validation data."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.6.0"
DATE = "2026-11-18"
SKIP_DIRS = {
    "__pycache__", ".pytest_cache", ".git", "build", "qa", "qa_final",
    "grid_cache", "cache",
}
SKIP_SUFFIXES = {
    ".pyc", ".pyo", ".so", ".dylib", ".dll", ".o", ".a", ".fmt",
    ".aux", ".toc", ".out", ".cubin", ".ptx", ".tmp",
}
SKIP_FILES = {
    "release_manifest_05.json", "release_manifest_06.json", "MANIFEST.sha256",
    "evidence06/regression.log", "evidence06/regression.xml",
    "evidence06/checkpoint_regression.xml", "evidence06/hostgrid_pilot.json",
    "evidence06/hostgrid_pilot.log", "evidence06/flexible_pilot_final.json",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def included(path):
    rel = path.relative_to(ROOT)
    return (
        path.is_file()
        and rel.as_posix() not in SKIP_FILES
        and not any(p in SKIP_DIRS or p.endswith(".egg-info") for p in rel.parts)
        and path.suffix not in SKIP_SUFFIXES
    )


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT.parent / "poremc06_deliverables"))
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    evidence = ROOT / "evidence06"

    def data(name):
        return json.loads((evidence / name).read_text())

    testlog = (evidence / "all_tests.log").read_text()
    match = re.search(r"(\d+) passed, (\d+) subtests passed", testlog)
    if not match or re.search(r"[1-9]\d* (?:failed|error)", testlog):
        raise RuntimeError("A successful final regression log is required.")
    passed, subtests = map(int, match.groups())
    physics = data("delayed_physics_audit.json")
    rigid = data("hostgrid_benchmark.json")
    flexible = data("flexible_benchmark.json")
    native_hash = physics["native_library_sha256"]
    if not all(check["passed"] for check in physics["checks"].values()):
        raise RuntimeError("Independent physics audit did not pass.")
    for benchmark in (rigid, flexible):
        for row in benchmark["rows"]:
            if row["mode"] != "baseline05" and row["native_sha256"] != native_hash:
                raise RuntimeError("Benchmark and physics-audit binaries differ.")
    built_library = ROOT / "poremc/libporemc.so"
    if built_library.exists() and digest(built_library) != native_hash:
        raise RuntimeError("The present native binary differs from the audited binary.")
    source_hash = re.search(
        r"SHA256:\s*([0-9a-f]{64})", (ROOT / "baseline_05/PROVENANCE.md").read_text()
    ).group(1)
    proof = {
        "version": VERSION,
        "date": DATE,
        "energy_unit": "U/k_B in kelvin",
        "native_library_sha256": native_hash,
        "input_05_archive_sha256": source_hash,
        "new_gpu_execution_performed": False,
        "scope": {
            "performance": "Paired single-thread CPU measurements against the unmodified PoreMC 0.5 source.",
            "inputs": "Public synthetic fixtures match the user-reported workload dimensions; they are not the user's original CIF or p-xylene force field.",
            "hamiltonian": "Main benchmarks preserve the full 19682-mode finite rectangular direct Ewald target; 0.6 stores its conjugate half spectrum.",
            "grid": "32^3 and 64^3 are host proposal grids, not PME meshes in these benchmarks.",
            "timing": "Production timing excludes initialization and final independent energy recomputation. Each variant has three seeds; reported speedup is the median of paired ratios.",
            "mixing": "Long-chain replicas ran concurrently. Their timings are not used for performance comparisons or ESS/s.",
            "raspa": "No new RASPA run or measured RASPA/PoreMC ratio is claimed. The separate user_hpc_context is user-reported.",
            "cuda": "The existing exact-candidate CUDA path is retained. The new delayed correction runs on the native CPU.",
        },
        "regression": {
            "passed": passed,
            "parameter_subtests": subtests,
            "junit_sha256": digest(evidence / "all_tests.xml"),
            "test_directories": ["tests", "tests_legacy", "tests03", "tests04", "tests06"],
        },
        "independent_delayed_physics": physics,
        "independent_reciprocal_physics": data("reciprocal_audit.json"),
        "independent_host_grid": data("host_grid_module.json"),
        "rigid_complete_loop_benchmark": rigid,
        "flexible_complete_loop_benchmark": flexible,
        "reciprocal_microbenchmark": data("reciprocal_microbenchmark.json"),
        "negative_result_linear_reference": data("linear_screening_benchmark.json"),
        "mixing_diagnostics": data("mixing_diagnostics.json"),
        "long_chain_raw_data": data("occupancy_mixing.json"),
        "offline_reciprocal_advice": data("reciprocal_tuning.json"),
        "user_hpc_context": data("user_hpc_context.json"),
    }

    files = sorted(path for path in ROOT.rglob("*") if included(path))
    manifest = {
        "version": VERSION,
        "date": DATE,
        "input_05_archive_sha256": source_hash,
        "audited_native_library_sha256": native_hash,
        "native_binaries_shipped": False,
        "build_command": "python native/build.py",
        "source_file_count_excluding_manifest": len(files),
        "current_report_directory": "report_06",
        "current_evidence_directory": "evidence06",
        "files": {path.relative_to(ROOT).as_posix(): digest(path) for path in files},
    }
    manifest_path = ROOT / "release_manifest_06.json"
    write_json(manifest_path, manifest)
    files.append(manifest_path)

    archive = out / "PoreMC_0.6_HostGrid_Source.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in files:
            z.write(path, "PoreMC_0.6_hostgrid/" + path.relative_to(ROOT).as_posix())
    latex = out / "PoreMC_0.6_LaTeX_Source.zip"
    with zipfile.ZipFile(latex, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in files:
            rel = path.relative_to(ROOT)
            if rel.parts[0] in ("report_06", "evidence06") and path.suffix != ".pdf":
                z.write(path, rel.as_posix())
        z.writestr(
            "README_LaTeX.md",
            "# PoreMC 0.6 report sources\n\n"
            "Keep report_06/ and evidence06/ adjacent. Build with:\n\n"
            "    python report_06/generate_results.py --require-complete\n"
            "    python report_06/build_report.py\n\n"
            "Requires Python 3.10+, XeLaTeX, fontspec, unicode-math, Noto Sans CJK SC, "
            "Liberation Sans, DejaVu Sans Mono and Latin Modern Math. Recorded data are "
            "included and all numeric tables are generated from them. Compilation does "
            "not rerun simulations. The complete simulation source and tests are in the "
            "separate PoreMC_0.6_HostGrid_Source.zip archive.\n\n"
            "Main performance results are workload-matched synthetic CPU benchmarks, "
            "not a new RASPA comparison or a GPU measurement. See the report for the "
            "negative linear-grid result, corrected Boltzmann proposal, long-chain "
            "mixing checks, and exact-target acceptance derivation.\n",
        )
    pdf = out / "PoreMC_0.6_HostGrid_Report.pdf"
    shutil.copy2(ROOT / "report_06/PoreMC_0.6_HostGrid_Report.pdf", pdf)
    numeric = out / "PoreMC_0.6_Validation_and_Benchmarks.json"
    write_json(numeric, proof)

    for path in (archive, latex):
        with zipfile.ZipFile(path) as z:
            if z.testzip() is not None:
                raise RuntimeError("Archive CRC validation failed.")
    with zipfile.ZipFile(archive) as z:
        prefix = "PoreMC_0.6_hostgrid/"
        for name, expected in manifest["files"].items():
            if hashlib.sha256(z.read(prefix + name)).hexdigest() != expected:
                raise RuntimeError("Source manifest verification failed: " + name)
    summary = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}
        for path in (archive, pdf, latex, numeric)
    ]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
