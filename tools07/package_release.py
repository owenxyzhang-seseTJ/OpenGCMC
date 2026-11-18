"""Assemble the 0.7 source, standalone 0.6 writer patch, report and reproducibility archive."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
SKIP_DIRS={'.git','__pycache__','.pytest_cache','build','qa','qa_final','cache','grid_cache'}
EXCLUDED_SOURCE={'MANIFEST.sha256','release_manifest_06.json'}
EXCLUDED_EXTENSIONS={'.pyc','.pyo','.so','.dylib','.o','.dll','.tmp','.aux','.fmt','.cubin','.ptx','.toc','.out'}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def select(path):
    rel=path.relative_to(ROOT)
    return (path.is_file() and rel.as_posix() not in EXCLUDED_SOURCE
            and path.suffix not in EXCLUDED_EXTENSIONS
            and not any(d in SKIP_DIRS or d.endswith('.egg-info') for d in rel.parts))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT.parent/'poremc07_deliverables')
    out=p.parse_args().out.resolve()
    if out.is_relative_to(ROOT):p.error('Use a release directory outside the source tree.')
    out.mkdir(parents=True,exist_ok=True)
    src=out/'PoreMC_0.7_Source.zip'
    repro=out/'PoreMC_0.7_LaTeX_and_Benchmarks.zip'
    pdf=out/'PoreMC_0.7_Report.pdf'
    patch=out/'PoreMC_0.6_HostCache.patch'
    shutil.copy2(ROOT/'report_07'/pdf.name,pdf)
    shutil.copy2(ROOT/'hotfix06'/patch.name,patch)
    files=sorted(f for f in ROOT.rglob('*') if select(f))
    manifest={f.relative_to(ROOT).as_posix():sha(f) for f in files}
    root='PoreMC_0.7/'
    with zipfile.ZipFile(src,'w',zipfile.ZIP_DEFLATED,compresslevel=7) as z:
        for f in files:z.write(f,root+f.relative_to(ROOT).as_posix())
        z.writestr(root+'release_manifest_07.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(repro,'w',zipfile.ZIP_DEFLATED) as z:
        for f in files:
            rel=f.relative_to(ROOT)
            if rel.parts[0] in ('report_07','evidence07') and f.suffix!='.pdf':
                z.write(f,rel.as_posix())
        z.writestr('README.md','# PoreMC 0.7\n\nKeep report_07 and evidence07 together. Generate the tables with '
                   '`python report_07/generate_results.py` and build the PDF with `python report_07/build_report.py`. '
                   'Use XeLaTeX with Noto Sans CJK SC, DejaVu Sans Mono and Latin Modern Math. '
                   'The measurements are included. The separate source archive contains the executable benchmarks, '
                   'cache tests, and the 0.6/0.5 baseline sources.\n')
    with zipfile.ZipFile(src) as z:
        for name,h in manifest.items():
            assert hashlib.sha256(z.read(root+name)).hexdigest()==h,name
    with zipfile.ZipFile(repro) as z:
        assert z.testzip() is None
    for f in (src,patch,pdf,repro):
        print(f.name,f.stat().st_size,sha(f))

if __name__=='__main__':main()
