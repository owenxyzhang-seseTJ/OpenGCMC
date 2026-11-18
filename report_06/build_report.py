#!/usr/bin/env python3
"""Compile the report with XeLaTeX; repair only missing local format caches.

A normal TeX Live/MiKTeX installation needs no fallback. The Linux fallback
handles a present TeX tree whose system filename database was not initialized;
it writes a private format cache under report/build, not to system directories.
"""
from pathlib import Path
import os
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parent
BUILD=ROOT/'build';BUILD.mkdir(exist_ok=True)
env=os.environ.copy()
xelatex=shutil.which('xelatex');xetex=shutil.which('xetex')
if not xelatex:raise SystemExit('Install XeLaTeX / TeX Live with fontspec, unicode-math and Noto CJK fonts.')
def kpse(name):
    result=subprocess.run(['kpsewhich',name],capture_output=True,text=True,env=env)
    return result.stdout.strip() if result.returncode==0 else ''
def run(command,logname):
    result=subprocess.run(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (BUILD/logname).write_text(result.stdout)
    if result.returncode:
        print(result.stdout[-6000:]);raise SystemExit(result.returncode)
base=Path('/usr/share/texlive/texmf-dist')
if not kpse('latex.ltx') and base.exists():
    env['TEXINPUTS']='.:'+str(base/'tex')+'//:'
    env['TFMFONTS']=str(base/'fonts/tfm')+'//:/usr/share/texmf/fonts/tfm//:'
    env['OPENTYPEFONTS']='/usr/share/texmf/fonts/opentype//:'+str(base/'fonts/opentype')+'//:'
    env['T1FONTS']=str(base/'fonts/type1')+'//:/usr/share/texmf/fonts/type1//:'
    env['VFFONTS']=str(base/'fonts/vf')+'//:'
    env['MISCFONTS']=str(base/'fonts/misc')+'//:'
    env['TEXFONTMAPS']=str(base/'fonts/map')+'//:/var/lib/texmf/fonts/map//:'+str(base/'fonts/misc')+'//:'
if not kpse('xelatex.fmt'):
    env['TEXFORMATS']=str(BUILD)+':'
    if not (BUILD/'xelatex.fmt').exists():
        ini=kpse('xelatex.ini') or str(base/'tex/latex/tex-ini-files/xelatex.ini')
        run([xetex,'-ini','-etex','-interaction=nonstopmode','-halt-on-error','-jobname=xelatex',
             '-output-directory='+str(BUILD),ini],'format.log')
for i in (1,2,3):
    run([xelatex,'-interaction=nonstopmode','-halt-on-error','-output-directory='+str(BUILD),'report.tex'],f'pass_{i}.log')
    if i >= 2:
        latest=(BUILD/f'pass_{i}.log').read_text()
        if 'Rerun to get' not in latest and 'Label(s) may have changed' not in latest:
            break
temporary=ROOT/'PoreMC_0.6_HostGrid_Report.pdf.tmp'
shutil.copy2(BUILD/'report.pdf',temporary)
os.replace(temporary,ROOT/'PoreMC_0.6_HostGrid_Report.pdf')
print(ROOT/'PoreMC_0.6_HostGrid_Report.pdf')
