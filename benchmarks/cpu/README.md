# CPU sampling and execution benchmarks

The benchmark pairs use Apple M5 and an Intel Xeon CPU Max 9468 HPC node. Time to precision is measured at the earliest retained checkpoint where every registered observable meets its Monte Carlo standard error target and rank-normalized split R̂ < 1.01.

## Results

| Paired experiment | Apple M5 | Intel Xeon HPC |
| --- | ---: | ---: |
| IRMOF-1 CO₂/CH₄, shared proposals, single-stage / delayed | 3.88 / 3.17 s | 3.00 / 2.34 s |
| UiO-66 CO₂, shared proposals, single-stage / delayed | 18.19 / 49.28 s | 38.09 / 98.05 s |
| IRMOF-1 CO₂/CH₄, OpenGCMC / RASPA3 CBMC | 1.12 / 10.03 s | 3.35 / 20.80 s |
| TON n-butane, 2 Pa, OpenGCMC / RASPA3 CBMC | 7.62 / 105.82 s | 9.83 / 91.95 s |

The two-pore enumeration resolves occupancy and conformation distributions at identical total-loading probabilities. In the UiO-66 comparison with shared proposals, delayed acceptance saved exact evaluations and increased loading autocorrelation. The frozen two-mode rule used a different baseline with exact evaluation of each candidate.

At 0.02 Pa p-xylene, OpenGCMC reached the target and RASPA3 hit its run budget. Both programs have right-censored times at 0.50 and 40 Pa p-xylene and at 100 Pa n-butane. The expanded-ensemble CFCMC records preserve integer counts and bias outputs.

| Material | Data and code |
| --- | --- |
| Detailed experimental report in Chinese | [PDF](report/cpu_experiment_report.pdf) · [Editable Word](report/cpu_experiment_report.docx) · [Figure source](report/plot_report.py) |
| Double-pore, NIST, RASPA3, holdout, and cold and warm caches | [Experimental notes](paired/README.md) · [516 checkpoint observations](paired/checkpoints.csv) · [Model inputs](paired/inputs/) · [M5](paired/m5/) · [HPC](paired/hpc/) |
| Incremental CPU kernels and MFI | [Experiments](cpu_kernels_and_workflows.md) · [M5 kernels](cpu_kernels_m5.json) · [MFI inputs](../inputs/mfi/) |
| Scientific case studies | [Models, samples, timing and reproduction](scientific/README.md) |
| MCP and Slurm | [M5 dispatch record](dispatch_m5.json) · [M5 samples](dispatch/m5/) · [HPC samples](paired/hpc/live_slurm/) |
| External data and force fields | [NIST inputs and terms](reference/README.md) · [RASPA3 license](reference/RASPA3_LICENSE.txt) |

The archive preserves the per-chain NPZ samples, original RASPA3 trajectory contents in `.pdb.gz`, runs' model inputs, restart coordinates in JSON, timing records, and validation outputs. The paired benchmarks contain 36 M5 and 32 HPC coordinate trajectories. Supplemental RASPA3 checks add three coordinate files. Filesystem paths and scheduler identifiers are normalized for publication. Runtime databases, compiled libraries, and materialized caches can be reconstructed from the inputs.

## Reproduce the analysis

Run the commands from the repository root. Use a fresh output directory for new runs.

```bash
python -m pip install -e '.[report]'
python benchmarks/cpu/paired/export_checkpoints.py benchmarks/cpu/paired checkpoints_reproduced.csv
python benchmarks/cpu/report/plot_report.py
```

The figure source uses Matplotlib and CJK fonts. Generating the editable report also requires `python-docx`, `lxml`, and the MathML-to-OMML stylesheet supplied with Microsoft Word. Set `OPENGCMC_MATHML_XSLT` if the stylesheet is installed in a different location.

```bash
python -m pip install python-docx lxml
python benchmarks/cpu/report/build_report.py
```

## Re-run the native or RASPA3 experiments

```bash
python native/build.py --portable --no-openmp
python benchmarks/cpu/paired/run_cpu_suite.py controls --output benchmarks/runs/cpu_controls --cache benchmarks/runs/cpu_fields
python benchmarks/cpu/paired/run_cpu_suite.py irmof_mixture --output benchmarks/runs/irmof_mixture --cache benchmarks/runs/cpu_fields --raspa3 /path/to/raspa3
```

The whole [CPU suite](paired/run_cpu_suite.py) and [frozen holdout protocol](paired/blind_protocol.json) specify individual or paired runs. [The Slurm launcher](paired/hpc_cpu.sbatch) takes an absolute staging directory with `OpenGCMC/`, the Python and RASPA3 executables, and the absolute study directory `benchmarks/cpu/paired`. For follow-up jobs select the relevant group index in the launcher array.
