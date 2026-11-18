# CPU reciprocal algorithms in 0.7

The direct Ewald Hamiltonian, real-space potential, and finite wave-vector set are the same across the measured cases. A site at fractional coordinate `f` contributes a separable tensor `q X[h] Y[k] Z[l]`, where `X[h]=exp(2πih fx)` and similarly for the other two axes. Harmonics are generated in binary64 from the three fundamental phases.

## Molecular structure factors

The structure factor is a sum of rank-one site tensors. Production stores the configured wave-vector set as runs of consecutive `l` for each `(h,k)`. Splitting runs also handles the missing origin, skew-cell radial masks, and zero-width reciprocal axes. Site harmonics are prepared in tiles of up to sixteen input atoms. Each output run stays hot while the tile is accumulated. Contiguous complex explicit-real arithmetic allows SIMD across its last dimension. The implementation uses ordinary double precision, with fast-math and FP contraction disabled in the provided build.

## One-site CBMC with an arbitrary grown prefix

For B≥4 one-site CPU trials define `A_k=c_k(S_current−S_excluded+S_prefix)`. All B trials share `A`. For the candidate position multiply the tensor `conj(A)` by the three short harmonic vectors. Contract first along `l`, and then multiply by `X[h]Y[k]` and accumulate. The reciprocal self part of this trial equals `q² Σ c_k` exactly in real arithmetic; configure this scalar once. Self and uniform background increments are then added to the constant part. Each tested position avoids writing and reading a full candidate structure factor. Trial parallelism remains independent when more than one thread is requested.

## Full molecular replacement

For `D=S_new−S_old` the reciprocal increment against the accepted state is `Σ c_k(2 Re(conj(S_current)D)+|D|²)`. Both self-energy and the net-charge/background difference enter once, making the routine valid even with a change in net charge or charge-square sum. For a rigid translation the code multiplies the cached molecular structure factor by the separable translation phase directly into its output in one pass. The presence of a changed energy in response to rotation does not imply a change in the number of modes.

## Reproduction

From the root of the source tree:

    python tools07/benchmark_cpu_fourier.py --cpu 7 --repeats 5
    python -m pytest -q tests07/test_cpu_fourier.py tests06/test_reciprocal_native.py

For `--cpu`, specify an available CPU index in your job's allowed affinity. The 0.5 source is bundled under `baseline_05/native`. The unmodified 0.6 source defaults to the bundled `baseline_06/native`. To compare an alternate 0.6 checkout, point `--baseline06` at its project root. The harness uses ~5 resident molecules and fixed synthetic positions to isolate per-kernel costs. Each process uses a fresh reciprocal state. Five repetitions interleave the three builds; per-row times and ratios are recorded in `cpu_fourier.json`. Cross-version checksums include mode-index-specific diagnostics and thus depend on the stored representation even when the physical energy is equal. Energy correctness is checked separately in `cpu_fourier_audit.json` and `cpu_reciprocal_audit.json`.

For the measured case the 0.6 half spectrum was faster than 0.5 in reciprocal subroutines on this workstation. Attributing the real-system HPC exact-path difference requires paired runtime profiles on the actual HPC machine. The production GCMC rates include host/guest short-range energies, proposal and commitment overhead, resident molecule counts, and the distribution of move types; standalone reciprocal rates measure a different quantity.
