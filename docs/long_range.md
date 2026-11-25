# Native long-range sampling

The C++ sampler handles mixtures of rigid and tree-flexible guests in a fixed periodic framework. It supports insertion, deletion, translation, rotation and complete configurational-bias regrowth. For charged guests, choose finite direct Ewald or particle-mesh Ewald (PME) when constructing the classical potential and sampler.

## Input and execution

After building the native library, run the charged flexible-chain example from the source directory.

```bash
opengcmc run examples/longrange/small_delayed_ewald.json -o output/delayed
```

The demonstration uses a four-site flexible chain, a two-site framework, six trials per growth stage, 100 warm-up attempts and 600 production attempts. The output includes thermodynamic statistics, a final configuration and a checkpoint. Results report the actual reciprocal backend and both acceptance stages.

| Example | Model and sampler |
|---|---|
| [Rigid Ewald](../examples/longrange/rigid_ewald.json) | Rigid guest with direct Ewald |
| [Rigid PME](../examples/longrange/rigid_pme.json) | Rigid guest with a PME target |
| [Flexible Ewald](../examples/longrange/flexible_ewald.json) | Tree-flexible guest with direct Ewald |
| [Flexible delayed Ewald](../examples/longrange/small_delayed_ewald.json) | Fixed-host reference grid with a direct Ewald target |
| [Flexible Ewald with CUDA](../examples/longrange/flexible_ewald_cuda.json) | Direct Ewald reciprocal candidate batches on an NVIDIA device |

For a charged fixed framework, the following fields can be placed in the input's existing sampling object.

```json
{
  "driver": "cpp",
  "sampling_mode": "delayed",
  "reciprocal_method": "direct",
  "reciprocal_backend": "cpu",
  "ewald_half_spectrum": true,
  "ewald_k_cutoff": 0.0,
  "host_grid_mesh": [64, 64, 64],
  "host_grid_steep": "boltzmann",
  "host_grid_cap": 100000.0,
  "host_grid_cache": "cache/framework.bin",
  "grid_root_bias": 0.0
}
```

Coordinates use Å, charges use elementary charge, and energies use K for $U/k_\mathrm{B}$. The direct Ewald target uses three-dimensional conducting boundaries with real-space, reciprocal, self and intramolecular corrections. A net charge requires an explicitly enabled uniform neutralizing background.

| Field | Effect |
|---|---|
| `sampling_mode` | `exact` evaluates the target for each candidate. `reference-mh` and `delayed` share grid CBMC proposals and use one-stage and two-stage acceptance, respectively. |
| `host_grid_mesh` | Number of grid points on each axis. Ewald grids use powers of two. The benchmark includes 32³ and 64³ meshes. |
| `host_grid_steep` | `boltzmann` uses thermal interpolation in steep cells. `linear` uses linear energy interpolation in those cells. |
| `host_grid_cap` | Finite energy cap for the reference field in K. Zero enables the exact framework fallback in steep cells. |
| `guest_update` | `auto` uses the incremental guest index for cell or cached kernels and the direct rebuild path for direct kernels. The explicit options are `incremental` and `rebuild`. |
| `partial_regrowth` | Choose a flexible suffix, retaining the same prefix in forward growth and retracing. |
| `host_field_cache` | Directory for separate raw LJ fields and a unit-charge electrostatic field. Compatible species and runs share the same immutable files. |
| `pme_refresh` | Maximum number of accepted charged changes between FFT refreshes. The default is 1. Sparse corrections preserve the chosen mesh energy. |
| `host_grid_cache` | Optional file for the raw framework fields. Relative paths resolve against the JSON file's directory in the CLI and against the working directory in Python. |
| `grid_root_bias` | Weight of the field-derived root proposal for delayed sampling. The remaining weight provides uniform support. The bias and an explicit root proposal are mutually exclusive. |
| `ewald_half_spectrum` | Stores only one of each conjugate wave-vector pair with the corresponding doubled coefficient. |
| `ewald_k_cutoff` | Zero preserves the full configured rectangular set. A positive radius in Å⁻¹ changes the finite direct Ewald target. |
| `pme_mesh` | Dimensions of the PME mesh for `reciprocal_method="pme"`. Each dimension is a power of two of at least eight. |

## Reference-field CBMC

The sampler tabulates framework LJ interactions for each unique guest parameter type and shares unit-charge static electric fields. Guest–guest short-range interactions use the accepted-state cache. Stage one grows and retraces candidates in the reference potential $U_0$. The acceptance ratio $r_0$ includes reservoir terms, move probabilities, Rosenbluth weights and any root proposal density. Stage two evaluates the selected complete configuration using the energy residual $R = U - U_0$.

### Forward proposals, retracing and acceptances

For each growth step $i$, let $g_i$ be the trial density relative to the normalized configurational measure at that prefix. The root measure is uniform over the box and the orientation measure is normalized. The rigid sampler treats a whole pose as one growth step. The flexible sampler draws coordinates in the molecule's specified internal reference measure. The trial weights and the Rosenbluth factor are

$$ w_{ij}=\exp[-\beta\,\Delta_i U_0(x_{ij})]/g_i(x_{ij}),\qquad
\overline w_i=\frac1{K_i}\sum_j w_{ij},\qquad W=\prod_i\overline w_i . $$

Native geometry trials have unit relative density except for an explicitly biased root, whose relative density is $V$ times its spatial density. Forward growth selects trial $j$ with probability $w_{ij}/\sum_l w_{il}$. The reverse retrace contains the actual coordinates and $K_i-1$ fresh trials using the same prefix and base law. The forced trial can occupy any slot by exchangeability. When the slot is coupled uniformly, its $1/K_i$ probability cancels the corresponding normalization in the forward selection. The unselected trial densities cancel on the augmented forward and reverse paths. The remaining growth-to-retrace proposal factor is $\exp[-\beta\Delta U_0]/W$ relative to the normalized pose measure.

For species $s$, write $N_s$ for the count before the move, $z_s$ for its normalized activity, $V$ for the volume and $p_+$ and $p_-$ for the insertion and deletion move probabilities. When using fugacity input, $z_s=f_s/(k_BT q_{\mathrm{int},s})$ in consistent volume units, with $q_{\mathrm{int},s}$ evaluated in the same normalized internal measure used for growth. State-independent species selection cancels. The reverse of an insertion chooses the added molecule with probability $1/(N_s+1)$. Starting from a deletion, the corresponding probability is $1/N_s$. The reference ratios are

$$
r_{0,+}=\frac{p_-}{p_+}\frac{z_sV}{N_s+1}W_{\rm new},
\qquad
r_{0,-}=\frac{p_+}{p_-}\frac{N_s}{z_s V W_{\rm old}} .
$$

For regrowth the same molecule is selected in both directions. A suffix start index is drawn independently of the conformation, and the two paths retain the same prefix. The Rosenbluth products run over the selected suffix and give $r_{0,\rm regrow}=W_{\rm new}/W_{\rm old}$. Symmetric translation and rotation give $r_0=\exp[-\beta(U_0(Y)-U_0(X))]$.

For all of these moves, the target ratio on the same augmented path is

$$ r=r_0 r_R,\qquad r_R=\exp\{-\beta[(U(Y)-U_0(Y))-(U(X)-U_0(X))]\}. $$

An insertion adds the selected molecular residual, a deletion removes it and a regrowth subtracts the old residual from the new one in the same environment. The implementation cancels guest–guest short-range terms shared by $U$ and $U_0$. The remaining terms include the exact framework correction and the reciprocal contribution, including whichever self and exclusion terms were omitted from the reference. Cached quantities follow the accepted state, so rejected trials leave those quantities unchanged.

The single-stage acceptance is $\min(1,r_0r_R)$. The delayed acceptance is $\min(1,r_0)\min(1,r_R)$. Both forward-to-reverse acceptance ratios are $r$. Therefore detailed balance holds after averaging over the auxiliary trials for either rule. A surrogate or proposal must retain support for the target moves. Detailed balance imposes no lower bound on mixing speed.

For the same trial path, $\min(1,r_0)\min(1,r_R)\leq\min(1,r_0r_R)$. The difference is positive when the reference and residual favour opposite decisions and neither ratio equals one. The delayed rule saves selected-configuration evaluations through the first-stage screen. Its cost advantage depends on those savings relative to the increase in correlation of the observable of interest. The `exact` path changes both candidate energies and acceptance and is a separate diagnostic comparison.

The classical CPU pair kernels accept `potential.shifted: "force"` for
$u(r)-u(r_c)-(r-r_c)u'(r_c)$ at $r<r_c$. Boolean shift settings continue to select cut or energy-shifted LJ.

$$
\alpha_1=\min(1,r_0),\qquad
\alpha_2=\min\left(1,\exp[-\beta\,\Delta R]\right).
$$

The product of the forward-to-reverse ratios recovers the target acceptance ratio. The terms common to $U$ and $U_0$ cancel in the residual. Acceptance commits the new guest configuration and reciprocal state together. Rejections retain the accepted state.

The periodic grid uses tricubic interpolation in regular regions. In steep cells the default interpolates Boltzmann factors with trilinear weights $w_i$. For energies $u_i = U_i/k_\mathrm{B}$ and temperature $T$, the interpolated reference energy is

$$
u_0(\mathbf r)=-T\log\left(\sum_i w_i(\mathbf r)\exp[-u_i/T]\right).
$$

Grid resolution can change acceptance and occupancy mixing even when the sampled Hamiltonian is fixed. The disk file stores temperature-independent fields. Matching physical and mesh parameters permit reuse across temperatures, activities and independent chains. Writes are staged in the same directory and atomically published once the payload is complete.

## CPU and CUDA reciprocal evaluation

A site at fractional position $(f_x,f_y,f_z)$ contributes the product of axis phases to the structure factor. The CPU direct Ewald sampler groups wave vectors with the same first two indices into contiguous runs along the third index. These runs support harmonic contraction for one-site growth trials and complete molecular replacements. For $D_\mathbf{k}=S_{\mathrm{new},\mathbf{k}}-S_{\mathrm{old},\mathbf{k}}$, the reciprocal replacement increment is

$$
\Delta U_\mathrm{recip}=\sum_\mathbf{k} c_\mathbf{k}
\left(2\,\mathrm{Re}[\overline{S_\mathbf{k}}D_\mathbf{k}]+|D_\mathbf{k}|^2\right).
$$

The accepted state supplies $S_\mathbf{k}$. A rigid translation multiplies the cached molecular factor by $e^{i\mathbf{k}\cdot\Delta\mathbf r}$. The conjugate half spectrum retains the finite Ewald sum. PME uses its specified mesh Hamiltonian with either the native CPU or the CUDA PME backend.

The native CUDA plug-in evaluates reciprocal candidates in FP64 for direct Ewald. With `reciprocal_backend="cuda-tensor"`, exact sampling batches candidate energies. Delayed sampling computes the selected molecular structure factor on the device, then applies the residual and commits the state on the CPU. The CPU runs proposals and accept/reject decisions. The native short-range backend is selected independently.

Use `reciprocal_backend="cuda-resident"` with delayed direct Ewald to keep the accepted spectrum on the device. The device evaluates the reciprocal energy increment from the signed charges of the selected new and old molecular sites. It returns one FP64 energy and applies the staged increment after acceptance. The CPU retains its reference state for energy reporting and checkpoint validation. Checkpoints include the actual device spectrum. The host ABI test uses a CPU model of the CUDA interface.

For the discrete PME Hamiltonian, `pme_refresh` controls how often accepted charge increments are merged into the FFT reference mesh. Until a refresh, every candidate includes the Green-operator cross term with all pending accepted increments. The pending state is saved with the last transformed mesh. Choose the refresh interval using the site support and acceptance rate of the workload.

Select `reciprocal_method="pme"` and `reciprocal_backend="cuda-pme"` to run the FP64 mesh refresh in cuFFT and evaluate batched cubic B-spline candidates on CUDA. The GPU retains the refreshed potential mesh and the periodic Green function. The host retains the accepted charge mesh and the sparse corrections required by exclusions, partial regrowth and deferred refresh. Single trials of at most two sites use the host energy path. Exact, delayed and reference-MH modes share the same PME Hamiltonian and checkpoint state. `gpu_fft_transforms` and `gpu_batches` report actual device work. The CUDA build requires the cuFFT library from the CUDA toolkit.

```bash
python native/build.py --cuda --cuda-arch sm_86
python tools/longrange/validate_tensor_gpu.py --input system.json --output output/tensor_device_results.json
```

Set the architecture to match the installed GPU. The device validator requires a compatible NVIDIA driver and CUDA toolkit. `long_range_profile` reports the resolved backend, mode counts, device and candidate batches. `acceleration_profile` reports grid reuse and acceptance-stage counters.

## Native CUDA short-range batches

Enable the FP64 real-space path in the C++ sampler with these `sampling` options.

```json
{
  "driver": "cpp",
  "short_range_backend": "cuda",
  "short_range_min_batch": 16
}
```

The CUDA plug-in produced by `native/build.py --cuda` contains the real-space and reciprocal kernels. `cuda_library` can select a particular build. The real-space path supports Lennard-Jones interactions, potential and force shifting, damped shifted-force Coulomb and the real term of Ewald. It uses the same cutoff, periodic geometry and molecular exclusions as the native CPU path. The framework and, when available, its cached neighbour plan stay on the device. Accepted guest positions are uploaded on the next eligible query after an accepted change.

The minimum batch size counts candidate configurations. Calls below that size use the native CPU pair evaluator. In delayed sampling, a threshold of 1 also offloads single-candidate exact framework corrections. Capped packed reference grids stay on CUDA and share a kernel with eligible guest pair batches, using the same cubic, linear or Boltzmann interpolation as the CPU. Uncapped grids with exact fallback and grids above the packed-field cache budget use CPU interpolation. With an empty guest environment, CUDA interpolates batches of at least 64 trial sites that meet the configuration threshold. Accepted guest changes refresh the affected range of the resident snapshot. `short_range_profile` reports framework and guest batches, reference-grid lookups and guest upload bytes. Choose the batch threshold using blocking-call latency and sampling throughput on the intended workload and device.

## Reproduction and source

```bash
python tools/longrange/benchmark_input.py system.json \
  --modes exact,delayed32,delayed64 --steps 10000 --burnin 1000 --repeats 3 \
  --output output/longrange_benchmark.json
python tools/longrange/mixing_assessment.py \
  --input cpu=output/longrange_benchmark.json --output output/count_mixing.json
python tools/longrange/audit_host_cache.py --input system.json \
  --workdir output --output output/host_grid_check.json
```

The benchmark alternates modes in separate single-threaded processes and records preparation and production times, count traces and a direct target-energy recomputation. Use the counts and accepted-move rates to interpret any throughput change. Count ESS describes the supplied observable and requires adequate trajectory length. The cache audit compares no-cache, cold-write and warm-read runs with the same seed.

The [native source files](../native) implement the reference fields and cache in `host_grid.cpp`, growth and acceptance in `fast_mc.cpp`, and reciprocal evaluation in `ewald_state.cpp` and `ewald_cuda.cu`. See the [numerical validation records](../benchmarks/results/longrange) and the [charged-MOF benchmark](benchmarks.md#charged-p-xylene-in-ni_tpt_tatba).
