# Native long-range sampling

The C++ sampler handles one species of rigid or tree-flexible guests in a fixed periodic framework. It supports insertion, deletion, translation, rotation and complete configurational-bias regrowth. For charged guests, choose finite direct Ewald or particle-mesh Ewald (PME) when constructing the classical potential and sampler.

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
| `sampling_mode` | `exact` evaluates the target for each candidate. `delayed` uses a reference potential and a second acceptance stage. |
| `host_grid_mesh` | Number of grid points on each axis. Ewald grids use powers of two. The benchmark includes 32³ and 64³ meshes. |
| `host_grid_steep` | `boltzmann` uses thermal interpolation in steep cells. `linear` uses linear energy interpolation in those cells. |
| `host_grid_cap` | Finite energy cap for the reference field in K. Zero enables the exact framework fallback in steep cells. |
| `host_grid_cache` | Optional file for the raw framework fields. Relative paths resolve against the JSON file's directory in the CLI and against the working directory in Python. |
| `grid_root_bias` | Weight of the field-derived root proposal for delayed sampling. The remaining weight provides uniform support. The bias and an explicit root proposal are mutually exclusive. |
| `ewald_half_spectrum` | Stores only one of each conjugate wave-vector pair with the corresponding doubled coefficient. |
| `ewald_k_cutoff` | Zero preserves the full configured rectangular set. A positive radius in Å⁻¹ changes the finite direct Ewald target. |
| `pme_mesh` | Dimensions of the PME mesh for `reciprocal_method="pme"`. Each dimension is a power of two of at least eight. |

## Reference-field CBMC

The sampler tabulates framework LJ interactions for each unique guest parameter type and shares unit-charge static electric fields. Guest–guest short-range interactions use the accepted-state cache. Stage one grows and retraces candidates in the reference potential $U_0$. The acceptance ratio $r_0$ includes reservoir terms, move probabilities, Rosenbluth weights and any root proposal density. Stage two evaluates the selected complete configuration using the energy residual $R = U - U_0$.

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

The accepted state supplies $S_\mathbf{k}$. A rigid translation multiplies the cached molecular factor by $e^{i\mathbf{k}\cdot\Delta\mathbf r}$. The conjugate half spectrum retains the finite Ewald sum. PME uses its specified mesh Hamiltonian and native CPU FFTs.

The native CUDA plug-in evaluates reciprocal candidates in FP64 for direct Ewald. With `reciprocal_backend="cuda-tensor"`, exact sampling batches candidate energies. Delayed sampling computes the selected molecular structure factor on the device, then applies the residual and commits the state on the CPU. The sampler's short-range work, proposals and accept/reject decisions run on the CPU. The CuPy short-range backend is a separate interface.

```bash
python native/build.py --cuda --cuda-arch sm_86
python tools/longrange/validate_tensor_gpu.py --input system.json --output output/tensor_device_results.json
```

Set the architecture to match the installed GPU. The device validator requires a compatible NVIDIA driver and CUDA toolkit. `long_range_profile` reports the resolved backend, mode counts, device and candidate batches. `acceleration_profile` reports grid reuse and acceptance-stage counters.

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
