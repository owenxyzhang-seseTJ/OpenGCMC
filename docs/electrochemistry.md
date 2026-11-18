# Thermodynamics and electrochemistry

OpenGCMC uses equilibrium Monte Carlo to sample electrolyte composition, molecular configurations and discrete reaction states. Its separate reaction network and diffusion solvers compute current in physical time using supplied kinetic parameters.

| Task | Interface | Example input |
|---|---|---|
| Neutral stoichiometric exchange in a primitive electrolyte | `opengcmc-thermo run` | [2:1 electrolyte](../examples/thermodynamics/electrolyte_21.json) |
| Constant potential with electrode-compensated ion exchange | `opengcmc-thermo run` | [Planar interface](../examples/thermodynamics/interface_constant_potential.json) |
| Chemical-potential response and MBAR | `opengcmc-thermo response`, `opengcmc-thermo mbar` and the [Python API](../poremc/free_energy.py) | [Window and response tools](../tools/thermodynamics) |
| SPC/E hydration and fixed-topology reaction sampling | `opengcmc-reactive run` | [Hydrated interface](../examples/electrochemistry/hydrated_interface.json) and [PCET control](../examples/electrochemistry/pcet_four_state.json) |
| Fixed-composition alchemical calibration | `opengcmc-reactive calibrate` | [Hydrated redox windows](../examples/electrochemistry/hydrated_redox_calibration.json) |
| State kinetics and diffusion currents | `opengcmc-reactive kinetics`, `opengcmc-reactive transport` | [Two-contact current](../examples/electrochemistry/two_contact_current.json), [chronoamperometry](../examples/electrochemistry/diffusion_chronoamperometry.json) and [cyclic voltammetry](../examples/electrochemistry/diffusion_voltammetry.json) |

## Ensembles and units

The primitive electrolyte uses a specified dielectric and monoatomic ions. Neutral channels exchange the specified salt stoichiometry. Single-ion channels pair exchange with compensating electrode charge. Gaussian electrodes impose prescribed potentials and constrain the total charge of the cell. The occupancy and charge landscape includes the conditional Gaussian contribution to electrode fluctuations.

The hydrated interface uses rigid SPC/E water with dielectric 1. Set lengths in Å, energies as `U/kB` in K, charges in units of `e` and electrode potentials in V. The reservoir log activities refer to activity in Å⁻³ with a 1 Å length reference. Use excess chemical potentials and force-field conventions from the chosen reservoir model when preparing experimental conditions.

For `forcefield.slab: true`, specify the accessible liquid region in fractional coordinates. The z extent must be restricted. For example, the liquid may occupy z fractions 0.375–0.625 with fully periodic x and y.

```json
{
  "region": {
    "lower": [0.0, 0.0, 0.375],
    "upper": [1.0, 1.0, 0.625]
  }
}
```

## Reaction sampling and calibration

Reaction edges connect states of the same molecular template. Proton and electron counts, intrinsic free energies and the hydrogen and electron reservoirs determine the state weights. A charge-changing move in an Ewald cell needs an explicit compensating electrode or a neutral coupled reaction. Staged NCMC propagates solvent between alchemical bridge states and uses accumulated work for the acceptance decision.

For calibration, `calibration.edge_index` selects the reaction, and `calibration.uid` identifies an initial molecule of that reaction's template. The solver samples the alchemical states at fixed water and ion counts and evaluates the cross-state reduced-potential matrix. The resulting overlap describes the support between windows. Independent sampling and bulk reference states govern the accuracy and transferability of a calibrated pKa or formal potential.

```bash
opengcmc-reactive calibrate examples/electrochemistry/hydrated_redox_calibration.json -o output/redox_windows
```

For a protonation edge, `reference_deprotonation_kelvin` follows HA → A⁻. For a redox edge, `reference_reduction_kelvin` follows Ox → Red. Match the direction when transferring a window result to the reservoir convention.

## Current calculations

The reaction network supports Butler–Volmer, Marcus and Marcus–Hush–Chidsey rates. It uses Gillespie sampling for stochastic events and a master equation for continuous voltage programs. The finite-volume diffusion model assumes supporting electrolyte and buffered pH. It uses SI inputs for time, length, concentration, diffusivity, area and heterogeneous rate constants.

```bash
opengcmc-reactive transport examples/electrochemistry/diffusion_chronoamperometry.json -o output/chrono
opengcmc-reactive transport examples/electrochemistry/diffusion_voltammetry.json -o output/cv
```

Output records separate Faradaic and capacitive currents. The example models provide runnable controls for free-energy, charge and flux calculations. To apply them to a material or electrode, supply the corresponding geometry, interaction parameters, reservoir, rate constants and supported conditions.

References: [SPC/E](https://doi.org/10.1021/j100308a038), [constant-potential charge fluctuations](https://doi.org/10.1039/C9CP06285H), [MBAR](https://doi.org/10.1063/1.2978177).
