# Scientific case analyses

OpenGCMC has been exercised on charged-MOF mixture adsorption, constant-potential electrolyte interfaces, explicit-water reaction free energies, and diffusion-controlled electrochemistry. These cases connect full system inputs to reproducible numerical outputs and cross-checks. They demonstrate the scientific workflows supported by the software; they do not constitute new experimental measurements or calibrated predictions for an unmeasured material.

## 1. CO2/CH4 mixture adsorption in charged IRMOF-1

This case uses the charged IRMOF-1 CIF, site charges, guest models, and force-field conventions from the official RASPA3 binary-mixture example. The framework contains 424 atoms with a net charge of approximately \(-5.3\times10^{-15}\,e\). The reservoir fugacities are 0.5 bar for CO2 and 1.5 bar for CH4 at 300 K. The calculation uses a 12 Å cutoff, shifted Lennard-Jones interactions, no tail correction, and Ewald electrostatics.

Three independent OpenGCMC chains each use 8,000 warm-up attempts followed by 50,000 production attempts. RASPA3 3.1.0 is run with 3,000 warm-up cycles and 20,000 production cycles.

| Component | OpenGCMC molecules per cell | RASPA3 molecules per cell |
|---|---:|---:|
| CO2 | 2.647, 95% block interval [2.529, 2.768] | 2.615 ± 0.027 |
| CH4 | 4.257, 95% block interval [4.128, 4.388] | 4.246 ± 0.033 |

The three-chain split \(\hat R\) values are 1.0061 for CO2 and 1.0005 for CH4. With CO2 and CH4 ordered as rows and applied \(\ln z\) perturbations as columns, the fluctuation response matrix is

\[
\chi =
\begin{bmatrix}
2.642 & 0.069\\
0.069 & 4.301
\end{bmatrix}.
\]

Three biased windows centred on total occupancies 3, 8, and 13, with 4,500 retained frames per window, produce a free-energy landscape whose minimum is at total occupancy 6. The MBAR result and an independent unbiased trajectory differ by a total-variation distance of 0.0217; the overlap spectral gap is 0.435.

This case demonstrates charged-framework GCMC, binary competitive adsorption, response-matrix analysis, independent-chain convergence diagnostics, and MBAR free-energy reconstruction in one workflow. It is a cross-engine and numerical-consistency case, not a new experimental isotherm.

## 2. NaCl in SPC/E water at a constant-potential interface

Two ideal Gaussian electrodes are separated by 16 Å and held at a 0.10 V potential difference. The cell contains 62 rigid SPC/E water molecules and one Na+/Cl- pair at 298.15 K. The ions use Joung-Cheatham parameters for SPC/E water. Water and ion charges are compensated by the electrodes.

Two independent seeds each use 8,000 warm-up steps and 16,000 production steps. The electrode charge agrees with an independent solve to within \(1.1\times10^{-15}\,e\). The final molecular energy agrees with a complete recomputation to within \(1.95\times10^{-7}\,K\).

| Quantity | Seed 1774 | Seed 6231 |
|---|---:|---:|
| Na-O coordination number | 5.03 | 5.09 |
| Cl-O coordination number | 6.45 | 6.46 |
| Mean conjugated electrode charge | +0.087 e | -0.045 e |

The first-shell coordination numbers use radii of 3.2 Å for Na-O and 3.8 Å for Cl-O. The fixed water and salt counts isolate the interface model without relying on an uncalibrated electrolyte reservoir. Longer sampling is required for an equilibrium capacitance estimate.

This case demonstrates explicit-solvent ion sampling, constrained Gaussian electrode charge, local energy consistency, and structural coordination analysis for an electrochemical interface.

## 3. Ferricyanide/ferrocyanide diffusion electrochemistry

The transport workflow uses literature diffusion coefficients at 25 °C, \(D_{\mathrm{ox}}=7.26\times10^{-10}\,\mathrm{m^2\,s^{-1}}\) and \(D_{\mathrm{red}}=6.67\times10^{-10}\,\mathrm{m^2\,s^{-1}}\), 1 mM oxidized species, and a 3 mm planar electrode. The formal potential is used as the zero and the heterogeneous kinetics are set fast to test the reversible diffusion limit.

| Calculation | OpenGCMC result | Analytical comparison |
|---|---|---:|
| -0.35 V step, 0.005-0.5 s | log-log slope -0.50018 | Cottrell slope -0.5; maximum current difference 0.108% |
| 100 mV/s CV cathodic peak | -16.1747 µA | Randles-Sevcik magnitude 16.1802 µA |
| 200 mV/s CV cathodic peak | -22.8718 µA | Randles-Sevcik magnitude 22.8823 µA |

The peak-current ratio between the two scan rates is 1.41404, consistent with the square root of two. Both voltammograms have a 58.5 mV peak separation.

This case demonstrates finite-volume diffusion, physical-time current integration, chronoamperometry, cyclic voltammetry, and validation against analytical electrochemical limits.

## 4. Explicit-water reactions and multistate free-energy calculations

The reaction workflow samples protonation, redox, and proton-coupled electron-transfer states with rigid SPC/E water, ions, and a fixed molecular topology. A seven-window free-energy calculation with 35 retained frames per window gives an overlap spectral gap of 0.0249. A fixed-snapshot seven-state charge calculation agrees with direct state-by-state evaluation to within \(1.17\times10^{-10}\,K\).

The runnable inputs expose the required molecular topology, reaction edges, reservoir conventions, charge compensation, and calibration fields. The example parameters are controls for workflow and numerical validation; physical pKa or formal-potential prediction requires a calibrated molecular model, converged reservoir activities, and independent reference states.

This case demonstrates non-equilibrium candidate Monte Carlo, explicit-water reaction sampling, multistate window construction, MBAR overlap diagnostics, and local electrostatic response evaluation.

## Interpretation

The cases above cover the three scientific modes exposed by OpenGCMC:

- equilibrium adsorption and mixture thermodynamics in porous materials;
- constant-potential electrolyte interfaces with explicit solvent;
- reaction free energies and physical-time electrochemical currents.

The retained checks establish implementation consistency and analytical-limit behavior. Material-specific predictive work additionally requires a validated structure, force field, charge model, reservoir treatment, system-size convergence, and sampling-error analysis for the target scientific question.

The technical foundations and equations are described in the [technical report](OpenGCMC_Technical_Report.pdf), and runnable input schemas are in [examples](../examples).
