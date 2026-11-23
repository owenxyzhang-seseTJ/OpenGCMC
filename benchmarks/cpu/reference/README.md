# Reference datasets

The NIST TMMC input files used for ZIF-8 CO₂ (300 K), ZIF-8 N₂/CO₂ (300 K), and UiO-66 CO₂ (350 K) are from `dwsideriusNIST/NIST_SRSW_Data` at `84d8d65`. The included files keep the upstream directory layout. [Original reference data](https://github.com/dwsideriusNIST/NIST_SRSW_Data/tree/84d8d65) · [NIST copyrights and disclaimers](https://www.nist.gov/copyrights-disclaimers)

- NIST. TMMC Reference Adsorption Isotherms for CO₂ in Metal–Organic Frameworks. [Pure-component data and parameters](https://www.nist.gov/mml/csd/chemical-informatics-group/tmmc-reference-adsorption-isotherms-co2-metal-organic-frameworks)
- NIST. TMMC Reference Adsorption Isotherms for N₂/CO₂ Mixtures in Metal–Organic Frameworks. [Mixture data and parameters](https://www.nist.gov/mml/csd/chemical-informatics-group/tmmc-reference-adsorption-isotherms-n2co2-mixtures-metal-organic)
- International Zeolite Association Structure Commission. [TON framework CIF](https://www.iza-structure.org/IZA-SC/cif/TON.cif) · [MFI framework CIF](https://www.iza-structure.org/IZA-SC/cif/MFI.cif)
- The IRMOF-1 mixture uses the RASPA3 example structure and force field. [RASPA3](https://github.com/iRASPA/RASPA3) · [License](RASPA3_LICENSE.txt)

Reweight the reference replicas from the repository root.

```bash
python benchmarks/cpu/paired/prepare_nist.py benchmarks/cpu/reference/nist benchmarks/runs/nist_models nist_co2_zif8 nist_n2_co2_zif8 holdout_co2_uio66
```
