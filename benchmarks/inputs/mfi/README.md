# CO₂ and CH₄ in siliceous MFI

This controlled case uses two molecular sizes, a charged CO₂ guest, and a common finite Ewald target. The fixed framework takes its geometry from the idealized public IZA structure. Place the downloaded structure next to the input as `MFI.cif`. [IZA structure data](https://www.iza-structure.org/IZA-SC/cif/MFI.cif)

The model uses TraPPE CO₂ and united-atom CH₄ parameters. The framework uses explicit neutral Si and O LJ sites with rounded UFF parameters and Lorentz–Berthelot cross interactions. Experimental fitting and long-range convergence require independent validation. The pressure sweep assumes ideal reservoir fugacity coefficients. [TraPPE CO₂](https://doi.org/10.1002/aic.690470719) · [TraPPE alkanes](https://doi.org/10.1021/jp972543%2B) · [UFF](https://doi.org/10.1021/ja00051a040)

Run an individual pressure with `python -m poremc run` or an isotherm with `python -m poremc.mcp submit` followed by `python -m poremc.mcp work`. Report per-species uptake, the composition-normalized selectivity and chain diagnostics together with temperature, pressure and the simulation budget.
