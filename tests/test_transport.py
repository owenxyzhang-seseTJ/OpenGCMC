import numpy as np
import pytest
from scipy.sparse.linalg import spsolve
from poremc.transport import *


def test_finite_volume_steady_flux_exact():
    m=DiffusionRedox(cells=60, bulk_oxidized_mol_m3=2.,bulk_reduced_mol_m3=.2)
    E=-.1;M,s=m.operator(E);c=spsolve(M,-s);f,r=m.boundary_coefficients(E)
    flux=f*c[0]-r*c[m.cells]
    np.testing.assert_allclose(flux,m.steady_flux(E),rtol=1e-11)


def test_equilibrium_zero_current():
    m=DiffusionRedox(cells=20,bulk_oxidized_mol_m3=1.,bulk_reduced_mol_m3=1.)
    r=m.run([0,.2],[0,0],samples=10)
    np.testing.assert_allclose(r['faradaic_current_A'],0,atol=1e-18)
    np.testing.assert_allclose(r['oxidized_mol_m3'],1,atol=1e-10)


def test_closed_diffusion_reaction_mass_balance():
    m=DiffusionRedox(cells=50,closed_bulk=True)
    r=m.run([0,.5],[-.12,-.12],samples=50)
    inventory=r['total_inventory_mol']
    np.testing.assert_allclose(inventory,inventory[0],rtol=1e-10)
    assert r['minimum_concentration_mol_m3']>-1e-8


def test_cottrell_limit():
    m=DiffusionRedox(length_m=3e-4,cells=600,k0_m_s=.1,temperature_K=300.)
    times=np.geomspace(.01,1.,31)
    r=m.run([0,1],[-.4,-.4],output_times_s=times)
    expected=m.bulk_oxidized_mol_m3*np.sqrt(m.diffusivity_oxidized_m2_s/(np.pi*times))
    np.testing.assert_allclose(r['molar_flux_mol_m2_s'],expected,rtol=.015)


def test_capacitive_current_ramp_units():
    m=DiffusionRedox(cells=20,double_layer_capacitance_F_m2=.2,area_m2=2e-6)
    r=m.run([0,1],[0,.1],samples=12)
    np.testing.assert_allclose(r['capacitive_current_A'],4e-8,rtol=1e-12)
    np.testing.assert_allclose(r['total_current_A'],r['capacitive_current_A']+r['faradaic_current_A'])


def test_extreme_voltage_stable_boundary():
    m=DiffusionRedox()
    for E in [-100,100]:
        a,b=m.boundary_coefficients(E)
        assert np.isfinite([a,b]).all() and min(a,b)>=0


def test_transport_reference_pH_shift():
    a=DiffusionRedox(protons=1,pH=8,reference_pH=7)
    shift=np.log(10)*a.temperature_K/VOLT_TO_K
    b=DiffusionRedox(protons=1,pH=7,reference_pH=7)
    np.testing.assert_allclose(a.boundary_coefficients(0),b.boundary_coefficients(shift))


def test_invalid_transport_parameters():
    with pytest.raises(ValueError):DiffusionRedox(diffusivity_oxidized_m2_s=0)
