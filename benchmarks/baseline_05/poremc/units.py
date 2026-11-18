"""Explicit units. Energies are U/k_B [K], length Angstrom, charge e."""
import math
KB = 1.380649e-23
NA = 6.02214076e23
COULOMB_KA = 167100.94689828736  # e^2/(4 pi eps0 kB) [K Angstrom]
KJ_MOL_TO_K = 1000.0 / (NA * KB)

def activity_from_fugacity_pa(fugacity_pa, temperature, log_q_internal=0.0):
    """z = f/(kB T Qref), for normalized geometric reference measure.

    Qref is the dimensionless isolated intramolecular partition integral in
    EXACTLY the sampler's reference measure. No molecular mass factor belongs
    here: physical fugacity already includes the gas internal partition sum.
    """
    if not math.isfinite(fugacity_pa) or fugacity_pa < 0 or temperature <= 0:
        raise ValueError("fugacity must be finite/nonnegative and T positive")
    if not math.isfinite(temperature) or not math.isfinite(log_q_internal):
        raise ValueError("T and log_q_internal must be finite")
    return fugacity_pa / (KB * temperature) * 1e-30 * math.exp(-log_q_internal)

def partial_fugacity_pa(total_pressure_pa, mole_fraction, coefficient=1.0):
    if total_pressure_pa < 0 or not 0 <= mole_fraction <= 1 or coefficient <= 0:
        raise ValueError("invalid pressure, composition or fugacity coefficient")
    return total_pressure_pa * mole_fraction * coefficient

def uptake_mmol_g(mean_molecules, host_mass_amu):
    if host_mass_amu <= 0:
        raise ValueError("host mass in amu must be positive")
    return 1000.0 * mean_molecules / host_mass_amu

def single_cosine_log_q(k_kelvin, temperature):
    """Integral exp[-k(1+cos(n phi-phase))/T] dphi/(2pi), n nonzero.

    Uses scipy.i0e if available, otherwise NumPy in its non-overflow range.
    Add constants from every other internal degree of freedom separately.
    """
    a = float(k_kelvin) / temperature
    try:
        from scipy.special import i0e
        return -a + abs(a) + math.log(float(i0e(a)))
    except ImportError:
        import numpy as np
        if abs(a) > 500:
            raise ValueError("large torsion amplitude requires scipy.special.i0e")
        return -a + math.log(float(np.i0(a)))
