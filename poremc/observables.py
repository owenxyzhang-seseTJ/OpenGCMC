"""Statistical observables and gas-reservoir thermodynamics.

The implementation uses U/k_B in kelvin and lengths in Angstrom. Fugacities
from Peng--Robinson use pascal. The formulas were checked against pinned
RASPA3/gRASPA implementations; no upstream source code is included here.

A Widom insertion uses the full cell as its reference translation measure.
The resulting Boltzmann-weighted insertion volume is not a geometric void
volume: attraction can make it larger than the cell. A hard, noninteracting
probe is the special case in which it becomes an accessible volume.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
from .cbmc import CBMC, molecule_energy
from .cell import validate_cell
from .statistics import block_standard_error
from .units import KB, NA

R_SI = KB * NA


def _positive_temperature(temperature):
    value = float(temperature)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("temperature must be positive and finite")
    return value


def logmeanexp(log_values):
    """Log of a mean, including every -inf (zero-weight) sample in its count."""
    values = np.asarray(log_values, float)
    if values.ndim != 1 or len(values) == 0 or np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("log weights must be nonempty and contain finite values or -inf")
    high = float(values.max())
    if high == -math.inf:
        return -math.inf
    return high + math.log(float(np.exp(values-high).mean()))


def _safe_exp(log_value):
    if log_value == -math.inf:
        return 0.0
    if log_value > math.log(np.finfo(float).max):
        return math.inf
    return math.exp(log_value)


def log_weight_summary(log_weights, blocks=10):
    """Stable weight statistics and block delta-method uncertainty.

    Kish ESS measures concentration of importance weights; it is not the
    time-series autocorrelation ESS. Block uncertainty can reflect serial
    correlation only if the selected blocks are longer than that correlation.
    The primary mean always uses all samples, including failed growths.
    """
    logs = np.asarray(log_weights, float)
    logmean = logmeanexp(logs)
    n = len(logs)
    valid = np.isfinite(logs)
    if not valid.any():
        return {"samples": n, "zero_weight_samples": n, "positive_weight_fraction": 0.0,
                "log_mean_weight": -math.inf, "mean_weight": 0.0,
                "kish_weight_ess": 0.0, "relative_sem_blocks": math.inf,
                "log_mean_sem_delta": math.inf}
    scaled = np.exp(logs-float(logs[valid].max()))
    mean = float(scaled.mean())
    sem = block_standard_error(scaled, blocks=blocks)
    relative_sem = sem/mean
    kish = float(scaled.sum()**2 / np.dot(scaled, scaled))
    return {"samples": n, "zero_weight_samples": int((~valid).sum()),
            "positive_weight_fraction": float(valid.mean()),
            "log_mean_weight": logmean, "mean_weight": _safe_exp(logmean),
            "kish_weight_ess": kish, "relative_sem_blocks": relative_sem,
            "log_mean_sem_delta": relative_sem}


def widom_from_log_weights(log_weights, temperature, log_q_internal,
                          blocks=10, volume_angstrom3=None,
                          log_q_internal_sem=0.0):
    """mu_ex/k_B = -T log(<W>/Q_ref), with normalized CBMC W.

    Input zero weights are -inf, never discarded. Q_ref must use the same
    internal reference measure and Hamiltonian zero. If its estimated SEM is
    supplied, it must come from an independent gas-reference calculation.
    For a mobile NVT background, thermodynamic mu_ex requires weights averaged
    over its equilibrium configurations as well as insertion trials. Weights
    from one fixed background give its conditional insertion free energy.
    """
    temperature = _positive_temperature(temperature)
    if not math.isfinite(log_q_internal) or not math.isfinite(log_q_internal_sem) or log_q_internal_sem < 0:
        raise ValueError("a finite log Q_ref and nonnegative finite SEM are required")
    stats = log_weight_summary(log_weights, blocks=blocks)
    log_ratio = stats["log_mean_weight"]-float(log_q_internal)
    stats.update({"log_q_internal": float(log_q_internal),
                  "log_mean_insertion_weight_over_q": log_ratio,
                  "mu_ex_kelvin": -temperature*log_ratio,
                  "mu_ex_kJ_mol": -R_SI*temperature*log_ratio/1000,
                  "mu_ex_sem_kelvin_delta": temperature*math.hypot(stats["log_mean_sem_delta"],log_q_internal_sem),
                  "reference_translation_measure": "uniform over full cell"})
    if volume_angstrom3 is not None:
        if not math.isfinite(volume_angstrom3) or volume_angstrom3 <= 0:
            raise ValueError("volume must be finite and positive")
        stats["boltzmann_weighted_insertion_volume_A3"] = _safe_exp(math.log(volume_angstrom3)+log_ratio)
    return stats


def widom_cbmc(cbmc, molecule, environment, samples=1000, seed=12345,
               log_q_internal=None, blocks=10, return_samples=False,
               target_potential=None, target_environment=None):
    """Test insertion at a fixed environment without changing its population.

    cbmc must use the target potential unless target_potential is supplied.
    In the latter case each grown-path Rosenbluth weight is importance-corrected
    by exp[-(Delta E_target-Delta E_surrogate)/T], where E=U/k_B and each
    Delta E subtracts the corresponding potential's energy of the same fixed
    background, with one complete target insertion-energy evaluation.
    target_environment may be passed explicitly for opaque prepared objects.
    This fixed-environment call returns a conditional insertion free energy.
    Thermodynamic mu_ex for a mobile NVT background also needs equilibrium
    background averaging of the weights before taking their logarithm.
    """
    samples = int(samples)
    if samples < 1:
        raise ValueError("at least one test insertion is required")
    if log_q_internal is None:
        log_q_internal = molecule.log_q_internal
        if log_q_internal is None:
            if not molecule.is_rigid:
                raise ValueError("flexible Widom insertion requires log_q_internal")
            log_q_internal = 0.0
    rng = np.random.default_rng(seed)
    logs = np.empty(samples)
    if target_potential is not None and target_environment is None:
        entries = getattr(environment, "entries", environment)
        target_environment = (target_potential.prepare_environment(entries)
                              if hasattr(target_potential, "prepare_environment") else entries)
    for i in range(samples):
        growth = cbmc.grow(molecule, environment, rng)
        logs[i] = growth.log_weight if growth.valid else -math.inf
        if growth.valid and target_potential is not None:
            target_energy = molecule_energy(target_potential, growth.positions, molecule, target_environment)
            if math.isnan(target_energy) or target_energy == -math.inf:
                raise FloatingPointError("invalid target insertion energy")
            logs[i] -= (target_energy-growth.energy)/cbmc.temperature
    result = widom_from_log_weights(logs, cbmc.temperature, log_q_internal,
                                  blocks=blocks, volume_angstrom3=float(np.linalg.det(cbmc.cell)))
    result["reweighted_to_target"] = target_potential is not None
    if return_samples:
        result["log_weights"] = logs.tolist()
    return result


def estimate_q_internal(molecule, temperature, internal_potential, samples=1000,
                        trials=8, seed=12345, blocks=10, return_samples=False):
    """Isolated intramolecular Rosenbluth integral and gas internal energy.

    The potential must explicitly declare is_isolated_internal=True. It must
    have no host, guest environment, periodic self-image or reciprocal energy.
    Every independent growth contributes one weight, including failed W=0.
    The normalized growth measure integrates translations to one; their
    arbitrary unit-cell scale therefore does not contribute to Q_ref.
    """
    temperature = _positive_temperature(temperature)
    if getattr(internal_potential,"is_isolated_internal",False) is not True:
        raise ValueError("gas reference requires an explicitly isolated internal potential")
    samples = int(samples)
    if samples < 1:
        raise ValueError("at least one gas-reference growth is required")
    rng = np.random.default_rng(seed)
    cbmc = CBMC(temperature, np.eye(3), internal_potential, trials=trials)
    env = internal_potential.prepare_environment([]) if hasattr(internal_potential,"prepare_environment") else []
    logs = np.empty(samples); energies = np.zeros(samples)
    for i in range(samples):
        growth = cbmc.grow(molecule,env,rng)
        logs[i] = growth.log_weight if growth.valid else -math.inf
        if growth.valid:
            energies[i] = growth.energy
    result = log_weight_summary(logs, blocks=blocks)
    result["log_q_internal"] = result["log_mean_weight"]
    result["q_internal"] = result["mean_weight"]
    finite = np.isfinite(logs)
    if finite.any():
        weights = np.exp(logs-float(logs[finite].max()))
        u = float(np.dot(weights,energies)/weights.sum())
        residual = weights*(energies-u)
        sem = block_standard_error(residual,blocks=blocks)/float(weights.mean())
        result["internal_energy_kelvin"] = u
        result["internal_energy_sem_kelvin_blocks"] = float(sem)
    else:
        result["internal_energy_kelvin"] = None
        result["internal_energy_sem_kelvin_blocks"] = None
    result["temperature"] = temperature
    result["reference_measure"] = "normalized tree geometry; isolated internal energy"
    if return_samples:
        result["log_weights"] = logs.tolist()
        result["selected_internal_energies_kelvin"] = energies.tolist()
    return result


def _qst_scalar(count, energy, temperature, gas_internal):
    centered = count-count.mean()
    denom = float(np.dot(centered,centered))
    if denom <= 0:
        return None
    slope = float(np.dot(centered,energy-energy.mean())/denom)
    return temperature+gas_internal-slope


def loading_fluctuations(counts, temperature, energies_kelvin=None,
                         gas_internal_energy_kelvin=None, blocks=10,
                         gas_internal_energy_sem_kelvin=0.0):
    """Number covariance, fugacity susceptibility and single-component Qst.

    d<N_i>/d ln f_j = cov(N_i,N_j) at fixed T and other fugacities.
    d<N_i>/d(mu_j/k_B) = cov(N_i,N_j)/T.
    For a single species with an ideal-gas heat reference,
    Qst/R = T + <U_internal,gas>/k_B - cov(U/k_B,N)/var(N).
    This also defines the fugacity-isostere slope; it does not include a
    nonideal gas residual enthalpy. Peng--Robinson fugacities alone do not
    supply that correction or turn the result into a general nonideal-gas heat.
    The potential-energy trajectory must include the guest internal energies
    in the same energy-zero convention as the isolated-gas reference. An
    explicit zero gas reference is appropriate for rigid-body Hamiltonians.
    """
    temperature = _positive_temperature(temperature)
    n = np.asarray(counts,float)
    if n.ndim == 1:
        n=n[:,None]
    if n.ndim != 2 or n.shape[0]<2 or n.shape[1]<1 or not np.isfinite(n).all() or np.any(n<0):
        raise ValueError("counts require at least two finite nonnegative samples")
    if not np.allclose(n,np.rint(n),atol=1e-10,rtol=0):
        raise ValueError("molecule counts must be integers")
    centered = n-n.mean(axis=0)
    covariance = centered.T@centered/(len(n)-1)
    result = {"samples": len(n), "mean_counts": n.mean(axis=0).tolist(),
              "number_covariance": covariance.tolist(),
              "dN_dlog_fugacity": covariance.tolist(),
              "dN_dchemical_potential_kelvin": (covariance/temperature).tolist(),
              "qst_kJ_mol": None}
    if energies_kelvin is None:
        result["qst_status"]="energy trajectory not supplied"
        return result
    e=np.asarray(energies_kelvin,float)
    if e.shape!=(len(n),) or not np.isfinite(e).all():
        raise ValueError("energy trajectory must be finite and align with all count samples")
    cov_ne=centered.T@(e-e.mean())/(len(n)-1)
    result["number_energy_covariance_kelvin"]=cov_ne.tolist()
    eigen=np.linalg.eigvalsh(covariance)
    if eigen.min()>max(eigen.max(),1.)*1e-12:
        result["partial_molar_config_energy_kelvin"]=np.linalg.solve(covariance,cov_ne).tolist()
    else:
        result["partial_molar_config_energy_kelvin"]=None
    if n.shape[1] != 1:
        result["qst_status"]="single-component Qst only; mixture covariance and susceptibilities provided"
        return result
    if gas_internal_energy_kelvin is None:
        result["qst_status"]="isolated-gas internal energy reference not supplied"
        return result
    gas=float(gas_internal_energy_kelvin)
    gas_sem=float(gas_internal_energy_sem_kelvin)
    if not math.isfinite(gas) or not math.isfinite(gas_sem) or gas_sem<0:
        raise ValueError("gas internal energy and nonnegative SEM must be finite")
    qst=_qst_scalar(n[:,0],e,temperature,gas)
    if qst is None:
        result["qst_status"]="number variance is zero"
        return result
    result.update({"qst_over_R_kelvin":qst, "qst_kJ_mol":R_SI*qst/1000,
                   "gas_internal_energy_kelvin":gas,"qst_status":"computed"})
    b=min(int(blocks),len(n)//2)
    sem=math.nan
    if b>=2:
        size=len(n)//b; keep=size*b; leave=[]
        for k in range(b):
            mask=np.ones(keep,dtype=bool);mask[k*size:(k+1)*size]=False
            q=_qst_scalar(n[:keep,0][mask],e[:keep][mask],temperature,gas)
            if q is not None:leave.append(q)
        if len(leave)==b:
            leave=np.asarray(leave)
            sem=math.sqrt((b-1)/b*float(np.sum((leave-leave.mean())**2))+gas_sem**2)
    result["qst_sem_kJ_mol_block_jackknife"]=R_SI*sem/1000
    return result


def _frames(frames):
    if isinstance(frames,np.ndarray):
        if frames.ndim==2:
            frames=[frames]
        elif frames.ndim!=3:
            raise ValueError("frames must have shape (frames,points,3) or be a list of point arrays")
    result=[np.asarray(x,float) for x in frames]
    if not result:
        raise ValueError("at least one frame is required")
    if any(x.ndim!=2 or x.shape[1]!=3 or not np.isfinite(x).all() for x in result):
        raise ValueError("every frame must contain finite (n,3) positions")
    return result


def rdf(frames, cell, edges_angstrom, other_frames=None):
    """Radial pair distribution with finite-N pair-opportunity normalization.

    Same-set pairs are unique unordered pairs (self pairs excluded). Cross-set
    pairs include every i,j; callers wanting self exclusions must use one set.
    The ideal reference per frame is N(N-1)/2 or N_A*N_B times shell_volume/V.
    Thus g(r)=1 for uniform independent points at any fixed population. In a
    spatially inhomogeneous pore this global-volume RDF includes pore geometry.
    """
    a=_frames(frames);b=None if other_frames is None else _frames(other_frames)
    if b is not None and len(a)!=len(b):raise ValueError("cross-RDF frame counts differ")
    edges=np.asarray(edges_angstrom,float)
    if edges.ndim!=1 or len(edges)<2 or not np.isfinite(edges).all() or edges[0]<0 or np.any(np.diff(edges)<=0):
        raise ValueError("RDF edges must be strictly increasing nonnegative distances")
    h,inv,_=validate_cell(cell,float(edges[-1]));volume=float(np.linalg.det(h))
    hist=np.zeros(len(edges)-1,dtype=np.int64);opportunities=0
    for frame_idx,x in enumerate(a):
        y=x if b is None else b[frame_idx]
        opportunities+=len(x)*(len(x)-1)/2 if b is None else len(x)*len(y)
        for i in range(len(x)):
            d=y[i+1:]-x[i] if b is None else y-x[i]
            fractional=d@inv
            distances=np.linalg.norm((fractional-np.rint(fractional))@h,axis=1)
            hist+=np.histogram(distances,bins=edges)[0]
    shell=4*math.pi/3*np.diff(edges**3)
    expected=opportunities*shell/volume
    g=np.divide(hist,expected,out=np.full(len(hist),np.nan),where=expected>0)
    return {"r_centers_angstrom":((edges[:-1]+edges[1:])/2).tolist(),
            "edges_angstrom":edges.tolist(),"g_r":g.tolist(),"pair_counts":hist.tolist(),
            "normalization_pair_opportunities":float(opportunities),"frames":len(a),
            "mean_number_density_A3":float(np.mean([len(x) for x in a]))/volume}


def density_grid(frames,cell,bins=(20,20,20)):
    """Periodic fractional-coordinate grid of average number density [A^-3]."""
    frames=_frames(frames);h,inv,_=validate_cell(cell)
    bins=np.asarray(bins,int)
    if bins.shape!=(3,) or np.any(bins<1):raise ValueError("density grid needs three positive bin counts")
    counts=np.zeros(tuple(bins),dtype=np.int64)
    for x in frames:
        fractional=np.mod(x@inv,1.)
        index=np.floor(fractional*bins).astype(int)
        index=np.minimum(index,bins-1)
        np.add.at(counts,tuple(index.T),1)
    voxel=float(np.linalg.det(h))/int(np.prod(bins))
    density=counts/(len(frames)*voxel)
    return {"number_density_A3":density,"counts":counts,"cell":h.copy(),
            "fractional_edges":[np.linspace(0,1,int(k)+1) for k in bins],
            "voxel_volume_A3":voxel,"frames":len(frames),
            "mean_population":float(density.sum()*voxel)}


def write_pdb_trajectory(path,frames,cell,atom_names=None):
    """Write MODEL/ENDMDL PDB frames; coordinates are not atomwise wrapped."""
    frames=_frames(frames);h,_,_=validate_cell(cell)
    lengths=np.linalg.norm(h,axis=1)
    alpha=math.degrees(math.acos(np.clip(np.dot(h[1],h[2])/(lengths[1]*lengths[2]),-1,1)))
    beta=math.degrees(math.acos(np.clip(np.dot(h[0],h[2])/(lengths[0]*lengths[2]),-1,1)))
    gamma=math.degrees(math.acos(np.clip(np.dot(h[0],h[1])/(lengths[0]*lengths[1]),-1,1)))
    lines=[f"CRYST1{lengths[0]:9.3f}{lengths[1]:9.3f}{lengths[2]:9.3f}{alpha:7.2f}{beta:7.2f}{gamma:7.2f} P 1           1"]
    for model,x in enumerate(frames,1):
        if len(x)>99999 or model>9999 or np.any(x < -999.999) or np.any(x>9999.999):
            raise ValueError("PDB field range exceeded; use a larger-capacity trajectory format")
        if atom_names is not None and len(atom_names)!=len(x):raise ValueError("atom_names must match each frame")
        lines.append(f"MODEL     {model:4d}")
        for i,p in enumerate(x,1):
            name="X" if atom_names is None else str(atom_names[i-1])
            if not name or len(name)>4 or not name.isalnum():raise ValueError("PDB atom names must be 1-4 alphanumeric characters")
            lines.append(f"HETATM{i:5d} {name:<4s} MOL A   1    {p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00  0.00")
        lines.append("ENDMDL")
    lines.append("END")
    Path(path).write_text("\n".join(lines)+"\n",encoding="ascii")


def peng_robinson_fugacity(temperature,pressure_pa,mole_fractions,
                           critical_temperatures,critical_pressures_pa,
                           acentric_factors,kij=None,phase="vapor"):
    """Peng--Robinson (1976) with quadratic a mixing and linear b mixing.

    All pressures, including Pc, are in Pa. kij must be explicitly supplied for
    mixtures, be symmetric, and have a zero diagonal. 'vapor' selects the
    largest physical Z root, 'liquid' the smallest, and 'stable' the root of
    minimum sum(x_i ln phi_i) at the specified composition. This root selection
    is not a vapor/liquid composition flash or a phase-stability analysis.
    """
    temperature=_positive_temperature(temperature);pressure=float(pressure_pa)
    if not math.isfinite(pressure) or pressure<0:raise ValueError("pressure must be finite and nonnegative")
    x=np.asarray(mole_fractions,float)
    tc=np.asarray(critical_temperatures,float);pc=np.asarray(critical_pressures_pa,float);omega=np.asarray(acentric_factors,float)
    if x.ndim!=1 or len(x)<1 or any(a.shape!=x.shape for a in [tc,pc,omega]):raise ValueError("PR component arrays must have equal nonzero length")
    if not all(np.isfinite(a).all() for a in [x,tc,pc,omega]) or np.any(x<0) or np.any(tc<=0) or np.any(pc<=0):raise ValueError("invalid PR composition or critical constants")
    if not math.isclose(float(x.sum()),1.,rel_tol=0,abs_tol=1e-10):raise ValueError("mole fractions must sum to one")
    x=x/x.sum()
    if kij is None:
        if len(x)>1:raise ValueError("mixtures require an explicit symmetric kij matrix, including explicit zeros")
        kij=np.zeros((1,1))
    kij=np.asarray(kij,float)
    if kij.shape!=(len(x),len(x)) or not np.isfinite(kij).all() or not np.allclose(kij,kij.T,atol=1e-12,rtol=0) or not np.allclose(np.diag(kij),0.,atol=1e-12,rtol=0):
        raise ValueError("kij must be finite, symmetric, and have a zero diagonal")
    if phase not in ("vapor","liquid","stable"):raise ValueError("phase must be vapor, liquid or stable")
    kappa=.37464+1.54226*omega-.26992*omega**2
    alpha=(1+kappa*(1-np.sqrt(temperature/tc)))**2
    ai=.45724*(R_SI*tc)**2/pc*alpha
    bi=.07780*R_SI*tc/pc
    aij=np.sqrt(ai[:,None]*ai[None,:])*(1-kij)
    amix=float(x@aij@x);bmix=float(x@bi)
    if amix<0 or not math.isfinite(amix+bmix):raise ValueError("kij produced a nonphysical or nonfinite mixed attraction")
    if pressure==0:
        return {"compressibility":1.,"physical_roots":[1.],"fugacity_coefficients":np.ones(len(x)).tolist(),
                "log_fugacity_coefficients":np.zeros(len(x)).tolist(),"fugacities_pa":np.zeros(len(x)).tolist(),
                "molar_volume_m3_mol":math.inf,"A":0.,"B":0.,"phase":phase,
                "residual_gibbs_over_RT":0.,"flash_performed":False}
    A=amix*pressure/(R_SI*temperature)**2;B=bmix*pressure/(R_SI*temperature)
    coefficients=np.array([1.,B-1.,A-3*B*B-2*B,-A*B+B*B+B**3])
    raw=np.roots(coefficients)
    candidates=[]
    for value in raw:
        if abs(value.imag)>1e-9*max(1.,abs(value.real)):continue
        z=float(value.real)
        # Newton polish without crossing the excluded-volume singularity.
        for _ in range(3):
            residual=float(np.polyval(coefficients,z));derivative=3*z*z+2*coefficients[1]*z+coefficients[2]
            if abs(derivative)<1e-14:break
            trial=z-residual/derivative
            if trial<=B:break
            z=trial
        if z>B and math.isfinite(z):candidates.append(z)
    candidates.sort();roots=[]
    for z in candidates:
        if not roots or abs(z-roots[-1])>1e-10*max(abs(z),abs(roots[-1]),1e-12):roots.append(z)
    if not roots:raise FloatingPointError("PR cubic has no resolved physical Z>B root")
    sqrt2=math.sqrt(2.)
    attraction=(2*(aij@x)-amix*bi/bmix)/(2*sqrt2*bmix*R_SI*temperature)
    def log_phi(z):
        denominator=z+(1-sqrt2)*B
        if denominator<=0:raise FloatingPointError("PR logarithm outside its physical domain")
        logratio=math.log1p(2*sqrt2*B/denominator)
        return (bi/bmix)*(z-1)-math.log(z-B)-attraction*logratio
    phis=[log_phi(z) for z in roots]
    if phase=="vapor":idx=len(roots)-1
    elif phase=="liquid":idx=0
    else:
        # Exclude mechanically unstable middle roots before comparing G.
        valid=[]
        for i,z in enumerate(roots):
            v=z*R_SI*temperature/pressure
            dpdv=-R_SI*temperature/(v-bmix)**2+amix*(2*v+2*bmix)/(v*v+2*bmix*v-bmix*bmix)**2
            if dpdv<=0:valid.append(i)
        if not valid:raise FloatingPointError("no mechanically stable PR root resolved")
        idx=min(valid,key=lambda j:float(x@phis[j]))
    z=roots[idx];lp=phis[idx]
    with np.errstate(over="raise",invalid="raise"):
        phi=np.exp(lp);fugacity=x*pressure*phi
    return {"compressibility":z,"physical_roots":roots,"fugacity_coefficients":phi.tolist(),
            "log_fugacity_coefficients":lp.tolist(),"fugacities_pa":fugacity.tolist(),
            "molar_volume_m3_mol":z*R_SI*temperature/pressure,"A":A,"B":B,"phase":phase,
            "residual_gibbs_over_RT":float(x@lp),"flash_performed":False,
            "cubic_residual":float(np.polyval(coefficients,z))}
