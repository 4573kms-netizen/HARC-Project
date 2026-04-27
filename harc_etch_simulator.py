"""
=============================================================================
HARC ETCH PHYSICS-BASED SIMULATOR
Project: Machine Learning-Based HARC Etch Optimization by CF4/Ar Plasma
Phase 1: Physics-based Forward Simulation + Inverse Optimization

Author: Auto-generated scaffold (requires experimental calibration)
Python: 3.9+  |  Dependencies: numpy, scipy, pandas, matplotlib

=============================================================================
PHYSICAL MODELS USED
=============================================================================
1. 0-D Global Plasma Model:
   Simplified power-balance / dissociation-fraction model.
   Estimates F-radical flux, CFx flux, Ar+ ion flux from CF4 fraction,
   source power, and pressure. Parameters are CALIBRATION TARGETS.

2. Sheath / Ion Energy Model:
   Mean ion energy = alpha_E * e * |V_bias| + E_thermal
   (simplified Child-Langmuir sheath approximation)

3. Feature-Scale Transport Model:
   - Ions: exponential attenuation exp(-z / (lambda_ion * CD_top))
     combined with geometric Clausing-like factor 1/(1 + (z/r)^2)
   - Neutrals: Knudsen/Clausing transmission probability for
     long cylinders, T_neutral = 1/(1 + AR_local/2) approximately

4. Surface Reaction Model (per unit area at depth z):
   R_vert(z) = K_chem*Gamma_F(z)
              + K_ie *Gamma_F(z)*Gamma_ion(z)*f_IE(E_ion)
              + K_sput*Gamma_ion(z)*Y_s(E_ion)
              - K_pass*Gamma_CFx(z)
   Lateral rate uses same terms with much smaller coefficients.

5. Bohdansky Sputtering Yield:
   Y_s(E) = Q_s * S_n(eps) * (1 - sqrt(E_th/E))^2   for E > E_th
   (Bohdansky 1984 formula for Si target by F+/Ar+ ions)

6. Profile Evolution:
   Staggered z-grid (dz adjustable), explicit Euler time stepping.
   CD(z,t) tracked; top/mid/bottom CD extracted at each snapshot.

=============================================================================
PARAMETER LABELS
=============================================================================
[FIX]  - Fixed by physics or geometry, not a free parameter
[CAL]  - CALIBRATION TARGET — must be fitted to experimental data
[EST]  - Initial educated estimate; should be verified/updated
[USER] - User-supplied process input

=============================================================================
WARNING
=============================================================================
ALL DEFAULT PARAMETER VALUES ARE INITIAL ASSUMPTIONS OR LITERATURE ESTIMATES.
QUANTITATIVE PREDICTIONS ARE NOT RELIABLE BEFORE CALIBRATION WITH
REAL EXPERIMENTAL DATA (depth, CD_top, CD_bot from ~10 runs).
=============================================================================
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from scipy.optimize import differential_evolution, minimize, least_squares
from scipy.interpolate import interp1d

warnings.warn(
    "\n[HARC SIMULATOR] Default parameters are ASSUMPTIONS, not calibrated values.\n"
    "Quantitative predictions require calibration with experimental data.\n",
    UserWarning,
    stacklevel=2
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: DATACLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProcessConditions:
    """
    [USER] Process input variables supplied by the operator or optimizer.
    All values must be physically realistic; see validate() for range checks.
    """
    # Gas flows
    cf4_flow: float = 15.0          # CF4 flow rate [sccm]
    ar_flow: float = 15.0           # Ar  flow rate [sccm]  CONSTRAINT: cf4+ar=30
    # Electrical
    v_bias: float = -800.0          # DC bias voltage [V]  (must be negative or zero)
    # Source
    source_power: float = 500.0     # ICP / source power [W]
    # Chamber
    pressure: float = 20.0          # Chamber pressure [mTorr]
    # Substrate
    substrate_temp: float = 60.0    # Substrate temperature [°C]
    # Time
    etch_time: float = 120.0        # Etch time [s]
    # Initial geometry
    cd_initial: float = 100.0       # Initial mask opening CD (diameter) [nm]  [FIX per run]
    mask_thickness: float = 300.0   # Mask (photoresist/hard mask) thickness [nm]  [FIX]
    target_depth: float = 1000.0    # Approximate target etch depth [nm]  (informational)

    def validate(self) -> None:
        """Raise ValueError if conditions are outside physically realistic ranges."""
        total_flow = self.cf4_flow + self.ar_flow
        if abs(total_flow - 30.0) > 0.1:
            raise ValueError(
                f"CF4 ({self.cf4_flow}) + Ar ({self.ar_flow}) = {total_flow:.2f} sccm "
                f"!= 30 sccm. Constraint violated."
            )
        if not (0 < self.cf4_flow < 30):
            raise ValueError(f"cf4_flow must be in (0, 30) sccm, got {self.cf4_flow}")
        if not (-3000 <= self.v_bias <= 0):
            raise ValueError(f"v_bias must be in [-3000, 0] V, got {self.v_bias}")
        if not (100 <= self.source_power <= 3000):
            raise ValueError(f"source_power must be in [100, 3000] W, got {self.source_power}")
        if not (1 <= self.pressure <= 500):
            raise ValueError(f"pressure must be in [1, 500] mTorr, got {self.pressure}")
        if not (10 <= self.cd_initial <= 1000):
            raise ValueError(f"cd_initial must be in [10, 1000] nm, got {self.cd_initial}")
        if self.etch_time <= 0:
            raise ValueError(f"etch_time must be positive, got {self.etch_time}")

    @property
    def cf4_fraction(self) -> float:
        """CF4 mole fraction in gas mixture [dimensionless]"""
        return self.cf4_flow / 30.0


@dataclass
class ModelParameters:
    """
    Physics model parameters.
    [CAL] = calibration target (fit to experimental data)
    [EST] = initial literature estimate
    [FIX] = fixed by physics
    """
    # --- 0-D Plasma model parameters ---
    A_F: float = 1.2e17          # [CAL] F-radical flux coefficient [cm-2 s-1 W-0.5 sccm-0.5]
    A_CFx: float = 3.0e16        # [CAL] CFx flux coefficient       [cm-2 s-1 W-0.5 sccm-0.5]
    A_ion: float = 4.0e15        # [CAL] Total ion flux coefficient  [cm-2 s-1 W-0.5 mTorr-0.5]
    beta_F: float = 0.85         # [CAL] CF4 dissociation-to-F efficiency [0–1]
    beta_CFx: float = 0.40       # [CAL] CF4 dissociation-to-CFx efficiency [0–1]
    ar_ion_fraction: float = 0.6 # [EST] Fraction of ion flux that is Ar+ (rest CF+/CF2+)

    # --- Sheath / ion energy model ---
    alpha_E: float = 0.60        # [CAL] Ion energy coupling efficiency [dimensionless]
    E_thermal: float = 5.0       # [FIX] Thermal/plasma potential contribution [eV]
    E_ion_min: float = 15.0      # [EST] Minimum ion energy for ion-enhanced etching [eV]
    E_sput_threshold: float = 20.0 # [EST] Sputtering threshold energy for Si by Ar+ [eV]

    # --- Feature-scale transport model ---
    lambda_ion: float = 2.0      # [CAL] Ion exponential attenuation length in units of local CD
    lambda_neutral: float = 1.5  # [CAL] Neutral exponential attenuation length in units of local CD
    ion_directionality: float = 0.90  # [EST] Ion beam collimation factor [0–1]
    clausing_exponent: float = 1.0    # [CAL] Exponent in Clausing-type factor

    # --- Surface reaction model ---
    K_chem: float = 1.5e-20      # [CAL] Chemical etch rate coefficient [nm cm2 s-1 per radical]
    K_ie: float = 2.8e-35        # [CAL] Ion-enhanced etch coefficient   [nm cm4 s-1 per radical·ion]
    K_sput: float = 1.0e-18      # [CAL] Physical sputtering coefficient [nm cm2 s-1 per ion × yield]
    K_pass: float = 5.0e-21      # [CAL] Passivation coefficient         [nm cm2 s-1 per CFx]
    lateral_ratio: float = 0.08  # [CAL] Lateral/vertical etch rate ratio at surface [dimensionless]
    lateral_ratio_depth: float = 0.02  # [CAL] Lateral/vertical ratio deep in hole

    # --- Sputtering yield (Bohdansky) ---
    Q_s: float = 0.042           # [EST] Bohdansky yield coefficient for Si by Ar+ [dimensionless]
    E_threshold: float = 35.0    # [EST] True sputtering threshold for Si by Ar+ [eV]
    # (literature: ~35 eV for Si/Ar+)

    # --- Profile evolution ---
    dz: float = 20.0             # [FIX] Depth grid spacing [nm]
    dt: float = 0.5              # [FIX] Time step [s]


@dataclass
class SimulationResult:
    """Holds all outputs from a single forward simulation run."""
    # Process conditions used
    conditions: ProcessConditions = field(default_factory=ProcessConditions)
    # Depth grid
    z_grid: np.ndarray = field(default_factory=lambda: np.array([]))  # [nm]
    # Flux profiles (at final time)
    ion_flux_profile: np.ndarray = field(default_factory=lambda: np.array([]))   # [cm-2 s-1]
    neutral_flux_profile: np.ndarray = field(default_factory=lambda: np.array([]))  # [cm-2 s-1]
    cfx_flux_profile: np.ndarray = field(default_factory=lambda: np.array([]))   # [cm-2 s-1]
    # Etch rate profiles
    vert_rate_profile: np.ndarray = field(default_factory=lambda: np.array([]))  # [nm/s]
    lat_rate_profile: np.ndarray = field(default_factory=lambda: np.array([]))   # [nm/s]
    # CD profile (z-resolved) at final time
    cd_profile: np.ndarray = field(default_factory=lambda: np.array([]))  # [nm]
    # Scalar outputs
    total_depth: float = 0.0      # [nm]
    cd_top: float = 0.0           # [nm]
    cd_mid: float = 0.0           # [nm]
    cd_bot: float = 0.0           # [nm]
    aspect_ratio: float = 0.0     # [dimensionless]
    taper_index: float = 0.0      # (cd_top - cd_bot) / cd_top  [positive = tapered]
    bowing_index: float = 0.0     # (cd_max - cd_top) / cd_top  [positive = bowing]
    # Surface fluxes (z=0)
    F_flux_surface: float = 0.0   # [cm-2 s-1]
    ion_flux_surface: float = 0.0 # [cm-2 s-1]
    mean_ion_energy: float = 0.0  # [eV]
    # Time-series snapshots
    depth_vs_time: List[float] = field(default_factory=list)   # [nm]
    cdtop_vs_time: List[float] = field(default_factory=list)   # [nm]
    time_snapshots: List[float] = field(default_factory=list)  # [s]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: 0-D GLOBAL PLASMA MODEL
# ─────────────────────────────────────────────────────────────────────────────

def calc_plasma_fluxes(
    cond: ProcessConditions,
    mp: ModelParameters
) -> Tuple[float, float, float]:
    """
    Simplified 0-D global plasma model.

    Estimates surface-incident fluxes based on:
      - CF4 fraction (controls chemistry / dissociation)
      - Source power (controls plasma density)
      - Pressure (modifies transport / recombination)

    Physical basis:
      F-radical flux ∝ P_source^0.5 * (CF4_flow)^0.5 * beta_F * f_pressure
      CFx flux ∝ similar but with different dissociation branch
      Ion flux ∝ P_source^0.5 * (total_flow)^0.5 / pressure^0.5

    All prefactors (A_F, A_ion, etc.) are [CAL] calibration targets.
    Pressure dependence uses square-root scaling (simplified diffusion limit).

    Parameters
    ----------
    cond : ProcessConditions
    mp   : ModelParameters

    Returns
    -------
    Gamma_F   : F radical flux  [cm-2 s-1]
    Gamma_CFx : CFx flux        [cm-2 s-1]
    Gamma_ion : Total ion flux  [cm-2 s-1]
    """
    P = cond.source_power          # [W]
    cf4_flow = cond.cf4_flow       # [sccm]
    pressure = cond.pressure       # [mTorr]
    total_flow = 30.0              # [sccm]

    # Pressure factor: higher pressure → more recombination → less F reaching surface
    # Using square-root dependence (simplified) [CAL exponent]
    f_pressure_neutral = 1.0 / (1.0 + pressure / 50.0)   # [EST] normalization at 50 mTorr
    f_pressure_ion     = np.sqrt(20.0 / max(pressure, 1.0))  # ions less affected; ref 20 mTorr

    # F-radical flux: from CF4 dissociation
    # More CF4 → more F radicals (up to saturation)
    # Power^0.5 ~ electron density scaling in ICP
    Gamma_F = (
        mp.A_F
        * np.sqrt(P)
        * np.sqrt(cf4_flow)
        * mp.beta_F
        * f_pressure_neutral
    )  # [cm-2 s-1]

    # CFx (passivation) flux: byproduct of CF4 dissociation
    # Peaks at moderate CF4 fraction (polymerizing regime)
    # Uses quadratic attenuation for very high CF4 (depletion)
    cf4_frac = cond.cf4_fraction
    cfx_factor = cf4_frac * (1.0 - 0.5 * cf4_frac)  # heuristic peak ~ CF4_frac=1
    Gamma_CFx = (
        mp.A_CFx
        * np.sqrt(P)
        * np.sqrt(total_flow)
        * mp.beta_CFx
        * cfx_factor
        * f_pressure_neutral
    )  # [cm-2 s-1]

    # Ion flux: scales with power / pressure (ICP density ∝ P/p)
    # Both CF-based ions and Ar+ contribute
    Gamma_ion = (
        mp.A_ion
        * np.sqrt(P)
        * f_pressure_ion
    )  # [cm-2 s-1]

    return float(Gamma_F), float(Gamma_CFx), float(Gamma_ion)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: SHEATH / ION ENERGY MODEL
# ─────────────────────────────────────────────────────────────────────────────

def calc_mean_ion_energy(
    cond: ProcessConditions,
    mp: ModelParameters
) -> float:
    """
    Simplified sheath model for mean ion energy.

    Mean ion energy ≈ alpha_E * e * |V_bias| + E_thermal

    alpha_E accounts for:
      - Collisional energy loss in the sheath
      - Charge-exchange reactions reducing ion energy
      - Multi-species ion distribution broadening

    Physical constraint: E_ion >= E_thermal (even at V_bias = 0)

    Parameters
    ----------
    cond : ProcessConditions
    mp   : ModelParameters

    Returns
    -------
    E_ion : mean ion energy [eV]
    """
    E_ion = mp.alpha_E * abs(cond.v_bias) + mp.E_thermal   # [eV]
    # Pressure correction: higher pressure → more collisions → less energy
    p_factor = 1.0 / (1.0 + (cond.pressure - 20.0) / 200.0)  # [EST] ref at 20 mTorr
    E_ion = max(E_ion * p_factor, mp.E_thermal)
    return float(E_ion)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: FEATURE-SCALE TRANSPORT MODEL
# ─────────────────────────────────────────────────────────────────────────────

def ion_transmission(
    z_array: np.ndarray,
    cd_array: np.ndarray,
    cd_top: float,
    mp: ModelParameters
) -> np.ndarray:
    """
    Ion flux transmission factor as function of depth.

    Physics:
    Ions are highly directional but can be deflected by:
      (a) Mask charging → angular spread
      (b) Geometric shadowing at the hole entrance
      (c) Residual gas scattering inside the hole

    Model (two-component):
      T_ion(z) = ion_directionality
                 * exp(-z / (lambda_ion * cd_top))
                 * geometric_factor(z, r(z))

    Geometric (Clausing-like) factor for a cylinder of depth z,
    local radius r = CD/2:
      G(z, r) = 1 / (1 + (z / r)^clausing_exponent)

    Parameters
    ----------
    z_array  : depth positions [nm]
    cd_array : local CD at each z [nm]
    cd_top   : CD at the top (entrance) [nm]
    mp       : ModelParameters

    Returns
    -------
    T_ion : transmission factor array [0, 1]
    """
    r_array = cd_array / 2.0  # local radius [nm]
    r_array = np.maximum(r_array, 1.0)  # avoid division by zero

    # Exponential attenuation: lambda_ion * cd_top sets the attenuation depth
    exp_factor = np.exp(-z_array / (mp.lambda_ion * cd_top + 1e-6))

    # Clausing-like geometric shadowing
    ar_local = z_array / (cd_array + 1e-6)  # local AR at each depth
    geometric_factor = 1.0 / (1.0 + ar_local ** mp.clausing_exponent)

    T_ion = mp.ion_directionality * exp_factor * geometric_factor
    return np.clip(T_ion, 0.0, 1.0)


def neutral_transmission(
    z_array: np.ndarray,
    cd_array: np.ndarray,
    cd_top: float,
    mp: ModelParameters
) -> np.ndarray:
    """
    Neutral radical (F, CFx) transmission factor as function of depth.

    Physics:
    Neutrals travel diffusively (Knudsen regime inside narrow holes).
    For a long cylinder (AR >> 1):
      T_Knudsen(AR) ≈ 1 / (1 + AR/2)    [Clausing 1932]

    We use this as the base and add exponential attenuation for
    wall recombination effects:
      T_neutral(z) = T_Knudsen(AR_z)
                     * exp(-z / (lambda_neutral * cd_top))

    Note: T_neutral ≠ T_ion at the same depth (different physics).

    Parameters
    ----------
    z_array  : depth positions [nm]
    cd_array : local CD [nm]
    cd_top   : top CD [nm]
    mp       : ModelParameters

    Returns
    -------
    T_neutral : transmission factor array [0, 1]
    """
    ar_local = z_array / (cd_array + 1e-6)
    # Clausing transmission for long cylinder
    T_clausing = 1.0 / (1.0 + ar_local / 2.0)

    # Additional exponential attenuation: wall recombination / sticking
    exp_factor = np.exp(-z_array / (mp.lambda_neutral * cd_top + 1e-6))

    T_neutral = T_clausing * exp_factor
    return np.clip(T_neutral, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: SURFACE REACTION MODEL
# ─────────────────────────────────────────────────────────────────────────────

def sputtering_yield(
    E_ion: float,
    mp: ModelParameters
) -> float:
    """
    Bohdansky (1984) sputtering yield formula.

    Y_s(E) = Q_s * S_n(eps) * (1 - sqrt(E_th / E))^s
    Simplified form used here (s ≈ 2.5 for Si/Ar+):
      Y_s(E) = Q_s * (1 - sqrt(E_th / E))^2   for E > E_th
             = 0                                 for E <= E_th

    Parameters
    ----------
    E_ion : mean ion energy [eV]
    mp    : ModelParameters (Q_s, E_threshold)

    Returns
    -------
    Y_s : sputtering yield [atoms / ion]  (dimensionless here, absorbed into K_sput)
    """
    if E_ion <= mp.E_threshold:
        return 0.0
    Y_s = mp.Q_s * (1.0 - np.sqrt(mp.E_threshold / E_ion)) ** 2
    return max(Y_s, 0.0)


def ion_enhanced_factor(
    E_ion: float,
    mp: ModelParameters
) -> float:
    """
    Ion-enhanced etching energy factor f_IE(E_ion).

    Models the increase in chemical etch rate when ions supply
    activation energy to adsorbed F radicals.

    f_IE(E) = max(0,  (E_ion - E_ion_min)^0.5 )  [EST]
    (threshold-based, square-root energy scaling)

    Returns
    -------
    f_IE : dimensionless energy enhancement factor [>=0]
    """
    if E_ion <= mp.E_ion_min:
        return 0.0
    return np.sqrt(max(E_ion - mp.E_ion_min, 0.0))


def calc_vertical_etch_rate(
    Gamma_F_z: np.ndarray,
    Gamma_ion_z: np.ndarray,
    Gamma_CFx_z: np.ndarray,
    E_ion: float,
    mp: ModelParameters
) -> np.ndarray:
    """
    Local vertical etch rate at each depth z.

    R_vert(z) = K_chem  * Gamma_F(z)
              + K_ie    * Gamma_F(z) * Gamma_ion(z) * f_IE(E_ion)
              + K_sput  * Gamma_ion(z) * Y_s(E_ion)
              - K_pass  * Gamma_CFx(z)

    All terms are physically motivated:
      - Chemical:       spontaneous F etching of Si
      - Ion-enhanced:   F etch activated by ion bombardment
      - Sputtering:     direct Ar+/CF+ sputtering
      - Passivation:    fluorocarbon film suppresses etching

    Rate is clipped at 0 (no deposition modeled here).

    Units check:
      K_chem [nm·cm²/s] * Gamma_F [cm⁻²s⁻¹] → [nm/s] ✓
      K_ie [nm·cm⁴/s]   * Gamma_F * Gamma_ion → [nm/s] ✓

    Parameters
    ----------
    Gamma_F_z   : F-radical flux at each z  [cm-2 s-1]
    Gamma_ion_z : Ion flux at each z         [cm-2 s-1]
    Gamma_CFx_z : CFx flux at each z         [cm-2 s-1]
    E_ion       : mean ion energy            [eV]
    mp          : ModelParameters

    Returns
    -------
    R_vert : vertical etch rate [nm/s] at each z
    """
    Y_s = sputtering_yield(E_ion, mp)
    f_IE = ion_enhanced_factor(E_ion, mp)

    R_chem  = mp.K_chem  * Gamma_F_z
    R_ie    = mp.K_ie    * Gamma_F_z * Gamma_ion_z * f_IE
    R_sput  = mp.K_sput  * Gamma_ion_z * Y_s
    R_pass  = mp.K_pass  * Gamma_CFx_z

    R_vert = R_chem + R_ie + R_sput - R_pass
    return np.maximum(R_vert, 0.0)   # no negative etch rate (no deposition modeled)


def calc_lateral_etch_rate(
    R_vert_z: np.ndarray,
    z_array: np.ndarray,
    depth_current: float,
    mp: ModelParameters
) -> np.ndarray:
    """
    Local lateral (sidewall) etch rate at each depth z.

    Lateral etching is driven by the same fluxes but:
      - Ions are mostly vertical → minimal lateral contribution
      - Neutrals isotropic → contribute more to lateral than ions do

    Model: R_lat(z) = R_vert(z) * lateral_coeff(z)
    where lateral_coeff decreases with depth (less neutral flux sideways deep in hole)

    lateral_coeff(z) = lateral_ratio * exp(-z / (depth_current + 1e-6) * 3)
                       → interpolates between lateral_ratio at surface
                          and lateral_ratio_depth deep in hole

    Parameters
    ----------
    R_vert_z      : vertical etch rate profile [nm/s]
    z_array       : depth positions [nm]
    depth_current : current etch depth [nm]
    mp            : ModelParameters

    Returns
    -------
    R_lat : lateral etch rate [nm/s] at each z
    """
    if depth_current < 1.0:
        depth_current = 1.0
    # Smooth interpolation from surface ratio to deep ratio
    frac = np.clip(z_array / depth_current, 0.0, 1.0)
    lat_coeff = mp.lateral_ratio * (1.0 - frac) + mp.lateral_ratio_depth * frac
    return R_vert_z * lat_coeff


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: FORWARD SIMULATION (PROFILE EVOLUTION)
# ─────────────────────────────────────────────────────────────────────────────

def run_forward_simulation(
    cond: ProcessConditions,
    mp: ModelParameters,
    verbose: bool = False
) -> SimulationResult:
    """
    Full physics-based forward simulation of HARC etch profile evolution.

    Algorithm:
    ---------
    1. Validate process conditions.
    2. Calculate surface (z=0) plasma fluxes and ion energy.
    3. Initialize z-grid and CD(z) array.
    4. Time-step loop:
       a. Determine current etch front depth.
       b. Build z-grid from 0 to current depth.
       c. Interpolate CD(z) onto current grid.
       d. Compute transmission factors T_ion(z), T_neutral(z).
       e. Compute local fluxes at each z.
       f. Compute local etch rates R_vert(z), R_lat(z).
       g. Update depth: z_front += R_vert_bottom * dt
       h. Update CD profile: CD(z) += 2 * R_lat(z) * dt  (both walls)
          (factor 2 because lateral etch widens diameter from both sides)
       i. Record snapshots.
    5. Extract final profile metrics.

    Parameters
    ----------
    cond    : ProcessConditions
    mp      : ModelParameters
    verbose : print progress if True

    Returns
    -------
    SimulationResult with all computed fields populated.
    """
    cond.validate()

    result = SimulationResult(conditions=cond)

    # --- Step 2: Surface fluxes ---
    Gamma_F_surf, Gamma_CFx_surf, Gamma_ion_surf = calc_plasma_fluxes(cond, mp)
    E_ion = calc_mean_ion_energy(cond, mp)

    result.F_flux_surface   = Gamma_F_surf
    result.ion_flux_surface = Gamma_ion_surf
    result.mean_ion_energy  = E_ion

    if verbose:
        print(f"  F-radical flux (surface): {Gamma_F_surf:.3e} cm-2 s-1")
        print(f"  Ion flux (surface):       {Gamma_ion_surf:.3e} cm-2 s-1")
        print(f"  Mean ion energy:          {E_ion:.1f} eV")
        print(f"  CF4 fraction:             {cond.cf4_fraction:.2f}")

    # --- Step 3: Initialize ---
    dz = mp.dz          # [nm]
    dt = mp.dt          # [s]
    t_end = cond.etch_time
    n_steps = int(t_end / dt)
    dt = t_end / n_steps  # adjust for exact time

    # Current etch depth (front position)
    depth_current = 0.0   # [nm]
    cd_top_current = cond.cd_initial   # [nm]

    # Store CD(z) as dict: key=z_index, val=CD [nm]
    # We will use a fixed fine z-grid up to a max possible depth
    max_depth_est = max(cond.target_depth * 1.5, 500.0)  # [nm]
    N_z_max = int(max_depth_est / dz) + 2
    z_full = np.arange(N_z_max) * dz   # [nm]
    cd_full = np.ones(N_z_max) * cond.cd_initial  # start all at initial CD

    # Time snapshots
    snap_interval = max(1, n_steps // 10)
    depth_vs_t = []
    cdtop_vs_t = []
    time_snaps = []

    # --- Step 4: Time loop ---
    for step in range(n_steps):
        t_current = step * dt

        # Current etch front index on z_full grid
        n_active = max(int(depth_current / dz) + 1, 2)
        n_active = min(n_active, N_z_max - 1)

        z_active = z_full[:n_active]           # active z-grid [nm]
        cd_active = cd_full[:n_active].copy()  # local CD [nm]
        cd_top_now = cd_active[0]

        # Transmission factors
        T_ion_arr     = ion_transmission(z_active, cd_active, cd_top_now, mp)
        T_neutral_arr = neutral_transmission(z_active, cd_active, cd_top_now, mp)

        # Local fluxes
        Gamma_F_z   = Gamma_F_surf   * T_neutral_arr   # [cm-2 s-1]
        Gamma_ion_z = Gamma_ion_surf * T_ion_arr        # [cm-2 s-1]
        Gamma_CFx_z = Gamma_CFx_surf * T_neutral_arr   # [cm-2 s-1]

        # Local etch rates
        R_vert_z = calc_vertical_etch_rate(
            Gamma_F_z, Gamma_ion_z, Gamma_CFx_z, E_ion, mp
        )  # [nm/s]
        R_lat_z = calc_lateral_etch_rate(
            R_vert_z, z_active, max(depth_current, 1.0), mp
        )  # [nm/s]

        # Update etch depth (advance bottom of hole)
        depth_advance = R_vert_z[-1] * dt   # bottom-most point
        depth_current += depth_advance

        # Update CD profile: lateral etch widens hole (both sidewalls)
        cd_full[:n_active] += 2.0 * R_lat_z * dt

        # Clamp CD to reasonable range
        cd_full[:n_active] = np.clip(cd_full[:n_active], 1.0, cond.cd_initial * 3.0)

        # Snapshots
        if step % snap_interval == 0 or step == n_steps - 1:
            depth_vs_t.append(depth_current)
            cdtop_vs_t.append(cd_full[0])
            time_snaps.append(t_current)

    # --- Step 5: Final profile extraction ---
    n_final = max(int(depth_current / dz) + 1, 2)
    n_final = min(n_final, N_z_max - 1)

    z_final   = z_full[:n_final]
    cd_final  = cd_full[:n_final]

    # Recompute final flux profiles for output
    cd_top_final = cd_final[0]
    T_ion_final     = ion_transmission(z_final, cd_final, cd_top_final, mp)
    T_neutral_final = neutral_transmission(z_final, cd_final, cd_top_final, mp)

    Gamma_F_final   = Gamma_F_surf   * T_neutral_final
    Gamma_ion_final = Gamma_ion_surf * T_ion_final
    Gamma_CFx_final = Gamma_CFx_surf * T_neutral_final

    R_vert_final = calc_vertical_etch_rate(
        Gamma_F_final, Gamma_ion_final, Gamma_CFx_final, E_ion, mp
    )
    R_lat_final = calc_lateral_etch_rate(
        R_vert_final, z_final, max(depth_current, 1.0), mp
    )

    # Scalar metrics
    cd_top = cd_final[0]
    cd_bot = cd_final[-1]
    n_mid  = len(cd_final) // 2
    cd_mid = cd_final[n_mid]

    AR = depth_current / max(cd_bot, 1.0)

    taper_index = (cd_top - cd_bot) / max(cd_top, 1.0)
    cd_max = np.max(cd_final)
    bowing_index = (cd_max - cd_top) / max(cd_top, 1.0)

    # Fill result
    result.z_grid             = z_final
    result.ion_flux_profile   = Gamma_ion_final
    result.neutral_flux_profile = Gamma_F_final
    result.cfx_flux_profile   = Gamma_CFx_final
    result.vert_rate_profile  = R_vert_final
    result.lat_rate_profile   = R_lat_final
    result.cd_profile         = cd_final
    result.total_depth        = depth_current
    result.cd_top             = cd_top
    result.cd_mid             = cd_mid
    result.cd_bot             = cd_bot
    result.aspect_ratio       = AR
    result.taper_index        = taper_index
    result.bowing_index       = bowing_index
    result.depth_vs_time      = depth_vs_t
    result.cdtop_vs_time      = cdtop_vs_t
    result.time_snapshots     = time_snaps

    if verbose:
        print(f"\n  === SIMULATION RESULT ===")
        print(f"  Total depth:    {depth_current:.1f} nm")
        print(f"  CD_top:         {cd_top:.2f} nm")
        print(f"  CD_mid:         {cd_mid:.2f} nm")
        print(f"  CD_bot:         {cd_bot:.2f} nm")
        print(f"  Aspect Ratio:   {AR:.2f}")
        print(f"  Taper index:    {taper_index:.4f}  (>0 = top wider)")
        print(f"  Bowing index:   {bowing_index:.4f} (>0 = bowing)")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: INVERSE OPTIMIZATION
# ─────────────────────────────────────────────────────────────────────────────

def objective_function(
    x: np.ndarray,
    mp: ModelParameters,
    base_cond: ProcessConditions,
    target_AR: float = 10.0,
    w_AR: float = 10.0,
    w_taper: float = 5.0,
    w_bowing: float = 3.0,
    w_depth: float = 0.5
) -> float:
    """
    Objective function for inverse optimization.

    Variables:
      x[0] = cf4_flow  [sccm]      bounds: (2, 28)
      x[1] = v_bias    [V, negative]  bounds: (-2500, -100)

    Ar flow = 30 - cf4_flow  (constraint)

    Objective:
      J = w_AR    * (AR - target_AR)^2
        + w_taper * taper_index^2
        + w_bowing* bowing_index^2
        + w_depth * max(0, 200 - depth)^2   (prefer depth > 200 nm)
        + penalty terms for nonphysical results

    Parameters
    ----------
    x         : optimization variable array [cf4_flow, v_bias]
    mp        : ModelParameters
    base_cond : ProcessConditions (other params taken from here)
    target_AR : desired aspect ratio
    w_AR, w_taper, w_bowing, w_depth : loss weights

    Returns
    -------
    J : scalar objective (lower = better)
    """
    cf4_flow = float(np.clip(x[0], 1.0, 29.0))
    v_bias   = float(np.clip(x[1], -2900.0, -50.0))

    try:
        cond = ProcessConditions(
            cf4_flow     = cf4_flow,
            ar_flow      = 30.0 - cf4_flow,
            v_bias       = v_bias,
            source_power = base_cond.source_power,
            pressure     = base_cond.pressure,
            substrate_temp = base_cond.substrate_temp,
            etch_time    = base_cond.etch_time,
            cd_initial   = base_cond.cd_initial,
            mask_thickness = base_cond.mask_thickness,
            target_depth = base_cond.target_depth,
        )
        cond.validate()
        res = run_forward_simulation(cond, mp, verbose=False)

        J  = w_AR    * (res.aspect_ratio - target_AR) ** 2
        J += w_taper * res.taper_index ** 2
        J += w_bowing* res.bowing_index ** 2

        # Prefer reasonable depth (at least 200 nm)
        if res.total_depth < 200.0:
            J += w_depth * (200.0 - res.total_depth) ** 2

        # Penalize physically degenerate results
        if res.total_depth < 10.0 or res.cd_bot < 1.0:
            J += 1e6

    except (ValueError, RuntimeError, FloatingPointError):
        J = 1e8

    return float(J)


def optimize_process_conditions(
    mp: ModelParameters,
    base_cond: ProcessConditions,
    target_AR: float = 10.0,
    w_AR: float = 10.0,
    w_taper: float = 5.0,
    w_bowing: float = 3.0,
    verbose: bool = True
) -> Tuple[ProcessConditions, SimulationResult, Dict]:
    """
    Physics-based inverse optimization to find CF4/Ar/Vbias for target AR.

    Strategy:
    ---------
    Step 1: Global search using Differential Evolution
            → finds basin of attraction, robust to local minima
    Step 2: Local refinement using L-BFGS-B from best global solution
            → fine-tunes the solution

    Search space:
      cf4_flow : [2, 28] sccm   (Ar = 30 - cf4)
      v_bias   : [-2500, -100] V

    Parameters
    ----------
    mp         : ModelParameters (fixed during optimization)
    base_cond  : ProcessConditions (source_power, pressure, etch_time, etc.)
    target_AR  : target aspect ratio [default 10]
    w_AR/taper/bowing : objective weights
    verbose    : print progress

    Returns
    -------
    best_cond    : ProcessConditions with optimal CF4/Ar/Vbias
    best_result  : SimulationResult for the best conditions
    opt_info     : dict with optimization diagnostic info
    """
    bounds = [
        (2.0,  28.0),      # cf4_flow [sccm]
        (-2500.0, -100.0)  # v_bias [V]
    ]

    obj_kwargs = dict(
        mp=mp, base_cond=base_cond, target_AR=target_AR,
        w_AR=w_AR, w_taper=w_taper, w_bowing=w_bowing
    )

    if verbose:
        print("=" * 60)
        print(f"  INVERSE OPTIMIZATION  |  Target AR = {target_AR}")
        print("  Step 1: Differential Evolution (global search)...")

    # Step 1: Global search
    de_result = differential_evolution(
        objective_function,
        bounds=bounds,
        args=(mp, base_cond, target_AR, w_AR, w_taper, w_bowing),
        maxiter=200,
        tol=1e-4,
        seed=42,
        popsize=12,
        mutation=(0.5, 1.5),
        recombination=0.7,
        polish=False,
        disp=False
    )

    x0_local = de_result.x

    if verbose:
        print(f"  Step 1 complete. Best J (global) = {de_result.fun:.4f}")
        print("  Step 2: L-BFGS-B local refinement...")

    # Step 2: Local refinement
    lbfgs_result = minimize(
        objective_function,
        x0=x0_local,
        args=(mp, base_cond, target_AR, w_AR, w_taper, w_bowing),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-7}
    )

    x_best = lbfgs_result.x
    cf4_best  = float(np.clip(x_best[0], 2.0, 28.0))
    vbias_best = float(np.clip(x_best[1], -2500.0, -100.0))

    if verbose:
        print(f"  Step 2 complete. Best J (local)  = {lbfgs_result.fun:.6f}")
        print(f"\n  OPTIMAL RECIPE:")
        print(f"    CF4 flow  = {cf4_best:.2f} sccm")
        print(f"    Ar  flow  = {30.0 - cf4_best:.2f} sccm")
        print(f"    V_bias    = {vbias_best:.1f} V")

    best_cond = ProcessConditions(
        cf4_flow     = cf4_best,
        ar_flow      = 30.0 - cf4_best,
        v_bias       = vbias_best,
        source_power = base_cond.source_power,
        pressure     = base_cond.pressure,
        substrate_temp = base_cond.substrate_temp,
        etch_time    = base_cond.etch_time,
        cd_initial   = base_cond.cd_initial,
        mask_thickness = base_cond.mask_thickness,
        target_depth = base_cond.target_depth,
    )

    best_result = run_forward_simulation(best_cond, mp, verbose=verbose)

    opt_info = {
        'de_result'   : de_result,
        'lbfgs_result': lbfgs_result,
        'final_J'     : float(lbfgs_result.fun),
        'n_eval_total': de_result.nfev + lbfgs_result.nfev,
    }

    return best_cond, best_result, opt_info


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: CALIBRATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_model_parameters(
    experimental_data: pd.DataFrame,
    mp_init: ModelParameters,
    calibrate_params: Optional[List[str]] = None,
    verbose: bool = True
) -> ModelParameters:
    """
    Calibrate model parameters using experimental etch data.

    Experimental data DataFrame must contain columns:
      cf4_flow  [sccm]
      ar_flow   [sccm]
      v_bias    [V]
      depth_meas       [nm]   measured etch depth
      cd_top_meas      [nm]   measured top CD
      cd_bot_meas      [nm]   measured bottom CD

    Optional columns (ignored if absent):
      source_power, pressure, etch_time, cd_initial

    Calibration method: scipy.optimize.least_squares (TRF algorithm)
    Objective: minimize sum of squared residuals for depth, cd_top, cd_bot.

    Parameters to calibrate (default):
      A_F, A_ion, alpha_E, lambda_ion, lambda_neutral,
      K_chem, K_ie, K_sput, K_pass

    Parameters
    ----------
    experimental_data : pd.DataFrame with experiment columns
    mp_init           : initial ModelParameters (starting point)
    calibrate_params  : list of parameter names to calibrate (None = default set)
    verbose           : print calibration progress

    Returns
    -------
    mp_calibrated : ModelParameters with updated values
    """
    import copy

    required_cols = ['cf4_flow', 'ar_flow', 'v_bias', 'depth_meas', 'cd_top_meas', 'cd_bot_meas']
    for col in required_cols:
        if col not in experimental_data.columns:
            raise ValueError(f"Experimental data missing required column: '{col}'")

    # Default set of calibration parameters
    if calibrate_params is None:
        calibrate_params = [
            'A_F', 'A_ion', 'alpha_E',
            'lambda_ion', 'lambda_neutral',
            'K_chem', 'K_ie', 'K_sput', 'K_pass'
        ]

    # Extract initial values and bounds
    x0 = []
    bounds_lo = []
    bounds_hi = []
    param_scales = {
        'A_F':             (1e15, 1e19),
        'A_CFx':           (1e14, 1e18),
        'A_ion':           (1e13, 1e17),
        'beta_F':          (0.1, 1.0),
        'beta_CFx':        (0.05, 1.0),
        'ar_ion_fraction': (0.1, 0.95),
        'alpha_E':         (0.1, 1.0),
        'E_thermal':       (1.0, 30.0),
        'lambda_ion':      (0.5, 20.0),
        'lambda_neutral':  (0.5, 20.0),
        'K_chem':          (1e-23, 1e-17),
        'K_ie':            (1e-38, 1e-31),
        'K_sput':          (1e-21, 1e-15),
        'K_pass':          (1e-24, 1e-18),
        'lateral_ratio':   (0.001, 0.3),
    }

    mp_work = copy.deepcopy(mp_init)
    for pname in calibrate_params:
        val = getattr(mp_work, pname)
        lo, hi = param_scales.get(pname, (val * 0.01, val * 100.0))
        x0.append(val)
        bounds_lo.append(lo)
        bounds_hi.append(hi)

    x0 = np.array(x0)
    bounds = (bounds_lo, bounds_hi)

    # Build experiment conditions list
    experiments = []
    for _, row in experimental_data.iterrows():
        cond = ProcessConditions(
            cf4_flow     = float(row['cf4_flow']),
            ar_flow      = float(row['ar_flow']),
            v_bias       = float(row['v_bias']),
            source_power = float(row.get('source_power', mp_init.A_F)),
            pressure     = float(row.get('pressure', 20.0)),
            etch_time    = float(row.get('etch_time', 120.0)),
            cd_initial   = float(row.get('cd_initial', 100.0)),
        )
        measurements = {
            'depth'  : float(row['depth_meas']),
            'cd_top' : float(row['cd_top_meas']),
            'cd_bot' : float(row['cd_bot_meas']),
        }
        experiments.append((cond, measurements))

    def residual_function(x_cal: np.ndarray) -> np.ndarray:
        """Compute residuals for all experiments given parameter vector x_cal."""
        mp_try = copy.deepcopy(mp_work)
        for i, pname in enumerate(calibrate_params):
            setattr(mp_try, pname, float(x_cal[i]))

        residuals = []
        for cond_exp, meas in experiments:
            try:
                res = run_forward_simulation(cond_exp, mp_try, verbose=False)
                # Normalize residuals by measurement magnitude
                r_depth  = (res.total_depth - meas['depth'])  / max(meas['depth'],  10.0)
                r_cdtop  = (res.cd_top       - meas['cd_top']) / max(meas['cd_top'],  5.0)
                r_cdbot  = (res.cd_bot       - meas['cd_bot']) / max(meas['cd_bot'],  5.0)
                residuals.extend([r_depth, r_cdtop, r_cdbot])
            except Exception:
                residuals.extend([1e3, 1e3, 1e3])
        return np.array(residuals)

    if verbose:
        print("=" * 60)
        print(f"  CALIBRATING {len(calibrate_params)} parameters against "
              f"{len(experiments)} experiments...")
        print(f"  Parameters: {calibrate_params}")

    # Run calibration
    cal_result = least_squares(
        residual_function,
        x0=x0,
        bounds=bounds,
        method='trf',
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        max_nfev=5000,
        verbose=2 if verbose else 0
    )

    # Update model parameters
    mp_calibrated = copy.deepcopy(mp_init)
    for i, pname in enumerate(calibrate_params):
        old_val = getattr(mp_init, pname)
        new_val = float(cal_result.x[i])
        setattr(mp_calibrated, pname, new_val)
        if verbose:
            print(f"  {pname:20s}: {old_val:.4e}  →  {new_val:.4e}")

    if verbose:
        print(f"\n  Calibration cost (final): {cal_result.cost:.4e}")
        print(f"  Optimality:               {cal_result.optimality:.4e}")
        print(f"  n_function_eval:          {cal_result.nfev}")

    return mp_calibrated


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: VISUALIZATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    'primary'   : '#2563EB',
    'secondary' : '#DC2626',
    'accent'    : '#16A34A',
    'warn'      : '#D97706',
    'purple'    : '#7C3AED',
    'bg'        : '#F8FAFC',
}

def _style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_facecolor(COLORS['bg'])


def plot_simulation_result(result: SimulationResult, save_path: Optional[str] = None):
    """
    Plot 1: Full simulation result overview (4 panels).

    Panels:
      (a) CD profile vs depth — cross-section shape
      (b) Local etch rate vs depth — vertical & lateral
      (c) Flux attenuation vs depth — F-radical, Ion, CFx
      (d) Time evolution — depth and top-CD vs time
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(
        f"HARC Etch Simulation Result\n"
        f"CF4={result.conditions.cf4_flow:.1f} sccm | "
        f"Ar={result.conditions.ar_flow:.1f} sccm | "
        f"Vbias={result.conditions.v_bias:.0f} V  →  "
        f"AR={result.aspect_ratio:.2f} | "
        f"Depth={result.total_depth:.0f} nm",
        fontsize=11, fontweight='bold'
    )

    z  = result.z_grid
    cd = result.cd_profile

    # --- Panel (a): CD profile (cross-section) ---
    ax = axes[0]
    ax.plot(-cd / 2, -z, color=COLORS['primary'], lw=2, label='Left wall')
    ax.plot( cd / 2, -z, color=COLORS['primary'], lw=2, label='Right wall')
    ax.fill_betweenx(-z, -cd/2, cd/2, alpha=0.12, color=COLORS['primary'])
    ax.axhline(0, color='gray', lw=1, ls='--', alpha=0.5)
    # Mark key points
    ax.scatter([-result.cd_top/2, result.cd_top/2], [0, 0],
               color=COLORS['secondary'], s=40, zorder=5, label=f'Top CD={result.cd_top:.1f}nm')
    ax.scatter([-result.cd_bot/2, result.cd_bot/2],
               [-result.total_depth, -result.total_depth],
               color=COLORS['accent'], s=40, zorder=5, label=f'Bot CD={result.cd_bot:.1f}nm')
    _style_ax(ax, 'Cross-Section Profile', 'x position [nm]', 'Depth [nm]')
    ax.legend(fontsize=7)

    # --- Panel (b): Etch rate vs depth ---
    ax = axes[1]
    ax.plot(result.vert_rate_profile, -z, color=COLORS['secondary'], lw=2, label='Vertical')
    ax.plot(result.lat_rate_profile,  -z, color=COLORS['accent'],    lw=2, ls='--', label='Lateral')
    _style_ax(ax, 'Local Etch Rate vs Depth', 'Etch Rate [nm/s]', 'Depth [nm]')
    ax.legend(fontsize=8)

    # --- Panel (c): Flux attenuation ---
    ax = axes[2]
    # Normalize to surface value for comparison
    norm_F   = result.neutral_flux_profile / (result.F_flux_surface   + 1e-30)
    norm_ion = result.ion_flux_profile     / (result.ion_flux_surface  + 1e-30)
    norm_cfx = result.cfx_flux_profile     / (result.F_flux_surface    + 1e-30)
    ax.plot(norm_F,   -z, color=COLORS['primary'],   lw=2, label='F radical')
    ax.plot(norm_ion, -z, color=COLORS['secondary'], lw=2, label='Ion')
    ax.plot(norm_cfx, -z, color=COLORS['purple'],    lw=2, ls=':', label='CFx')
    _style_ax(ax, 'Flux Attenuation vs Depth', 'Normalized Flux [-]', 'Depth [nm]')
    ax.legend(fontsize=8)

    # --- Panel (d): Time evolution ---
    ax = axes[3]
    t = result.time_snapshots
    ax2 = ax.twinx()
    ax.plot(t, result.depth_vs_time,  color=COLORS['primary'],   lw=2, label='Depth')
    ax2.plot(t, result.cdtop_vs_time, color=COLORS['secondary'], lw=2, ls='--', label='Top CD')
    ax.set_xlabel('Time [s]', fontsize=9)
    ax.set_ylabel('Depth [nm]', fontsize=9, color=COLORS['primary'])
    ax2.set_ylabel('Top CD [nm]', fontsize=9, color=COLORS['secondary'])
    ax.set_title('Time Evolution', fontsize=11, fontweight='bold', pad=8)
    ax.grid(True, alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def plot_process_window(
    mp: ModelParameters,
    base_cond: ProcessConditions,
    cf4_range: Tuple[float, float] = (5, 25),
    vbias_range: Tuple[float, float] = (-2000, -200),
    n_cf4: int = 15,
    n_vbias: int = 15,
    save_path: Optional[str] = None
):
    """
    Plot 2: Process window — AR, depth, taper as heatmaps over
            CF4 fraction × |Vbias| space.
    """
    cf4_vals  = np.linspace(cf4_range[0], cf4_range[1], n_cf4)
    vbias_vals = np.linspace(vbias_range[0], vbias_range[1], n_vbias)

    AR_map     = np.zeros((n_vbias, n_cf4))
    depth_map  = np.zeros((n_vbias, n_cf4))
    taper_map  = np.zeros((n_vbias, n_cf4))

    print("  Computing process window (may take a moment)...")
    for i, vb in enumerate(vbias_vals):
        for j, cf4 in enumerate(cf4_vals):
            try:
                cond = ProcessConditions(
                    cf4_flow=cf4, ar_flow=30.0-cf4, v_bias=vb,
                    source_power=base_cond.source_power,
                    pressure=base_cond.pressure,
                    etch_time=base_cond.etch_time,
                    cd_initial=base_cond.cd_initial,
                )
                res = run_forward_simulation(cond, mp, verbose=False)
                AR_map[i, j]    = res.aspect_ratio
                depth_map[i, j] = res.total_depth
                taper_map[i, j] = res.taper_index
            except Exception:
                AR_map[i, j] = np.nan
                depth_map[i, j] = np.nan
                taper_map[i, j] = np.nan

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Process Window: AR / Depth / Taper vs CF4 & V_bias',
                 fontsize=12, fontweight='bold')

    extent = [cf4_range[0], cf4_range[1], abs(vbias_range[0]), abs(vbias_range[1])]

    for ax, data, title, cmap in zip(
        axes,
        [AR_map, depth_map, taper_map],
        ['Aspect Ratio', 'Etch Depth [nm]', 'Taper Index'],
        ['RdYlGn', 'viridis', 'RdYlBu_r']
    ):
        im = ax.imshow(data, origin='lower', aspect='auto',
                       extent=[cf4_range[0], cf4_range[1],
                               abs(vbias_range[1]), abs(vbias_range[0])],
                       cmap=cmap)
        plt.colorbar(im, ax=ax, shrink=0.85)
        if title == 'Aspect Ratio':
            # Contour at AR=10
            cs = ax.contour(
                np.linspace(cf4_range[0], cf4_range[1], n_cf4),
                np.abs(np.linspace(vbias_range[0], vbias_range[1], n_vbias)),
                data, levels=[10.0], colors='white', linewidths=2
            )
            ax.clabel(cs, fmt='AR=10', fontsize=8, colors='white')
        _style_ax(ax, title, 'CF4 Flow [sccm]', '|V_bias| [V]')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def plot_optimization_result(
    best_cond: ProcessConditions,
    best_result: SimulationResult,
    opt_info: Dict,
    save_path: Optional[str] = None
):
    """
    Plot 3: Optimization result — recommended profile + key metrics.
    """
    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # (a) Cross-section of optimal profile
    ax1 = fig.add_subplot(gs[0])
    z  = best_result.z_grid
    cd = best_result.cd_profile
    ax1.fill_betweenx(-z, -cd/2, cd/2, alpha=0.25, color=COLORS['primary'])
    ax1.plot(-cd/2, -z, color=COLORS['primary'], lw=2.5)
    ax1.plot( cd/2, -z, color=COLORS['primary'], lw=2.5)
    # Mask
    ax1.fill_betweenx(
        [0, best_cond.mask_thickness],
        [-best_cond.cd_initial/2 * 2, -best_cond.cd_initial/2],
        [best_cond.cd_initial/2 * 2,  best_cond.cd_initial/2],
        alpha=0.3, color='gray'
    )
    ax1.set_ylim(-best_result.total_depth * 1.15, best_cond.mask_thickness * 1.5)
    _style_ax(ax1, f'Optimal Profile\nAR={best_result.aspect_ratio:.2f}',
              'x [nm]', 'z [nm]')

    # (b) Key metrics bar chart
    ax2 = fig.add_subplot(gs[1])
    metrics = {
        'AR / 10': best_result.aspect_ratio / 10.0,
        'Taper×10': best_result.taper_index * 10.0,
        'Bowing×10': best_result.bowing_index * 10.0,
    }
    bars = ax2.bar(metrics.keys(), metrics.values(),
                   color=[COLORS['primary'], COLORS['warn'], COLORS['secondary']],
                   alpha=0.8, edgecolor='black', linewidth=0.7)
    ax2.axhline(1.0, color=COLORS['primary'], ls='--', lw=1.5, label='Target AR/10=1')
    ax2.axhline(0.0, color='black', lw=0.5)
    for bar, val in zip(bars, metrics.values()):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    _style_ax(ax2, 'Normalized Metrics', 'Metric', 'Normalized value')
    ax2.legend(fontsize=8)

    # (c) Recipe summary table
    ax3 = fig.add_subplot(gs[2])
    ax3.axis('off')
    table_data = [
        ['Parameter', 'Optimal Value'],
        ['CF4 flow', f'{best_cond.cf4_flow:.2f} sccm'],
        ['Ar  flow', f'{best_cond.ar_flow:.2f} sccm'],
        ['CF4 fraction', f'{best_cond.cf4_fraction:.3f}'],
        ['V_bias', f'{best_cond.v_bias:.1f} V'],
        ['─'*15, '─'*15],
        ['Depth', f'{best_result.total_depth:.1f} nm'],
        ['CD_top', f'{best_result.cd_top:.2f} nm'],
        ['CD_mid', f'{best_result.cd_mid:.2f} nm'],
        ['CD_bot', f'{best_result.cd_bot:.2f} nm'],
        ['Aspect Ratio', f'{best_result.aspect_ratio:.3f}'],
        ['Taper index', f'{best_result.taper_index:.4f}'],
        ['Bowing index', f'{best_result.bowing_index:.4f}'],
        ['─'*15, '─'*15],
        ['Obj. J (final)', f'{opt_info["final_J"]:.4e}'],
        ['# Evaluations', f'{opt_info["n_eval_total"]}'],
    ]
    table = ax3.table(
        cellText=table_data,
        colWidths=[0.55, 0.45],
        loc='center',
        cellLoc='left'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.6)
    # Style header row
    for j in range(2):
        table[0, j].set_facecolor(COLORS['primary'])
        table[0, j].set_text_props(color='white', fontweight='bold')
    ax3.set_title('Recommended Recipe', fontsize=11, fontweight='bold', pad=8)

    fig.suptitle(
        'Physics-Based Inverse Optimization Result\n'
        '[WARNING: Parameters not calibrated — for planning only]',
        fontsize=11, fontweight='bold', color=COLORS['secondary']
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def plot_transport_curves(
    result: SimulationResult,
    mp: ModelParameters,
    save_path: Optional[str] = None
):
    """
    Plot 4: Flux transport model curves — compare ion vs neutral attenuation.
    Shows how T_ion and T_neutral depend on local AR and z.
    """
    z  = result.z_grid
    cd = result.cd_profile
    cd_top = result.cd_top if result.cd_top > 0 else result.conditions.cd_initial

    T_ion     = ion_transmission(z, cd, cd_top, mp)
    T_neutral = neutral_transmission(z, cd, cd_top, mp)
    AR_local  = z / (cd + 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Feature-Scale Transport Model', fontsize=12, fontweight='bold')

    # (a) Transmission vs depth
    ax = axes[0]
    ax.plot(T_ion,     -z, color=COLORS['secondary'], lw=2.5, label='Ion T(z)')
    ax.plot(T_neutral, -z, color=COLORS['primary'],   lw=2.5, label='Neutral T(z)')
    ax.fill_betweenx(-z, 0, T_ion,     alpha=0.1, color=COLORS['secondary'])
    ax.fill_betweenx(-z, 0, T_neutral, alpha=0.1, color=COLORS['primary'])
    _style_ax(ax, 'Transmission vs Depth', 'Transmission factor [-]', 'Depth [nm]')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1.05)

    # (b) Transmission vs local AR
    ax = axes[1]
    ax.plot(AR_local, T_ion,     color=COLORS['secondary'], lw=2.5, label='Ion')
    ax.plot(AR_local, T_neutral, color=COLORS['primary'],   lw=2.5, label='Neutral')
    # Add theoretical Clausing curve
    ar_th = np.linspace(0, max(AR_local.max(), 1.0), 200)
    T_clausing_th = 1.0 / (1.0 + ar_th / 2.0)
    ax.plot(ar_th, T_clausing_th, color='gray', lw=1.5, ls='--', label='Clausing (theory)')
    _style_ax(ax, 'Transmission vs Local AR', 'Local Aspect Ratio [-]', 'Transmission [-]')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def plot_etch_rate_model(
    mp: ModelParameters,
    base_cond: ProcessConditions,
    save_path: Optional[str] = None
):
    """
    Plot 5: Etch rate model — show how each term contributes as a function
    of ion energy and CF4 fraction.
    """
    E_range  = np.linspace(10, 600, 200)
    cf4_range = np.linspace(2, 28, 50)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Surface Reaction Model Characterization', fontsize=12, fontweight='bold')

    # (a) Etch rate terms vs ion energy at fixed flux
    ax = axes[0]
    Gamma_F_ref   = 1e17  # reference F-radical flux [cm-2 s-1]
    Gamma_ion_ref = 1e15  # reference ion flux
    Gamma_CFx_ref = 2e16  # reference CFx flux

    R_chem_arr  = np.array([mp.K_chem * Gamma_F_ref for _ in E_range])
    R_ie_arr    = np.array([
        mp.K_ie * Gamma_F_ref * Gamma_ion_ref * ion_enhanced_factor(E, mp)
        for E in E_range
    ])
    R_sput_arr  = np.array([
        mp.K_sput * Gamma_ion_ref * sputtering_yield(E, mp)
        for E in E_range
    ])
    R_pass_arr  = np.array([mp.K_pass * Gamma_CFx_ref for _ in E_range])
    R_total     = np.maximum(R_chem_arr + R_ie_arr + R_sput_arr - R_pass_arr, 0)

    ax.fill_between(E_range, 0, R_chem_arr,           alpha=0.3, color=COLORS['accent'])
    ax.fill_between(E_range, R_chem_arr,
                    R_chem_arr + R_ie_arr,              alpha=0.3, color=COLORS['primary'])
    ax.fill_between(E_range, R_chem_arr + R_ie_arr,
                    R_chem_arr + R_ie_arr + R_sput_arr, alpha=0.3, color=COLORS['warn'])
    ax.plot(E_range, R_total, color=COLORS['secondary'], lw=2.5, label='Total')
    ax.plot(E_range, R_chem_arr,                          color=COLORS['accent'],  lw=1.5, ls='--', label='Chemical')
    ax.plot(E_range, R_chem_arr + R_ie_arr,               color=COLORS['primary'], lw=1.5, ls='--', label='+ IE')
    ax.axvline(mp.E_sput_threshold, color='gray', ls=':', lw=1.5, label=f'E_sput={mp.E_sput_threshold:.0f}eV')
    _style_ax(ax, 'Etch Rate Terms vs Ion Energy', 'Ion Energy [eV]', 'Etch Rate [nm/s]')
    ax.legend(fontsize=8)

    # (b) AR vs CF4 fraction (parametric sweep)
    ax = axes[1]
    ar_vals   = []
    depth_vals = []
    for cf4 in cf4_range:
        try:
            cond = ProcessConditions(
                cf4_flow=cf4, ar_flow=30.0-cf4,
                v_bias=base_cond.v_bias,
                source_power=base_cond.source_power,
                pressure=base_cond.pressure,
                etch_time=base_cond.etch_time,
                cd_initial=base_cond.cd_initial,
            )
            res = run_forward_simulation(cond, mp, verbose=False)
            ar_vals.append(res.aspect_ratio)
            depth_vals.append(res.total_depth)
        except Exception:
            ar_vals.append(np.nan)
            depth_vals.append(np.nan)

    ax2 = ax.twinx()
    ax.plot(cf4_range, ar_vals,    color=COLORS['primary'],   lw=2.5, label='AR')
    ax2.plot(cf4_range, depth_vals, color=COLORS['secondary'], lw=2.5, ls='--', label='Depth')
    ax.axhline(10, color=COLORS['accent'], ls=':', lw=1.5, label='AR=10 target')
    ax.set_xlabel('CF4 Flow [sccm]', fontsize=9)
    ax.set_ylabel('Aspect Ratio [-]', fontsize=9, color=COLORS['primary'])
    ax2.set_ylabel('Etch Depth [nm]', fontsize=9, color=COLORS['secondary'])
    ax.set_title('AR & Depth vs CF4 Flow', fontsize=11, fontweight='bold', pad=8)
    ax.grid(True, alpha=0.3)
    lines1, lb1 = ax.get_legend_handles_labels()
    lines2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lb1 + lb2, fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10: MAIN — DEMO RUN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """
    Demonstration run of the HARC etch simulator.

    Steps:
      1. Define default process conditions and model parameters.
      2. Run a forward simulation at a nominal recipe.
      3. Run a parametric sweep over CF4 fraction.
      4. Run inverse optimization to find AR ≈ 10.
      5. Plot all results.
      6. Print calibration instructions.
    """
    print("=" * 70)
    print("  HARC ETCH PHYSICS-BASED SIMULATOR  |  CF4/Ar Plasma")
    print("  Project: ML-Based HARC Etch Optimization")
    print("  [WARNING] All parameters are assumptions — not calibrated yet!")
    print("=" * 70)

    # ── 1. Initialize parameters ──────────────────────────────────────────
    mp = ModelParameters()    # default [EST]/[CAL] parameters

    # Nominal process conditions (example starting point)
    cond_nominal = ProcessConditions(
        cf4_flow     = 6.0,   # sccm
        ar_flow      = 24.0,   # sccm  (= 30 - 15)
        v_bias       = -1000.0, # V
        source_power = 250.0,  # W
        pressure     = 10.0,   # mTorr
        substrate_temp = 15.0, # °C
        etch_time    = 300.0,  # s
        cd_initial   = 200.0,  # nm
        mask_thickness = 1350.0,# nm
        target_depth = 2000.0  # nm
    )

    # ── 2. Forward simulation at nominal recipe ───────────────────────────
    print("\n[STEP 1] Forward simulation at nominal recipe...")
    result_nominal = run_forward_simulation(cond_nominal, mp, verbose=True)

    # ── 3. Summary table ──────────────────────────────────────────────────
    print("\n[STEP 2] Parametric sweep: CF4 fraction effect")
    cf4_sweep = np.linspace(5, 25, 9)
    rows = []
    for cf4 in cf4_sweep:
        cond = ProcessConditions(
            cf4_flow=cf4, ar_flow=30.-cf4, v_bias=-800.,
            source_power=500., pressure=20., etch_time=180., cd_initial=100.
        )
        try:
            r = run_forward_simulation(cond, mp)
            rows.append({
                'CF4 [sccm]': cf4,
                'Ar [sccm]':  30.-cf4,
                'Depth [nm]': f'{r.total_depth:.1f}',
                'CD_top [nm]': f'{r.cd_top:.2f}',
                'CD_bot [nm]': f'{r.cd_bot:.2f}',
                'AR':          f'{r.aspect_ratio:.3f}',
                'Taper':       f'{r.taper_index:.4f}',
                'Bowing':      f'{r.bowing_index:.4f}',
            })
        except Exception as e:
            rows.append({'CF4 [sccm]': cf4, 'Error': str(e)})

    df_sweep = pd.DataFrame(rows)
    print(df_sweep.to_string(index=False))

    # ── 4. Inverse optimization ───────────────────────────────────────────
    print("\n[STEP 3] Inverse optimization: finding AR ≈ 10 recipe...")
    best_cond, best_result, opt_info = optimize_process_conditions(
        mp=mp,
        base_cond=cond_nominal,
        target_AR=10.0,
        w_AR=10.0,
        w_taper=5.0,
        w_bowing=3.0,
        verbose=True
    )

    # ── 5. Final summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  OPTIMIZATION COMPLETE — RECOMMENDED RECIPE")
    print("=" * 70)
    print(f"  CF4 flow    : {best_cond.cf4_flow:.3f} sccm")
    print(f"  Ar  flow    : {best_cond.ar_flow:.3f} sccm")
    print(f"  V_bias      : {best_cond.v_bias:.1f} V")
    print(f"  ─────────────────────────────")
    print(f"  Predicted depth  : {best_result.total_depth:.1f} nm")
    print(f"  Predicted CD_top : {best_result.cd_top:.2f} nm")
    print(f"  Predicted CD_bot : {best_result.cd_bot:.2f} nm")
    print(f"  Predicted AR     : {best_result.aspect_ratio:.4f}")
    print(f"  Taper index      : {best_result.taper_index:.5f}")
    print(f"  Bowing index     : {best_result.bowing_index:.5f}")
    print("=" * 70)
    print("\n  [!] These results are PRE-CALIBRATION predictions.")
    print("  [!] Run ~10 experiments and call calibrate_model_parameters()")
    print("      to fit: A_F, A_ion, alpha_E, lambda_ion, lambda_neutral,")
    print("              K_chem, K_ie, K_sput, K_pass")
    print("=" * 70)

    # ── 6. Plots ──────────────────────────────────────────────────────────
    print("\n[STEP 4] Generating plots...")

    plot_simulation_result(result_nominal, save_path='harc_result_nominal.png')
    plot_transport_curves(result_nominal, mp, save_path='harc_transport.png')
    plot_etch_rate_model(mp, cond_nominal, save_path='harc_etch_rate_model.png')
    plot_optimization_result(best_cond, best_result, opt_info,
                             save_path='harc_optimization_result.png')

    # Process window (can be slow; reduce n_cf4/n_vbias if needed)
    plot_process_window(mp, cond_nominal,
                        cf4_range=(4, 26),
                        vbias_range=(-2000, -200),
                        n_cf4=12, n_vbias=12,
                        save_path='harc_process_window.png')

    # ── 7. Calibration reminder ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  CALIBRATION INSTRUCTIONS")
    print("=" * 70)
    print("""
  After collecting ~10 experimental data points, prepare a DataFrame:

      import pandas as pd
      exp_data = pd.DataFrame({
          'cf4_flow':    [...],   # sccm
          'ar_flow':     [...],   # sccm
          'v_bias':      [...],   # V (negative)
          'source_power':[...],   # W
          'pressure':    [...],   # mTorr
          'etch_time':   [...],   # s
          'cd_initial':  [...],   # nm
          'depth_meas':  [...],   # nm  (SEM measurement)
          'cd_top_meas': [...],   # nm
          'cd_bot_meas': [...],   # nm
      })

      mp_calibrated = calibrate_model_parameters(exp_data, mp)

  Then replace mp with mp_calibrated in all further simulations.

  Parameters that MOST AFFECT results (calibrate in this priority):
    1. K_chem, K_ie, K_sput  — control etch depth
    2. A_F, A_ion             — control surface fluxes
    3. lambda_ion, lambda_neutral — control AR / profile shape
    4. alpha_E                — controls ion energy → sputtering
    5. K_pass                 — controls passivation / bowing
    """)

    return result_nominal, best_cond, best_result, mp


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    result_nominal, best_cond, best_result, mp = main()
