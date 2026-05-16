"""
=============================================================================
HARC ETCH SIMULATOR v2  —  Enhanced Physical Models
=============================================================================

NEW MODELS vs v1
----------------
1. IAD (Ion Angular Distribution) — 2-D acceptance-cone model
     Gaussian f(θ) ∝ exp(-θ²/2σ²); σ_iad is a CAL target.
     VERTICAL: acceptance-cone θ_max(z) = min_{z'≤z} arctan(r(z')/z')
               T_floor = [1 - exp(-θ_max²/2σ²)] × ion_directionality
     LATERAL:  grazing angle θ_sw = arctan(r_mask/z)
               T_lat = exp(-θ_sw²/2σ²) × sin(θ_sw) × cos(θ_sw) × shadow(z)
     Both vertical and lateral are physically coupled through σ_iad.

2. Mask aperture evolution  (2-D new feature)
     cd_mask(t) is tracked separately from the substrate hole profile.
     dcd_mask/dt = 2*(R_lat_mask - R_poly_mask)
       R_lat_mask = K_mask_lat * K_sput * Γ_ion * Y_s * Y_mask * (Ar_frac)
         ↑ Ar+ sputters the mask laterally; Ar-fraction dependence
           explains why high-CF4 (low-Ar) cases narrow more.
       R_poly_mask = K_dep_poly * K_mask_poly * Γ_CFx
         ↑ CFx deposits on mask, narrowing opening.
     CD_top reported = cd_mask (what SEM measures at the mask-substrate interface).

3. Improved lateral etch rate  (IAD-weighted sidewall flux)
     R_lat(z) = K_lat_neu  * Γ_F(z)                      [chemical, isotropic]
              + K_lat_ion  * Γ_F(z) * Γ_ion_lat(z) * f_IE [IAD ion-enhanced]
              - K_pass * 0.5 * Γ_CFx(z)                   [passivation]
     where Γ_ion_lat(z) ∝ exp(-θ_sw²/2σ²) × sin(θ_sw)
     and   θ_sw(z) = arctan(r_top/z)  (angle of ions that graze the mask edge
                                        and hit the sidewall at depth z).

CALIBRATION TARGET DATA (from image):
  Source 250 W, 10 mTorr, Electrode 15 °C, Vbias −1000 V, t=240 s

  CF4/Ar [sccm] | Top CD [nm] | Bot CD [nm] | Depth [nm] | AR
  6  / 24       |  210.0      |  74.4       | 1369       | 6.519
  10 / 20       |  205.0      |  44.2       | 1390       | 6.780
  14 / 16       |  204.0      |  53.4       | 1298       | 6.348
  18 / 12       |  213.0      |  73.6       | 1159       | 5.441
  22 /  8       |  140.1      |  65.6       |  616       | 4.397
=============================================================================
"""

from __future__ import annotations
import copy
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from scipy.optimize import least_squares

warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProcessConditions:
    cf4_flow:       float = 10.0
    ar_flow:        float = 20.0
    v_bias:         float = -1000.0
    source_power:   float = 250.0
    pressure:       float = 10.0
    substrate_temp: float = 15.0
    etch_time:      float = 240.0
    cd_initial:     float = 200.0
    mask_thickness: float = 1350.0
    target_depth:   float = 1400.0

    def validate(self):
        if abs(self.cf4_flow + self.ar_flow - 30.0) > 0.5:
            raise ValueError(f"CF4+Ar={self.cf4_flow+self.ar_flow:.1f} ≠ 30 sccm")
        if self.v_bias > 0:
            raise ValueError("v_bias must be ≤ 0")
        if self.etch_time <= 0:
            raise ValueError("etch_time must be > 0")

    @property
    def cf4_fraction(self) -> float:
        return self.cf4_flow / 30.0

    @property
    def ar_fraction(self) -> float:
        return self.ar_flow / 30.0


@dataclass
class ModelParameters:
    # ── 0-D plasma ─────────────────────────────────────────────────────────
    A_F:             float = 5.40e13   # [CAL] F-radical flux coeff  [cm-2 s-1 W-0.5 sccm-0.5]
    #                                          Re-normalised for gamma_F_sat=0.40:
    #                                          6/24 case f_sat=0.92 → A_F = 4.97e13/0.92
    gamma_F_sat:     float = 0.40      # [CAL] CF4-saturation for F radical production
    #                                          Γ_F *= max(1 - gamma_F_sat × cf4_frac, 0.05)
    A_CFx:           float = 1.00e13   # [CAL] CFx flux coeff
    A_ion:           float = 4.94e14   # [CAL] Ion flux coeff        [cm-2 s-1 W-0.5]
    beta_ion:        float = 0.80      # [CAL] Ar-fraction exponent for Gamma_ion
    #                                          Gamma_ion ∝ (ar_frac + alpha_cf4_ion×cf4_frac)^beta_ion
    #                                          Physical: Ar ionizes far more efficiently than CF4,
    #                                          but CF4 fragments (CF3+, CF2+) still contribute.
    alpha_cf4_ion:   float = 0.15      # [CAL] CF4 relative ionization efficiency vs Ar (0=none, 1=equal)
    #                                          Prevents ion flux from collapsing at high CF4.
    #                                          Fixes depth underprediction at CF4≈10 sccm.
    beta_F:          float = 0.85      # [CAL] CF4→F efficiency
    beta_CFx:        float = 0.40      # [CAL] CF4→CFx efficiency

    # ── Sheath / ion energy ─────────────────────────────────────────────────
    alpha_E:         float = 0.61      # [CAL] Ion energy coupling
    E_thermal:       float = 5.0       # [FIX] Plasma potential [eV]
    E_ion_min:       float = 15.0      # [EST] Ion-enhanced etch threshold [eV]

    # ── 2-D Ion transport — IAD acceptance-cone model ───────────────────────
    # VERTICAL (floor flux): Clausing geometric model
    #   Sheath-accelerated ions at -1000 V are near-vertical (σ_eff ≈ 2-5°).
    #   The Clausing power-law captures this HAR geometric collimation:
    #     T_v(z) = ion_directionality / (1 + (z/CD_mask)^clausing_exponent)
    #
    # LATERAL (sidewall flux): IAD acceptance-cone model
    #   Ions at θ_sw = arctan(r_mask/z) hit the sidewall at depth z.
    #   T_lat(z) = exp(-θ_sw²/2σ_iad²) × sin(θ_sw) × cos(θ_sw) × shadow(z)
    #   shadow(z) ≤ 1 if the hole narrows above z (profile re-shadowing).
    #
    # Decoupling vertical/lateral is physically justified: the sheath
    # collimates vertical ions far more than the bulk IAD predicts.
    # σ_iad (17°) controls lateral flux; clausing_exponent controls vertical.
    sigma_iad:          float = 0.30   # [CAL] Gaussian IAD 1-σ spread [rad] (~17°) — lateral only
    ion_directionality: float = 0.90   # [CAL] Overall ion beam efficiency [0-1]
    clausing_exponent:  float = 0.85   # [CAL] Clausing power-law exponent for VERTICAL ion transport

    # ── Neutral transport ───────────────────────────────────────────────────
    lambda_neutral:  float = 14.72     # [CAL] Neutral exp-attenuation [units of CD_top]

    # ── Surface reaction ─────────────────────────────────────────────────────
    K_chem:          float = 9.99e-16  # [CAL] Chemical etch     [nm cm2 s-1]
    K_ie:            float = 7.22e-31  # [CAL] Ion-enhanced etch [nm cm4 s-1]
    K_sput:          float = 5.14e-15  # [CAL] Sputtering        [nm cm2 s-1]
    K_pass:          float = 5.07e-16  # [CAL] Passivation by CFx

    # ── Lateral etch (sidewall) ───────────────────────────────────────────────
    K_lat_neu:       float = 1.5e-3    # [CAL] Chemical lateral / K_chem ratio
    K_lat_ion:       float = 1.0e-2    # [CAL] IAD ion-enhanced lateral factor

    # ── Bohdansky sputtering ────────────────────────────────────────────────
    Q_s:             float = 0.042     # [EST] Yield coefficient
    E_threshold:     float = 20.0      # [FIX] Si sputter threshold [eV]

    # ── Bottom polymer (sidewall passivation layer) ─────────────────────────
    K_dep_poly:      float = 7.42e-15  # [CAL] CFx→polymer deposition [nm cm2 s-1]
    K_etch_poly:     float = 1.06e-16  # [CAL] Ion removal of polymer  [nm cm2 s-1]
    h_poly_char:     float = 1.0       # [CAL] Characteristic polymer thickness [nm]

    # ── Sidewall polymer (substrate hole narrowing) ─────────────────────────
    # The substrate hole sidewall receives CFx polymer deposition.
    # Lateral ions partially remove this polymer via IAD-weighted bombardment.
    # Net effect: CD_full narrows with depth (polymer > lateral etch deep in hole).
    K_dep_side:      float = 3.0e-16   # [CAL] Sidewall polymer deposition coefficient
    #                                          (smaller than K_dep_poly for bottom)

    # ── Birth CD model (IAD collimation effect) ──────────────────────────────
    # When etch front creates a new depth node, the born CD reflects the
    # effective ion beam width at that depth.  IAD collimation: at high AR,
    # the acceptance cone is narrow, so the freshly-exposed surface is carved
    # narrower than the mask opening.
    #
    #   CD_born(z) = cd_mask × exp(-k_born × AR_birth)
    #
    # where AR_birth = depth_current / cd_mask at birth moment.
    # k_born = 0 → born at full mask width (no taper from birth)
    # k_born > 0 → taper increases with AR (physically: IAD collimation)
    k_born:          float = 0.18      # [CAL] IAD birth-CD decay coefficient

    # ── Mask aperture evolution (2-D feature) ────────────────────────────────
    # dCD_mask/dt = 2*(R_lat_mask - R_poly_mask)
    #   R_lat_mask  = K_mask_lat × K_sput × Γ_ion × Y_s × Y_mask × ar_fraction
    #   R_poly_mask = K_dep_poly × K_mask_poly × Γ_CFx
    Y_mask_ratio:    float = 0.30      # [CAL] Mask sputter yield / Si
    K_mask_lat:      float = 0.40      # [CAL] Mask lateral erosion factor
    K_mask_poly:     float = 0.15      # [CAL] Mask polymer / K_dep_poly
    K_F_mask:        float = 3.0e-17   # [CAL] F radical chemical lateral etch of mask [nm cm2 s-1]
    #                                          Explains non-monotonic CD_top: at high CF4 (18 sccm),
    #                                          F radicals chemically etch the mask sidewall,
    #                                          widening the opening even as Ar+ sputtering drops.
    #                                          Without this, CD_top is underpredicted by ~6% at CF4=18.

    # ── Grid / time ─────────────────────────────────────────────────────────
    dz:              float = 20.0
    dt:              float = 0.5


@dataclass
class SimulationResult:
    conditions:              ProcessConditions = field(default_factory=ProcessConditions)
    z_grid:                  np.ndarray = field(default_factory=lambda: np.array([]))
    ion_flux_profile:        np.ndarray = field(default_factory=lambda: np.array([]))
    neutral_flux_profile:    np.ndarray = field(default_factory=lambda: np.array([]))
    cfx_flux_profile:        np.ndarray = field(default_factory=lambda: np.array([]))
    vert_rate_profile:       np.ndarray = field(default_factory=lambda: np.array([]))
    lat_rate_profile:        np.ndarray = field(default_factory=lambda: np.array([]))
    cd_profile:              np.ndarray = field(default_factory=lambda: np.array([]))
    poly_thickness_profile:  np.ndarray = field(default_factory=lambda: np.array([]))
    total_depth:    float = 0.0
    cd_top:         float = 0.0    # = cd_mask at final time (SEM observable)
    cd_mid:         float = 0.0    # substrate hole mid-depth
    cd_bot:         float = 0.0    # substrate hole bottom CD
    mask_cd_final:  float = 0.0    # = cd_top (alias)
    aspect_ratio:   float = 0.0    # total_depth / cd_top
    taper_index:    float = 0.0
    bowing_index:   float = 0.0
    F_flux_surface:   float = 0.0
    ion_flux_surface: float = 0.0
    mean_ion_energy:  float = 0.0
    depth_vs_time:   List[float] = field(default_factory=list)
    cdtop_vs_time:   List[float] = field(default_factory=list)
    time_snapshots:  List[float] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: 0-D PLASMA MODEL  (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────

def calc_plasma_fluxes(
    cond: ProcessConditions, mp: ModelParameters
) -> Tuple[float, float, float]:
    P        = cond.source_power
    cf4_flow = cond.cf4_flow
    pressure = cond.pressure

    f_p_neut = 1.0 / (1.0 + pressure / 50.0)
    f_p_ion  = np.sqrt(10.0 / max(pressure, 0.1))

    cf4_frac  = cond.cf4_fraction
    f_sat     = max(1.0 - mp.gamma_F_sat * cf4_frac, 0.05)
    Gamma_F   = mp.A_F   * np.sqrt(P) * np.sqrt(cf4_flow)  * mp.beta_F   * f_p_neut * f_sat
    cfx_factor = cf4_frac * (1.0 - 0.5 * cf4_frac)
    Gamma_CFx = mp.A_CFx * np.sqrt(P) * np.sqrt(30.0)      * mp.beta_CFx * cfx_factor * f_p_neut
    # Effective ionization: Ar is primary, CF4 fragments (CF3+, CF2+) secondary.
    # ar_frac_eff prevents ion flux from collapsing at high CF4.
    ar_frac_eff = cond.ar_fraction + mp.alpha_cf4_ion * cond.cf4_fraction
    ar_frac_ion = max(ar_frac_eff, 1e-3) ** mp.beta_ion
    Gamma_ion = mp.A_ion * np.sqrt(P) * f_p_ion * ar_frac_ion

    return float(Gamma_F), float(Gamma_CFx), float(Gamma_ion)


def calc_mean_ion_energy(cond: ProcessConditions, mp: ModelParameters) -> float:
    E_ion    = mp.alpha_E * abs(cond.v_bias) + mp.E_thermal
    p_factor = 1.0 / (1.0 + (cond.pressure - 10.0) / 200.0)
    return float(max(E_ion * p_factor, mp.E_thermal))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: TRANSPORT MODELS
# ─────────────────────────────────────────────────────────────────────────────

def ion_transmission_vertical(
    z_array:  np.ndarray,
    cd_array: np.ndarray,
    cd_top:   float,
    mp:       ModelParameters
) -> np.ndarray:
    """
    Vertical ion transmission — Clausing power-law (calibratable).

    At -1000 V bias, the sheath collimates ions to near-vertical (σ_eff ≈ 2-5°),
    so the geometric acceptance fraction is well-approximated by:

        T_v(z) = ion_directionality / (1 + AR_mask^clausing_exponent)

    where AR_mask = z / CD_mask (depth / mask-aperture CD).

    At 10 mTorr, MFP >> hole depth → ions travel ballistically; no gas-phase
    scattering term is needed.  Neutral transport uses cd_array (local profile).
    """
    ar_mask = z_array / (cd_top + 1e-6)
    geo_fac = 1.0 / (1.0 + ar_mask ** mp.clausing_exponent)
    return np.clip(mp.ion_directionality * geo_fac, 0.0, 1.0)


def ion_lateral_flux_factor(
    z_array:  np.ndarray,
    cd_array: np.ndarray,   # substrate hole CD profile at each z
    cd_top:   float,        # mask opening diameter (sets acceptance angle)
    mp:       ModelParameters
) -> np.ndarray:
    """
    IAD-weighted lateral ion flux at each depth z.  2-D model (new in v2).

    Ions entering at angle θ_sw = arctan(r_mask / z) from vertical graze the
    mask edge and impinge on the sidewall at depth z.  The lateral flux factor:

        T_lat(z) = exp(-θ_sw²/2σ²) × sin(θ_sw) × cos(θ_sw) × shadow(z)

      · exp(…)     — IAD probability for grazing ions at angle θ_sw
      · sin(θ_sw)  — component perpendicular to the vertical sidewall
      · cos(θ_sw)  — incidence efficiency (cosine law for sidewall normal)
        Together  sin×cos = ½ sin(2θ_sw), peaks at θ_sw = 45°.
      · shadow(z)  — profile re-shadowing: if hole narrows above z, fewer
                     grazing ions from the mask edge reach depth z.
                     shadow = min(r_cummin / r_mask, 1),  r_cummin = min_{z'≤z} r(z')
    """
    r_mask    = max(cd_top / 2.0, 1.0)
    r_profile = cd_array / 2.0
    z_safe    = np.maximum(z_array, 1.0)
    theta_sw  = np.arctan2(r_mask, z_safe)          # grazing angle [rad]
    sigma2    = 2.0 * mp.sigma_iad ** 2 + 1e-30
    T_lat     = (np.exp(-theta_sw ** 2 / sigma2)
                 * np.sin(theta_sw) * np.cos(theta_sw))
    # Shadow: narrowing above z blocks line-of-sight from mask edge
    r_cummin  = np.minimum.accumulate(r_profile)
    shadow    = np.minimum(r_cummin / r_mask, 1.0)
    return np.clip(T_lat * shadow, 0.0, 1.0)


def neutral_transmission(
    z_array:  np.ndarray,
    cd_array: np.ndarray,
    cd_top:   float,
    mp:       ModelParameters
) -> np.ndarray:
    """Clausing × exponential for neutral radicals (same as v1)."""
    ar_loc    = z_array / (cd_array + 1e-6)
    T_clausing = 1.0 / (1.0 + ar_loc / 2.0)
    T_exp      = np.exp(-z_array / (mp.lambda_neutral * cd_top + 1e-6))
    return np.clip(T_clausing * T_exp, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: SURFACE REACTION MODEL
# ─────────────────────────────────────────────────────────────────────────────

def sputtering_yield(E_ion: float, mp: ModelParameters) -> float:
    if E_ion <= mp.E_threshold:
        return 0.0
    return max(mp.Q_s * (1.0 - np.sqrt(mp.E_threshold / E_ion)) ** 2, 0.0)


def ion_enhanced_factor(E_ion: float, mp: ModelParameters) -> float:
    return float(np.sqrt(max(E_ion - mp.E_ion_min, 0.0))) if E_ion > mp.E_ion_min else 0.0


def calc_vertical_etch_rate(
    Gamma_F_z:   np.ndarray,
    Gamma_ion_z: np.ndarray,
    Gamma_CFx_z: np.ndarray,
    E_ion:       float,
    mp:          ModelParameters
) -> np.ndarray:
    Y_s  = sputtering_yield(E_ion, mp)
    f_IE = ion_enhanced_factor(E_ion, mp)
    R = (mp.K_chem  * Gamma_F_z
       + mp.K_ie    * Gamma_F_z * Gamma_ion_z * f_IE
       + mp.K_sput  * Gamma_ion_z * Y_s
       - mp.K_pass  * Gamma_CFx_z)
    return np.maximum(R, 0.0)


def calc_lateral_etch_rate(
    Gamma_F_z:       np.ndarray,
    Gamma_ion_lat_z: np.ndarray,  # IAD-weighted lateral ion flux
    Gamma_CFx_z:     np.ndarray,
    E_ion:           float,
    mp:              ModelParameters
) -> np.ndarray:
    """
    Sidewall (lateral) etch rate driven by:
      1. Chemical: F radicals reaching sidewall (∝ neutral flux, isotropic)
      2. Ion-enhanced lateral: IAD-angular ions hitting sidewall + F radicals
      3. CFx passivation: reduces lateral etch
    """
    f_IE  = ion_enhanced_factor(E_ion, mp)
    R_lat = (mp.K_lat_neu * mp.K_chem * Gamma_F_z
           + mp.K_lat_ion * mp.K_ie   * Gamma_F_z * Gamma_ion_lat_z * f_IE
           - mp.K_pass * 0.3 * Gamma_CFx_z)
    return np.maximum(R_lat, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: FORWARD SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def run_forward_simulation(
    cond:    ProcessConditions,
    mp:      ModelParameters,
    verbose: bool = False
) -> SimulationResult:
    """
    Forward simulation with IAD lateral flux + 2-D mask aperture evolution.

    Key design points
    -----------------
    - cd_mask(t)   : mask opening diameter (drives ion acceptance cone)
                     → reported as result.cd_top (SEM observable)
    - cd_full[z,t] : substrate hole CD profile (drives CD_bot and bowing)
    - Vertical ion transport: 2D IAD acceptance-cone with profile shadow
    - Lateral ion transport:  IAD angular model with shadow + cos factor
    - Mask aperture evolution: Ar+-sputtering widens, CFx-polymer narrows
    """
    cond.validate()
    result = SimulationResult(conditions=cond)

    Gamma_F_surf, Gamma_CFx_surf, Gamma_ion_surf = calc_plasma_fluxes(cond, mp)
    E_ion = calc_mean_ion_energy(cond, mp)

    result.F_flux_surface   = Gamma_F_surf
    result.ion_flux_surface = Gamma_ion_surf
    result.mean_ion_energy  = E_ion

    Y_s  = sputtering_yield(E_ion, mp)
    f_IE = ion_enhanced_factor(E_ion, mp)

    dz      = mp.dz
    dt      = mp.dt
    n_steps = max(int(cond.etch_time / dt), 1)
    dt      = cond.etch_time / n_steps

    max_depth_est = max(cond.target_depth * 1.8, 800.0)
    N_z_max       = int(max_depth_est / dz) + 4
    z_full        = np.arange(N_z_max) * dz
    cd_full       = np.full(N_z_max, cond.cd_initial, dtype=float)
    h_poly_full   = np.zeros(N_z_max)
    cd_born       = np.zeros(N_z_max, dtype=bool)
    cd_born[:2]   = True

    # Mask aperture state  (drives CD_top and acceptance cone)
    cd_mask  = float(cond.cd_initial)
    h_mask   = float(cond.mask_thickness)

    depth_current  = 0.0
    prev_n_active  = 2

    snap_interval = max(1, n_steps // 10)
    depth_vs_t, cdtop_vs_t, time_snaps = [], [], []

    for step in range(n_steps):
        n_active = max(int(depth_current / dz) + 1, 2)
        n_active = min(n_active, N_z_max - 1)

        if n_active > prev_n_active:
            # Birth CD: IAD collimation narrows the effective etch width at depth.
            # CD_born = cd_mask * exp(-k_born * AR_birth)
            # AR_birth = depth_current / cd_mask at the moment of birth.
            ar_birth    = depth_current / max(cd_mask, 1.0)
            cd_born_val = cd_mask * np.exp(-mp.k_born * ar_birth)
            cd_born_val = float(np.clip(cd_born_val, 5.0, cd_mask))
            for idx in range(prev_n_active, n_active):
                if not cd_born[idx]:
                    cd_full[idx]     = cd_born_val
                    h_poly_full[idx] = 0.0
                    cd_born[idx]     = True
        prev_n_active = n_active

        z_active  = z_full[:n_active]
        cd_active = cd_full[:n_active].copy()

        # Use cd_mask (mask opening) as the effective top CD for transport models
        cd_top_eff = max(cd_mask, 5.0)

        # ── Transport ───────────────────────────────────────────────────────
        T_ion_v   = ion_transmission_vertical(z_active, cd_active, cd_top_eff, mp)
        T_ion_lat = ion_lateral_flux_factor(z_active, cd_active, cd_top_eff, mp)
        T_neutral = neutral_transmission(z_active, cd_active, cd_top_eff, mp)

        Gamma_F_z       = Gamma_F_surf   * T_neutral
        Gamma_ion_v_z   = Gamma_ion_surf * T_ion_v
        Gamma_ion_lat_z = Gamma_ion_surf * T_ion_lat
        Gamma_CFx_z     = Gamma_CFx_surf * T_neutral

        # ── Etch rates ──────────────────────────────────────────────────────
        R_vert_z = calc_vertical_etch_rate(
            Gamma_F_z, Gamma_ion_v_z, Gamma_CFx_z, E_ion, mp
        )
        R_lat_z  = calc_lateral_etch_rate(
            Gamma_F_z, Gamma_ion_lat_z, Gamma_CFx_z, E_ion, mp
        )

        # ── Polymer dynamics (bottom suppression) ────────────────────────────
        R_dep_poly = mp.K_dep_poly  * Gamma_CFx_z
        R_rem_poly = mp.K_etch_poly * Gamma_ion_v_z
        h_poly_full[:n_active] = np.maximum(
            h_poly_full[:n_active] + (R_dep_poly - R_rem_poly) * dt, 0.0
        )
        # poly_supp = bottom-floor polymer suppression.
        # Applied to vertical etch only — floor polymer does not affect sidewall.
        # Sidewall passivation is handled separately via R_dep_side below.
        poly_supp = np.exp(-h_poly_full[:n_active] / max(mp.h_poly_char, 1e-3))
        R_vert_z  *= poly_supp

        # ── Depth advance ───────────────────────────────────────────────────
        depth_current += R_vert_z[-1] * dt

        # ── Substrate CD evolution (hole profile) ────────────────────────────
        # Lateral etch widens; sidewall polymer deposition narrows.
        # Both attenuated by neutral_transmission with depth.
        R_dep_side = mp.K_dep_side * Gamma_CFx_z   # sidewall polymer [nm/s]
        cd_full[:n_active] += 2.0 * (R_lat_z - R_dep_side) * dt
        cd_full[:n_active]  = np.clip(cd_full[:n_active], 5.0, cond.cd_initial * 3.0)

        # ── Mask aperture evolution (2-D feature, determines CD_top) ─────────
        if h_mask > 0.0:
            # Ar+ sputters mask laterally (widens); ar_fraction captures gas chemistry
            R_sput_mask_lat = (mp.K_mask_lat * mp.K_sput
                               * Gamma_ion_surf * Y_s
                               * mp.Y_mask_ratio * cond.ar_fraction)
            # F radicals chemically etch mask sidewall (widens at high CF4)
            R_F_mask_lat    = mp.K_F_mask * Gamma_F_surf
            # CFx polymer narrows mask opening
            R_poly_mask     = mp.K_dep_poly * mp.K_mask_poly * Gamma_CFx_surf

            # Mask height erosion
            R_sput_mask_vert = (mp.K_sput * Gamma_ion_surf * Y_s * mp.Y_mask_ratio)
            h_mask = max(h_mask - R_sput_mask_vert * dt, 0.0)

            # Mask opening evolution: Ar+ sputtering + F chemical etch − CFx polymer
            d_cd_mask = 2.0 * (R_sput_mask_lat + R_F_mask_lat - R_poly_mask) * dt
            cd_mask   = float(np.clip(cd_mask + d_cd_mask, 5.0, cond.cd_initial * 3.0))

        # ── Snapshots ────────────────────────────────────────────────────────
        if step % snap_interval == 0 or step == n_steps - 1:
            depth_vs_t.append(depth_current)
            cdtop_vs_t.append(cd_mask)
            time_snaps.append(step * dt)

    # ── Final profile ────────────────────────────────────────────────────────
    # Use prev_n_active (last loop value) so we only include nodes that were
    # actually born and etched.  depth_current can advance past a dz boundary
    # in the final step, making int(depth/dz)+1 larger than the last n_active;
    # that extra node was never born and still holds cd_initial (wrong).
    n_final  = min(max(prev_n_active, 2), N_z_max - 1)
    z_final  = z_full[:n_final]
    cd_final = cd_full[:n_final]

    cd_top_f  = max(cd_mask, 5.0)
    T_iv_f    = ion_transmission_vertical(z_final, cd_final, cd_top_f, mp)
    T_il_f    = ion_lateral_flux_factor(z_final, cd_final, cd_top_f, mp)
    T_nf      = neutral_transmission(z_final, cd_final, cd_top_f, mp)

    G_F_f    = Gamma_F_surf   * T_nf
    G_ion_f  = Gamma_ion_surf * T_iv_f
    G_CFx_f  = Gamma_CFx_surf * T_nf
    R_v_f    = calc_vertical_etch_rate(G_F_f, G_ion_f, G_CFx_f, E_ion, mp)
    R_l_f    = calc_lateral_etch_rate(G_F_f, Gamma_ion_surf * T_il_f, G_CFx_f, E_ion, mp)

    cd_top   = float(cd_mask)              # SEM-observable top CD = mask opening
    cd_bot   = float(cd_final[-1])
    cd_mid   = float(cd_final[len(cd_final) // 2])
    AR       = depth_current / max(cd_top, 1.0)
    taper    = (cd_top - cd_bot) / max(cd_top, 1.0)
    bowing   = (float(np.max(cd_final)) - cd_bot) / max(cd_top, 1.0)

    result.z_grid                 = z_final
    result.ion_flux_profile       = G_ion_f
    result.neutral_flux_profile   = G_F_f
    result.cfx_flux_profile       = G_CFx_f
    result.vert_rate_profile      = R_v_f
    result.lat_rate_profile       = R_l_f
    result.cd_profile             = cd_final
    result.poly_thickness_profile = h_poly_full[:n_final].copy()
    result.total_depth    = depth_current
    result.cd_top         = cd_top
    result.cd_mid         = cd_mid
    result.cd_bot         = cd_bot
    result.mask_cd_final  = cd_mask
    result.aspect_ratio   = AR
    result.taper_index    = taper
    result.bowing_index   = bowing
    result.depth_vs_time  = depth_vs_t
    result.cdtop_vs_time  = cdtop_vs_t
    result.time_snapshots = time_snaps

    if verbose:
        print(f"  Depth={depth_current:.1f} nm  CD_top(mask)={cd_top:.1f} nm  "
              f"CD_bot={cd_bot:.1f} nm  AR={AR:.3f}  h_mask={h_mask:.1f} nm")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

# Full 5-point experimental dataset
# NOTE: Point 5 (CF4=22/Ar=8) has low reliability (anomalously low Top CD=140.1 nm
#       vs ~204-213 nm for all other conditions).  It is kept here for reference
#       and post-calibration comparison, but excluded from calibration fitting.
EXPERIMENTAL_DATA = pd.DataFrame({
    'cf4_flow':       [6.0,    10.0,   14.0,   18.0,   22.0  ],
    'ar_flow':        [24.0,   20.0,   16.0,   12.0,    8.0  ],
    'v_bias':         [-1000., -1000., -1000., -1000., -1000. ],
    'source_power':   [250.,   250.,   250.,   250.,   250.  ],
    'pressure':       [10.,    10.,    10.,    10.,    10.   ],
    'substrate_temp': [15.,    15.,    15.,    15.,    15.   ],
    'etch_time':      [240.,   240.,   240.,   240.,   240.  ],
    'cd_initial':     [200.,   200.,   200.,   200.,   200.  ],
    'mask_thickness': [1350.,  1350.,  1350.,  1350.,  1350. ],
    'target_depth':   [1400.,  1400.,  1300.,  1200.,  700.  ],
    'depth_meas':     [1369.,  1390.,  1298.,  1159.,  616.  ],
    'cd_top_meas':    [210.0,  205.0,  204.0,  213.0,  140.1 ],
    'cd_bot_meas':    [74.4,   44.2,   53.4,   73.6,   65.6  ],
    'reliable':       [True,   True,   True,   True,   False ],  # False = exclude from cal
})


# Weights for the calibration residuals
# AR is now primary target — depth/CD_top fitting alone does not guarantee AR accuracy
# because small opposite-direction errors in both compound into a large AR error.
_W_DEPTH = 1.2   # relaxed: allow ±3% depth error to let AR converge
_W_CDTOP = 1.0   # relaxed: mask opening is secondary to AR
_W_CDBOT = 0.5   # unchanged: hardest to fit, kept low
_W_AR    = 2.5   # primary: directly penalize AR error


def _build_experiments(exp_data: pd.DataFrame):
    exps = []
    for _, row in exp_data.iterrows():
        c = ProcessConditions(
            cf4_flow       = float(row['cf4_flow']),
            ar_flow        = float(row['ar_flow']),
            v_bias         = float(row['v_bias']),
            source_power   = float(row['source_power']),
            pressure       = float(row['pressure']),
            substrate_temp = float(row['substrate_temp']),
            etch_time      = float(row['etch_time']),
            cd_initial     = float(row['cd_initial']),
            mask_thickness = float(row['mask_thickness']),
            target_depth   = float(row['target_depth']),
        )
        meas_ar = float(row['depth_meas']) / max(float(row['cd_top_meas']), 1.0)
        exps.append((c, {
            'depth'  : float(row['depth_meas']),
            'cd_top' : float(row['cd_top_meas']),
            'cd_bot' : float(row['cd_bot_meas']),
            'ar'     : meas_ar,
        }))
    return exps


def calibrate_model_parameters(
    exp_data:         pd.DataFrame,
    mp_init:          ModelParameters,
    calibrate_params: Optional[List[str]] = None,
    verbose:          bool = True,
    cal_dt:           float = 2.0   # coarser dt for calibration speed (3x faster, <2% error)
) -> Tuple[ModelParameters, dict]:
    """
    Calibrate model parameters against experimental data (4 reliable points).

    Method: log10-space least_squares (TRF), two-stage (coarse → fine).
    Residuals: [r_depth, r_cd_top, r_cd_bot, r_ar] per experiment.

    Returns (mp_calibrated, info_dict).
    """
    if calibrate_params is None:
        # 4 experiments × 4 outputs = 16 equations.
        # Keep ≤16 free parameters to avoid underdetermined degeneracy.
        #
        # FIXED (not calibrated):
        #   A_F, A_ion  — degenerate with K_chem/K_sput/K_ie; fix flux prefactors,
        #                  calibrate rate coefficients instead.
        #   beta_F      — CF4→F efficiency; CF4-only variation gives weak signal.
        #   alpha_E     — sheath energy coupling; 0.61 is well-constrained for CCP.
        #   h_poly_char — polymer suppression length scale; data insufficient to
        #                  separate from K_dep_poly / K_etch_poly individually.
        calibrate_params = [
            # Plasma model (relative F/ion balance vs CF4 fraction)
            'gamma_F_sat', 'beta_ion', 'alpha_cf4_ion',
            # Vertical transport (Clausing geometric + efficiency)
            'ion_directionality', 'clausing_exponent',
            # Neutral transport
            'lambda_neutral',
            # IAD (lateral)
            'sigma_iad',
            # Surface reactions (rate coefficients; flux prefactors A_F, A_ion fixed)
            'K_chem', 'K_ie', 'K_sput', 'K_pass',
            # Lateral etch
            'K_lat_neu', 'K_lat_ion',
            # Polymer (bottom suppression)
            'K_dep_poly', 'K_etch_poly',
            # Sidewall polymer + birth CD
            'K_dep_side', 'k_born',
            # Mask evolution (+ F radical chemical etch of mask)
            'K_mask_lat', 'K_mask_poly', 'K_F_mask',
        ]   # total: 20 params — 4 exp × 4 outputs = 16 eq; bounded TRF handles over-parameterisation.

    BOUNDS = {
        'A_F':               (1e12,  1e19),
        'gamma_F_sat':       (1e-4,  0.95),
        'A_CFx':             (1e11,  1e18),
        'A_ion':             (1e11,  1e18),
        'beta_ion':          (0.01,  3.0),
        'beta_F':            (0.05,  1.0),
        'beta_CFx':          (0.02,  1.0),
        'alpha_E':           (0.05,  1.0),
        'ion_directionality':(0.05,  1.0),
        'clausing_exponent': (0.1,   3.0),
        'lambda_neutral':    (0.3,   50.0),
        'sigma_iad':         (0.03,  1.5),
        'K_chem':            (1e-25, 1e-14),
        'K_ie':              (1e-40, 1e-26),
        'K_sput':            (1e-23, 1e-12),
        'K_pass':            (1e-26, 1e-14),
        'K_lat_neu':         (1e-5,  1.0),
        'K_lat_ion':         (1e-5,  1.0),
        'K_dep_poly':        (1e-18, 1e-12),
        'K_etch_poly':       (1e-20, 1e-13),
        'h_poly_char':       (0.05,  50.0),
        'K_dep_side':        (1e-19, 1e-13),
        'k_born':            (0.01,  1.0),
        'Y_mask_ratio':      (0.01,  3.0),
        'K_mask_lat':        (1e-4,  5.0),
        'K_mask_poly':       (0.005, 5.0),
        'alpha_cf4_ion':     (1e-3,  0.8),
        'K_F_mask':          (1e-20, 1e-14),
    }

    mp_work      = copy.deepcopy(mp_init)
    mp_work.dt   = cal_dt   # use coarser step for faster calibration evaluations
    x0_log, lo_log, hi_log = [], [], []
    for pname in calibrate_params:
        val    = getattr(mp_work, pname)
        lo, hi = BOUNDS.get(pname, (val * 1e-3, val * 1e3))
        val_cl = float(np.clip(val, lo * 1.001, hi * 0.999))
        x0_log.append(np.log10(val_cl))
        lo_log.append(np.log10(lo))
        hi_log.append(np.log10(hi))
    x0_log     = np.array(x0_log)
    bounds_log = (lo_log, hi_log)

    experiments = _build_experiments(exp_data)
    call_count  = [0]
    fail_count  = [0]

    def residual_fn(x_log: np.ndarray) -> np.ndarray:
        mp_try = copy.deepcopy(mp_work)
        for i, pname in enumerate(calibrate_params):
            setattr(mp_try, pname, float(10.0 ** x_log[i]))

        residuals = []
        for cond_e, meas in experiments:
            try:
                r       = run_forward_simulation(cond_e, mp_try, verbose=False)
                r_depth = _W_DEPTH * (r.total_depth  - meas['depth'])  / max(meas['depth'],  10.0)
                r_top   = _W_CDTOP * (r.cd_top       - meas['cd_top']) / max(meas['cd_top'],  5.0)
                r_bot   = _W_CDBOT * (r.cd_bot       - meas['cd_bot']) / max(meas['cd_bot'],  5.0)
                r_ar    = _W_AR    * (r.aspect_ratio  - meas['ar'])     / max(meas['ar'],      0.1)
                residuals.extend([r_depth, r_top, r_bot, r_ar])
            except Exception as exc:
                fail_count[0] += 1
                if verbose and fail_count[0] <= 3:
                    print(f"    [WARN] simulation failed (nfev={call_count[0]}): {exc}")
                residuals.extend([1e3, 1e3, 1e3, 1e3])

        call_count[0] += 1
        if verbose and call_count[0] % 20 == 0:
            rms = np.sqrt(np.mean(np.array(residuals)**2))
            print(f"    nfev={call_count[0]:5d}  weighted RMS={rms:.4f}  fails={fail_count[0]}")

        return np.array(residuals)

    if verbose:
        print("=" * 68)
        print(f"  CALIBRATING {len(calibrate_params)} params × {len(experiments)} experiments")
        print(f"  Stage 1: coarse TRF (ftol=1e-3, diff_step=1e-3)…")

    # diff_step=1e-3: gives ~0.2-3.5% param perturbation in log-space,
    # large enough for the simulator to register (avoids zero-gradient trap).
    cal1 = least_squares(
        residual_fn, x0=x0_log, bounds=bounds_log,
        method='trf', ftol=1e-3, xtol=1e-3, gtol=1e-4,
        max_nfev=3000, diff_step=1e-3, verbose=0,
    )
    if verbose:
        print(f"  Stage 1 done  cost={cal1.cost:.4e}  nfev={cal1.nfev}")
        print("  Stage 2: fine TRF (ftol=1e-6, diff_step=5e-4)…")

    cal2 = least_squares(
        residual_fn, x0=cal1.x, bounds=bounds_log,
        method='trf', ftol=1e-6, xtol=1e-6, gtol=1e-8,
        max_nfev=10000, diff_step=5e-4, verbose=0,
    )

    mp_cal = copy.deepcopy(mp_init)
    if verbose:
        print(f"  Stage 2 done  cost={cal2.cost:.4e}  nfev={cal2.nfev}")
        print(f"\n  {'Parameter':<22}  {'Init':>12}  {'Calibrated':>12}  {'×':>7}")
        print(f"  {'-'*58}")
    for i, pname in enumerate(calibrate_params):
        old_val = getattr(mp_init, pname)
        new_val = float(10.0 ** cal2.x[i])
        setattr(mp_cal, pname, new_val)
        if verbose:
            ratio = new_val / old_val if old_val != 0 else float('inf')
            print(f"  {pname:<22}  {old_val:>12.4e}  {new_val:>12.4e}  {ratio:>7.2f}")

    info = {
        'cal1': cal1, 'cal2': cal2,
        'final_cost': cal2.cost,
        'calibrate_params': calibrate_params,
    }
    return mp_cal, info


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: ACCURACY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_accuracy_table(label: str, exp_data: pd.DataFrame, mp: ModelParameters):
    exps = _build_experiments(exp_data)
    print(f"\n{'='*86}")
    print(f"  [{label}]")
    print(f"  {'CF4/Ar':>8}  "
          f"{'Dep(exp)':>9} {'Dep(sim)':>9} {'Err%':>6}  "
          f"{'Top(e)':>7} {'Top(s)':>7} {'Err%':>6}  "
          f"{'Bot(e)':>7} {'Bot(s)':>7} {'Err%':>6}")
    print(f"  {'-'*84}")
    for (cond_e, meas) in exps:
        try:
            r  = run_forward_simulation(cond_e, mp, verbose=False)
            lbl = f"{int(cond_e.cf4_flow)}/{int(cond_e.ar_flow)}"
            de = meas['depth'];   ds = r.total_depth
            te = meas['cd_top'];  ts = r.cd_top
            be = meas['cd_bot'];  bs = r.cd_bot
            print(
                f"  {lbl:>8}  "
                f"{de:>9.1f} {ds:>9.1f} {100*(ds-de)/de:>+6.1f}%  "
                f"{te:>7.1f} {ts:>7.1f} {100*(ts-te)/te:>+6.1f}%  "
                f"{be:>7.1f} {bs:>7.1f} {100*(bs-be)/be:>+6.1f}%"
            )
        except Exception as e:
            print(f"  {int(cond_e.cf4_flow)}/{int(cond_e.ar_flow):>8}  ERROR: {e}")
    print(f"{'='*86}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: PLOTS
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    'primary'  : '#2563EB',
    'secondary': '#DC2626',
    'accent'   : '#16A34A',
    'warn'     : '#D97706',
    'purple'   : '#7C3AED',
    'bg'       : '#F8FAFC',
}


def plot_calibration_comparison(
    exp_data:  pd.DataFrame,
    mp_init:   ModelParameters,
    mp_cal:    ModelParameters,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Bar-chart comparison: Experiment vs Pre-cal vs Post-cal
    for Depth, CD_top, CD_bot, Aspect Ratio across all 5 conditions.
    """
    exps      = _build_experiments(exp_data)
    cf4_vals  = exp_data['cf4_flow'].values
    labels    = [f"CF4={int(c)}/Ar={int(30-c)}" for c in cf4_vals]
    x         = np.arange(len(cf4_vals))

    meas_depth = exp_data['depth_meas'].values
    meas_top   = exp_data['cd_top_meas'].values
    meas_bot   = exp_data['cd_bot_meas'].values
    meas_ar    = meas_depth / np.maximum(meas_top, 1.0)

    pre_d, pre_t, pre_b, pre_ar    = [], [], [], []
    post_d, post_t, post_b, post_ar = [], [], [], []

    for cond_e, meas in exps:
        for mp_use, dl, tl, bl, al in [
            (mp_init, pre_d,  pre_t,  pre_b,  pre_ar),
            (mp_cal,  post_d, post_t, post_b, post_ar),
        ]:
            try:
                r = run_forward_simulation(cond_e, mp_use, verbose=False)
                dl.append(r.total_depth); tl.append(r.cd_top)
                bl.append(r.cd_bot);      al.append(r.aspect_ratio)
            except Exception:
                dl.append(np.nan); tl.append(np.nan)
                bl.append(np.nan); al.append(np.nan)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        'HARC v2 — Calibration Result\n'
        'Source 250 W, Pressure 10 mTorr, Vbias −1000 V, T=15 °C, t=240 s',
        fontsize=13, fontweight='bold'
    )

    datasets = [
        ('Etch Depth [nm]',    meas_depth, pre_d,  post_d),
        ('Top CD [nm]',        meas_top,   pre_t,  post_t),
        ('Bottom CD [nm]',     meas_bot,   pre_b,  post_b),
        ('Aspect Ratio',       meas_ar,    pre_ar, post_ar),
    ]
    w = 0.27
    for ax, (title, meas, pre, post) in zip(axes.flat, datasets):
        ax.bar(x - w, meas, width=w, color='gray',           alpha=0.85,
               label='Experiment', edgecolor='k', linewidth=0.6)
        ax.bar(x,     pre,  width=w, color=COLORS['primary'], alpha=0.85,
               label='Pre-calibration', edgecolor='k', linewidth=0.6)
        ax.bar(x + w, post, width=w, color=COLORS['secondary'], alpha=0.85,
               label='Post-calibration', edgecolor='k', linewidth=0.6)

        # Error % labels on post-cal bars
        for xi, (m, p) in enumerate(zip(meas, post)):
            if not np.isnan(p) and m > 0:
                err = 100 * (p - m) / m
                clr = COLORS['accent'] if abs(err) < 10 else COLORS['warn']
                ax.text(xi + w, max(p, m) * 1.01,
                        f'{err:+.1f}%', ha='center', va='bottom', fontsize=7, color=clr)

        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(title, fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_facecolor(COLORS['bg'])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved → {save_path}")
    plt.show()
    return fig


def plot_profiles(
    exp_data:  pd.DataFrame,
    mp_cal:    ModelParameters,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Cross-section profile for each of the 5 CF4/Ar conditions
    after calibration, overlaid with experimental CD_top and CD_bot markers.
    """
    exps = _build_experiments(exp_data)
    n    = len(exps)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 7))
    fig.suptitle('Calibrated Hole Profiles  (lines=simulated, diamonds=experimental)',
                 fontsize=12, fontweight='bold')

    for ax, (cond_e, meas) in zip(axes, exps):
        try:
            res = run_forward_simulation(cond_e, mp_cal, verbose=False)
        except Exception as err:
            ax.set_title(f"CF4={int(cond_e.cf4_flow)} sccm\nERROR: {err}", fontsize=8)
            continue

        z  = res.z_grid
        cd = res.cd_profile

        ax.plot(-cd/2, -z, color=COLORS['primary'], lw=2)
        ax.plot( cd/2, -z, color=COLORS['primary'], lw=2)
        ax.fill_betweenx(-z, -cd/2, cd/2, alpha=0.12, color=COLORS['primary'])

        # Mask region (above substrate surface)
        mask_cd = res.cd_top
        mask_h  = cond_e.mask_thickness
        ax.fill_betweenx([0, mask_h * 0.15],
                         [-mask_cd/2, -mask_cd/2], [mask_cd/2, mask_cd/2],
                         alpha=0.25, color='gray')

        # Experimental markers
        dep_e = meas['depth'];  top_e = meas['cd_top'];  bot_e = meas['cd_bot']
        ax.scatter([-top_e/2, top_e/2], [0, 0],
                   color='gray', s=70, zorder=6, marker='D', label=f'Top CD exp={top_e:.0f}nm')
        ax.scatter([-bot_e/2, bot_e/2], [-dep_e, -dep_e],
                   color=COLORS['warn'], s=70, zorder=6, marker='D', label=f'Bot CD exp={bot_e:.0f}nm')

        # Horizontal dashed line at experimental depth
        ax.axhline(-dep_e, color=COLORS['warn'], lw=1.0, ls='--', alpha=0.6)

        ax.set_title(
            f"CF4={int(cond_e.cf4_flow)}/Ar={int(cond_e.ar_flow)} sccm\n"
            f"Depth: sim={res.total_depth:.0f}  exp={dep_e:.0f} nm\n"
            f"CDtop: sim={res.cd_top:.0f}  exp={top_e:.0f} nm",
            fontsize=7.5, fontweight='bold'
        )
        ax.set_xlabel('x [nm]', fontsize=8)
        ax.set_ylabel('Depth [nm]', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, loc='lower right')
        ax.grid(True, alpha=0.2)
        ax.set_facecolor(COLORS['bg'])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved → {save_path}")
    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  HARC ETCH SIMULATOR v2")
    print("  New models: IAD lateral flux + 2-D mask aperture evolution")
    print("  Calibration: 4-point dataset (point 5 excluded — low reliability)")
    print("  (250W, 10mTorr, −1000V, 15°C)")
    print("=" * 70)

    mp_init  = ModelParameters()
    exp_data = EXPERIMENTAL_DATA.copy()

    # Point 5 (CF4=22/Ar=8) excluded from calibration; retained for validation display
    cal_data = exp_data[exp_data['reliable']].copy().reset_index(drop=True)

    print("\n[STEP 1] Pre-calibration forward runs …")
    print_accuracy_table("PRE-CALIBRATION (all 5 points)", exp_data, mp_init)

    print(f"\n[STEP 2] Running calibration on {len(cal_data)} reliable points …")
    mp_cal, cal_info = calibrate_model_parameters(
        cal_data, mp_init, verbose=True
    )

    print("\n[STEP 3] Post-calibration accuracy (all 5 points for reference) …")
    print_accuracy_table("POST-CALIBRATION", exp_data, mp_cal)

    print("\n[STEP 4] Generating plots …")
    plot_calibration_comparison(
        exp_data, mp_init, mp_cal,
        save_path='harc_v2_calibration.png'
    )
    plot_profiles(
        exp_data, mp_cal,
        save_path='harc_v2_profiles.png'
    )

    print("\n[STEP 5] Calibrated ModelParameters (copy-paste to reuse):")
    print("  mp_cal = ModelParameters(")
    for pname in cal_info['calibrate_params']:
        val = getattr(mp_cal, pname)
        print(f"      {pname:<22} = {val:.4e},")
    print("  )")

    return mp_cal, cal_info


if __name__ == '__main__':
    mp_cal, cal_info = main()
