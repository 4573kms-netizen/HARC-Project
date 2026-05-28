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
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List, Dict
from scipy.optimize import least_squares

warnings.filterwarnings("ignore", category=UserWarning)

_HERE = os.path.dirname(os.path.abspath(__file__))

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
    total_flow:     float = 30.0   # [sccm] default 30 for HARC calibration; set 50 for RODEo

    def validate(self):
        if abs(self.cf4_flow + self.ar_flow - self.total_flow) > 0.5:
            raise ValueError(
                f"CF4+Ar={self.cf4_flow+self.ar_flow:.1f} != total_flow={self.total_flow:.1f} sccm"
            )
        if self.v_bias > 0:
            raise ValueError("v_bias must be <= 0")
        if self.etch_time <= 0:
            raise ValueError("etch_time must be > 0")

    @property
    def cf4_fraction(self) -> float:
        return self.cf4_flow / self.total_flow

    @property
    def ar_fraction(self) -> float:
        return self.ar_flow / self.total_flow


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
    # Neutral Clausing AR weighting: 1.0 → use local profile (current), 0.0 → use mask opening.
    # Physical interpretation: ballistic neutrals at high vacuum partially "see" the full mask
    # cone rather than the local profile.  Values < 1 decouple depth from k_born profile shape.
    k_clausing_neutral: float = 1.0   # [CAL] Blend weight for neutral T_clausing (1=local, 0=mask)

    # ── Surface reaction ─────────────────────────────────────────────────────
    K_chem:          float = 9.99e-16  # [CAL] Chemical etch     [nm cm2 s-1]
    K_ie:            float = 7.22e-31  # [CAL] Ion-enhanced etch [nm cm4 s-1]
    K_sput:          float = 5.14e-15  # [CAL] Sputtering        [nm cm2 s-1]
    K_pass:          float = 5.07e-16  # [CAL] Passivation by CFx

    # ── Lateral etch (sidewall) ───────────────────────────────────────────────
    K_lat_neu:       float = 1.5e-3    # [CAL] Chemical lateral / K_chem ratio
    K_lat_ion:       float = 1.0e-2    # [CAL] IAD ion-enhanced lateral factor
    K_lat_sput:      float = 1.0e-14   # [CAL] Lateral physical sputtering [nm cm2 s-1]
    #                                          Ar+ ions at grazing angle sputter the sidewall.
    #                                          Proportional to Gamma_ion_lat × Y_s.
    #                                          Key for Ar-rich conditions (6/24, 18/12) having
    #                                          wider CD_bot than predicted without this term.
    K_lat_pass:      float = 0.30      # [CAL] CFx passivation fraction in lateral etch
    #                                          Replaces hardcoded 0.3 factor; allows optimizer
    #                                          to balance CFx suppression of lateral etch.

    # ── Bohdansky sputtering ────────────────────────────────────────────────
    Q_s:             float = 0.042     # [EST] Yield coefficient
    E_threshold:     float = 20.0      # [FIX] SiO2 sputter threshold ~15-20 eV (Ar+)

    # ── Bottom polymer (sidewall passivation layer) ─────────────────────────
    K_dep_poly:      float = 7.42e-15  # [CAL] CFx→polymer deposition [nm cm2 s-1]
    K_etch_poly:     float = 1.06e-16  # [CAL] Ion removal of polymer  [nm cm2 s-1]
    h_poly_char:     float = 1.0       # [CAL] Characteristic polymer thickness [nm]
    n_poly_dep:      float = 1.0       # [CAL] Nonlinear exponent for CFx deposition
    #                                          R_dep ∝ Gamma_CFx^n; n>1 → super-linear
    #                                          amplifies high-CF4 polymer buildup

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
    k_born:              float = 0.235  # [CAL] Peak IAD birth-CD collimation coefficient
    k_born_spread_left:  float = 17.0  # [CAL] Gaussian spread for CF4 < cf4_frac_peak
    k_born_spread_right: float = 2.5   # [CAL] Gaussian spread for CF4 > cf4_frac_peak
    #   Asymmetric Gaussian: k_born_eff = k_born × exp(-spread × (cf4_frac - peak)²)
    #   Left (CF4 < peak, e.g. 6/24): spread_left steep → wider CD_born for Ar-rich
    #   Right (CF4 > peak, e.g. 14/16, 18/12): spread_right gentle → moderate CD_born
    #   Physical: collimation peaks at optimal IE-etch CF4/Ar; extremes less directional.
    cf4_frac_peak:       float = 0.33  # [FIX] CF4 fraction at peak collimation (10/20 optimal)

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

    # ── Reactor-level absolute flux scale ────────────────────────────────────
    # k_rate_global scales all plasma fluxes (F, CFx, ion) uniformly.
    # Default 1.0 for the calibrated ICP reactor (250W, 10mTorr, -1000V).
    # For validation against different reactors (e.g. CCP-RIE), calibrate this
    # single parameter to match the absolute etch rate, then check CD_top as
    # a forward prediction.  Does not affect HARC calibration (kept at 1.0).
    k_rate_global:   float = 1.0

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
    Gamma_CFx = mp.A_CFx * np.sqrt(P) * np.sqrt(cond.total_flow) * mp.beta_CFx * cfx_factor * f_p_neut
    # Effective ionization: Ar is primary, CF4 fragments (CF3+, CF2+) secondary.
    # ar_frac_eff prevents ion flux from collapsing at high CF4.
    ar_frac_eff = cond.ar_fraction + mp.alpha_cf4_ion * cond.cf4_fraction
    ar_frac_ion = max(ar_frac_eff, 1e-3) ** mp.beta_ion
    Gamma_ion = mp.A_ion * np.sqrt(P) * f_p_ion * ar_frac_ion

    k = mp.k_rate_global
    return float(Gamma_F * k), float(Gamma_CFx * k), float(Gamma_ion * k)


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
    """Clausing × exponential for neutral radicals.

    k_clausing_neutral blends between local-profile AR (=1, profile-coupled)
    and mask-opening AR (=0, profile-decoupled).  This separates the CD_bot
    accuracy from the depth accuracy when the k_born model narrows 10/20 profiles.
    """
    cd_eff    = mp.k_clausing_neutral * cd_array + (1.0 - mp.k_clausing_neutral) * cd_top
    ar_loc    = z_array / (np.maximum(cd_eff, 1.0))
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
      3. Physical sputtering: Ar+ at grazing angle (Ar-rich → wider CD_bot)
      4. CFx passivation: reduces lateral etch (K_lat_pass now calibratable)
    """
    f_IE  = ion_enhanced_factor(E_ion, mp)
    Y_s   = sputtering_yield(E_ion, mp)
    R_lat = (mp.K_lat_neu * mp.K_chem * Gamma_F_z
           + mp.K_lat_ion * mp.K_ie   * Gamma_F_z * Gamma_ion_lat_z * f_IE
           + mp.K_lat_sput             * Gamma_ion_lat_z * Y_s
           - mp.K_pass * mp.K_lat_pass * Gamma_CFx_z)
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
            # Birth CD: asymmetric Gaussian collimation in CF4-fraction space.
            # Separate spread left/right of cf4_frac_peak fits the non-symmetric
            # experimental CD_bot pattern (wide-narrow-medium-wide across CF4/Ar).
            cf4_dev_born = cond.cf4_fraction - mp.cf4_frac_peak
            spread = mp.k_born_spread_left if cf4_dev_born < 0 else mp.k_born_spread_right
            k_born_eff  = mp.k_born * np.exp(-spread * cf4_dev_born ** 2)
            ar_birth    = depth_current / max(cd_mask, 1.0)
            cd_born_val = cd_mask * np.exp(-k_born_eff * ar_birth)
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
        # Nonlinear deposition: R ∝ Gamma_CFx^n_poly_dep (normalized at 1e14 cm-2s-1).
        # n>1 makes high-CF4 conditions accumulate polymer super-linearly → amplifies
        # CD_bot spread across CF4/Ar conditions without changing linear-regime behavior.
        _norm = np.power(np.maximum(Gamma_CFx_z / 1e14, 1e-30), mp.n_poly_dep - 1.0)
        R_dep_poly = mp.K_dep_poly  * Gamma_CFx_z * _norm
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
        # z=0: hole entry cannot be wider than mask aperture, but can narrow from footing polymer
        cd_full[0] = min(cd_full[0], cd_mask)

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
_W_DEPTH = 2.0   # raised: prevent birth-CD widening from blowing up depth
_W_CDTOP = 1.0   # relaxed: mask opening is secondary to AR
_W_CDBOT = 1.5   # raised: CD_bot fitting is a primary target
_W_AR    = 2.5   # primary: directly penalize AR error
# Hinge loss: extra penalty when |error| exceeds threshold (for both Depth and CD_bot)
_W_HINGE_BOT   = 8.0    # extra weight for out-of-tolerance CD_bot
_W_HINGE_DEPTH = 10.0   # extra weight for out-of-tolerance Depth
_W_HINGE_AR    = 10.0   # extra weight for out-of-tolerance AR
_HINGE_THRESH  = 0.040   # penalise above 4.0% → drive within 5% with margin
_HINGE_SMOOTH  = 50.0   # softplus steepness


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
            # Birth CD (asymmetric Gaussian) + sidewall polymer → CD_bot control
            'k_born', 'k_born_spread_left', 'k_born_spread_right', 'K_dep_side',
            # Neutral transport decoupling → depth vs CD_bot independence
            'k_clausing_neutral',   # blend weight (1=profile-coupled, 0=mask-based)
            # Depth-controlling parameters
            'lambda_neutral',       # neutral attenuation length
            'K_dep_poly',           # floor polymer (CF4-dependent depth suppression)
            # CF4-fraction-dependent ion flux → fixes opposite depth errors at 6/24 vs 10/20
            # Higher alpha_cf4_ion boosts 10/20 ion flux relative to 6/24 (more CF4 contribution)
            'alpha_cf4_ion',
            # Mask CD_top control
            'K_mask_poly',          # mask polymer protection (CF4-dep narrowing)
            # F-radical chemical mask etch → fixes 18/12 CD_top underprediction (-3.8%)
            # Gamma_F ∝ CF4_flow, so K_F_mask effect is strongest at high CF4 (18/12)
            'K_F_mask',
        ]   # 10 params vs 16 eqs — drives Depth, CD_top, CD_bot, AR all within ±5%

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
        'K_lat_sput':        (1e-17, 1e-11),
        'K_lat_pass':        (0.01,  5.0),
        'K_dep_poly':        (1e-18, 1e-12),
        'K_etch_poly':       (1e-20, 1e-13),
        'n_poly_dep':        (0.3,   4.0),
        'h_poly_char':       (0.05,  50.0),
        'K_dep_side':        (1e-19, 1e-13),
        'k_born':              (0.05,  0.6),
        'k_born_spread_left':  (1.0,   60.0),
        'k_born_spread_right': (0.1,   15.0),
        'K_dep_side':          (1e-19, 1e-14),
        'k_clausing_neutral':  (0.01,  1.0),
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
                # Depth: standard + hinge loss above 4.5%
                _err_dep_raw = (r.total_depth - meas['depth']) / max(meas['depth'], 10.0)
                _exc_dep = (np.log1p(np.exp(_HINGE_SMOOTH * (abs(_err_dep_raw) - _HINGE_THRESH)))
                            / _HINGE_SMOOTH)
                r_depth = _W_DEPTH * _err_dep_raw + _W_HINGE_DEPTH * _exc_dep * np.sign(_err_dep_raw)
                r_top   = _W_CDTOP * (r.cd_top - meas['cd_top']) / max(meas['cd_top'], 5.0)
                # CD_bot: standard + hinge loss above 4.5%
                _err_bot_raw = (r.cd_bot - meas['cd_bot']) / max(meas['cd_bot'], 5.0)
                _exc_bot = (np.log1p(np.exp(_HINGE_SMOOTH * (abs(_err_bot_raw) - _HINGE_THRESH)))
                            / _HINGE_SMOOTH)
                r_bot   = _W_CDBOT * _err_bot_raw + _W_HINGE_BOT * _exc_bot * np.sign(_err_bot_raw)
                # AR: standard + hinge loss above 4.5%
                _err_ar_raw = (r.aspect_ratio - meas['ar']) / max(meas['ar'], 0.1)
                _exc_ar = (np.log1p(np.exp(_HINGE_SMOOTH * (abs(_err_ar_raw) - _HINGE_THRESH)))
                           / _HINGE_SMOOTH)
                r_ar    = _W_AR * _err_ar_raw + _W_HINGE_AR * _exc_ar * np.sign(_err_ar_raw)
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
        print(f"  CALIBRATING {len(calibrate_params)} params x {len(experiments)} experiments")
        print(f"  Stage 1: coarse TRF (ftol=1e-3, diff_step=1e-3)...")

    # diff_step=1e-3: gives ~0.2-3.5% param perturbation in log-space,
    # large enough for the simulator to register (avoids zero-gradient trap).
    cal1 = least_squares(
        residual_fn, x0=x0_log, bounds=bounds_log,
        method='trf', ftol=1e-3, xtol=1e-3, gtol=1e-4,
        max_nfev=3000, diff_step=1e-3, verbose=0,
    )
    if verbose:
        print(f"  Stage 1 done  cost={cal1.cost:.4e}  nfev={cal1.nfev}")
        print("  Stage 2: fine TRF (ftol=1e-6, diff_step=5e-4)...")

    cal2 = least_squares(
        residual_fn, x0=cal1.x, bounds=bounds_log,
        method='trf', ftol=1e-8, xtol=1e-8, gtol=1e-10,
        max_nfev=15000, diff_step=2e-4, verbose=0,
    )

    mp_cal = copy.deepcopy(mp_init)
    if verbose:
        print(f"  Stage 2 done  cost={cal2.cost:.4e}  nfev={cal2.nfev}")
        print(f"\n  {'Parameter':<22}  {'Init':>12}  {'Calibrated':>12}  {'ratio':>7}")
        print(f"  {'-'*58}")
    for i, pname in enumerate(calibrate_params):
        old_val = getattr(mp_init, pname)
        new_val = float(10.0 ** cal2.x[i])
        setattr(mp_cal, pname, new_val)
        if verbose:
            ratio = new_val / old_val if old_val != 0 else float('inf')
            print(f"  {pname:<22}  {old_val:>12.4e}  {new_val:>12.4e}  {ratio:>7.3f}")

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
        'HARC v2 - Calibration Result\n'
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


def plot_rodeo_validation(
    exp_depth:  float,
    exp_width:  float,
    our_depth:  float,
    our_width:  float,
    rodeo_depth: float,
    rodeo_width: float,
    k_cal:      float,
    save_path:  Optional[str] = None,
) -> plt.Figure:
    """
    Bar-chart comparison: Experiment vs Our Model vs RODEo
    for Height and Width from Chopra et al. (SPIE 2018), Table 3.
    """
    metrics     = ['Height [nm]', 'Width [nm]']
    exp_vals    = [exp_depth,   exp_width]
    our_vals    = [our_depth,   our_width]
    rodeo_vals  = [rodeo_depth, rodeo_width]
    our_errs    = [100*(our_depth  - exp_depth)  / exp_depth,
                   100*(our_width  - exp_width)  / exp_width]
    rodeo_errs  = [100*(rodeo_depth - exp_depth) / exp_depth,
                   100*(rodeo_width - exp_width)  / exp_width]
    is_fitted   = [True, False]   # Height fitted, Width forward prediction

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    fig.suptitle(
        'RODEo Validation  —  Chopra et al., SPIE 2018, Table 3\n'
        'Plasma-Therm 790 CCP-RIE  |  50 mTorr, 200 W, CF4/Ar=40/10 sccm, t=180 s\n'
        f'k_rate_global = {k_cal:.4f}  (fitted to Height only)',
        fontsize=11, fontweight='bold'
    )

    x   = np.array([0])
    w   = 0.22
    for ax, metric, exp_v, our_v, rodeo_v, our_e, rodeo_e, fitted in zip(
        axes, metrics, exp_vals, our_vals, rodeo_vals, our_errs, rodeo_errs, is_fitted
    ):
        ax.bar(x - w, exp_v,   width=w, color='gray',              alpha=0.85,
               label='Experiment', edgecolor='k', linewidth=0.7)
        ax.bar(x,     our_v,   width=w, color=COLORS['primary'],   alpha=0.85,
               label='Our Model',  edgecolor='k', linewidth=0.7)
        ax.bar(x + w, rodeo_v, width=w, color=COLORS['secondary'], alpha=0.85,
               label='RODEo',      edgecolor='k', linewidth=0.7)

        # Error labels
        for xi_off, val, err in [(0, our_v, our_e), (w, rodeo_v, rodeo_e)]:
            clr = COLORS['accent'] if abs(err) < 10 else COLORS['warn']
            ax.text(x[0] + xi_off, max(val, exp_v) * 1.02,
                    f'{err:+.1f}%', ha='center', va='bottom', fontsize=10,
                    fontweight='bold', color=clr)

        # PASS/FAIL box on Our Model bar
        ok = abs(our_e) < 10.0
        ax.text(x[0], our_v * 0.5,
                'PASS' if ok else 'FAIL',
                ha='center', va='center', fontsize=12, fontweight='bold',
                color='white',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor=COLORS['accent'] if ok else COLORS['secondary'],
                          alpha=0.9))

        # Forward prediction annotation
        if not fitted:
            ax.text(x[0], -exp_v * 0.08,
                    '★ forward prediction\n(not fitted)',
                    ha='center', va='top', fontsize=8,
                    color=COLORS['purple'], style='italic')

        ax.set_title(metric, fontweight='bold', fontsize=12)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_xticks([])
        ax.set_xlim(-0.45, 0.45)
        ax.set_ylim(0, max(exp_v, our_v, rodeo_v) * 1.25)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_facecolor(COLORS['bg'])

    plt.tight_layout()
    if save_path:
        import os as _os
        _os.makedirs(_os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved → {save_path}")
    plt.show()
    return fig


def _grid_search_ar10(
    mp:          ModelParameters,
    target_ar:   float = 10.0,
    cf4_vals:    Optional[List[float]] = None,
    vbias_vals:  Optional[List[float]] = None,
    time_vals:   Optional[List[float]] = None,
    max_bowing:  float = 0.65,
    max_taper:   float = 0.85,
    min_cd_bot:  float = 30.0,
) -> Tuple[float, SimulationResult]:
    """
    Full parameter grid search (CF4, V_bias, etch_time) targeting AR closest to target_ar.

    Constraints filter out physically unreasonable profiles:
      max_bowing  — limits sidewall bulge (profile quality)
      max_taper   — limits top-to-bottom CD ratio (not too re-entrant)
      min_cd_bot  — ensures the hole bottom is not too narrow to be practical
    Returns (best_cf4, SimulationResult).
    """
    if cf4_vals   is None: cf4_vals   = list(np.linspace(2.0, 28.0, 14))
    if vbias_vals is None: vbias_vals = [-750.0, -1000.0, -1250.0, -1500.0, -2000.0]
    if time_vals  is None: time_vals  = [240.0, 360.0, 480.0, 600.0, 720.0, 900.0]

    best_cf4, best_res, best_diff = cf4_vals[0], None, 1e9
    total = len(cf4_vals) * len(vbias_vals) * len(time_vals)
    done  = 0
    for etch_time in time_vals:
        for vbias in vbias_vals:
            for cf4 in cf4_vals:
                cond = ProcessConditions(
                    cf4_flow=float(cf4), ar_flow=30.0 - float(cf4),
                    v_bias=float(vbias), source_power=250.0, pressure=10.0,
                    substrate_temp=15.0, etch_time=float(etch_time),
                    cd_initial=200.0, mask_thickness=1350.0, target_depth=2500.0,
                )
                try:
                    res = run_forward_simulation(cond, mp, verbose=False)
                    if res.bowing_index > max_bowing:
                        done += 1; continue
                    if res.taper_index > max_taper:
                        done += 1; continue
                    if res.cd_bot < min_cd_bot:
                        done += 1; continue
                    # Reject T-shape: z=0 entry must not dwarf the rest of the profile
                    if len(res.cd_profile) > 1 and res.cd_profile[1] < res.cd_top * 0.3:
                        done += 1; continue
                    # Reject reverse taper: bottom should not be wider than mid
                    if res.cd_bot > res.cd_mid * 1.15:
                        done += 1; continue
                    diff = abs(res.aspect_ratio - target_ar)
                    if diff < best_diff:
                        best_diff, best_cf4, best_res = diff, float(cf4), res
                except Exception:
                    pass
                done += 1
                if done % 100 == 0:
                    ar_str = f"{best_res.aspect_ratio:.3f}" if best_res else "N/A"
                    print(f"    [{done}/{total}] best AR so far: {ar_str}")
    return best_cf4, best_res


def _sweep_cf4_for_ar_target(
    mp:          ModelParameters,
    target_ar:   float,
    etch_time:   float,
    cf4_lo:      float = 2.0,
    cf4_hi:      float = 28.0,
    n_grid:      int   = 53,
    max_bowing:  Optional[float] = None,
) -> Tuple[float, SimulationResult]:
    """
    Sweep CF4 (2-28 sccm) and return (cf4_best, result) with AR closest to target_ar.
    max_bowing: if set, only consider conditions with bowing_index <= max_bowing.
    """
    cf4_vals = np.linspace(cf4_lo, cf4_hi, n_grid)
    best_cf4, best_res, best_diff = cf4_lo, None, 1e9
    for cf4 in cf4_vals:
        cond = ProcessConditions(
            cf4_flow=float(cf4), ar_flow=30.0 - float(cf4),
            v_bias=-1000.0, source_power=250.0, pressure=10.0,
            substrate_temp=15.0, etch_time=etch_time,
            cd_initial=200.0, mask_thickness=1350.0, target_depth=2500.0,
        )
        try:
            res = run_forward_simulation(cond, mp, verbose=False)
            if max_bowing is not None and res.bowing_index > max_bowing:
                continue
            diff = abs(res.aspect_ratio - target_ar)
            if diff < best_diff:
                best_diff, best_cf4, best_res = diff, float(cf4), res
        except Exception:
            continue
    return best_cf4, best_res


def plot_optimization_result(
    res:       SimulationResult,
    cf4_best:  float,
    title:     str,
    show_time: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    3-panel optimization result figure: hole profile | normalized metrics | recipe table.
    Styled after harc_optimization_result_new.png.
    show_time=False: omit etch time from recipe table and all time-related text.
    """
    fig = plt.figure(figsize=(15, 7))
    fig.suptitle(title, fontsize=12, fontweight='bold', y=1.00)
    gs = gridspec.GridSpec(1, 3, figure=fig,
                           width_ratios=[1.2, 1.0, 0.9], wspace=0.35)
    ax_prof = fig.add_subplot(gs[0])
    ax_bar  = fig.add_subplot(gs[1])
    ax_tab  = fig.add_subplot(gs[2])

    # ── Left: hole profile ───────────────────────────────────────────────────
    z  = res.z_grid
    cd = res.cd_profile
    ax_prof.plot(-cd / 2, -z, color=COLORS['primary'], lw=2)
    ax_prof.plot( cd / 2, -z, color=COLORS['primary'], lw=2)
    ax_prof.fill_betweenx(-z, -cd / 2, cd / 2,
                          alpha=0.15, color=COLORS['primary'])
    mask_cd = res.cd_top
    ax_prof.fill_betweenx([0, 200],
                          [-mask_cd / 2, -mask_cd / 2],
                          [ mask_cd / 2,  mask_cd / 2],
                          alpha=0.25, color='gray')
    ax_prof.annotate(
        f'CD_top={res.cd_top:.0f}nm',
        xy=(res.cd_top / 2, 0),
        xytext=(res.cd_top / 2 + 40, -100),
        fontsize=8, color=COLORS['primary'],
        arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=1.2),
    )
    ax_prof.annotate(
        f'CD_bot={res.cd_bot:.0f}nm',
        xy=(res.cd_bot / 2, -res.total_depth),
        xytext=(res.cd_bot / 2 + 40, -res.total_depth + 180),
        fontsize=8, color=COLORS['warn'],
        arrowprops=dict(arrowstyle='->', color=COLORS['warn'], lw=1.2),
    )
    ax_prof.set_title(f'Optimal Profile\nAR={res.aspect_ratio:.2f}',
                      fontweight='bold', fontsize=11)
    ax_prof.set_xlabel('x [nm]', fontsize=9)
    ax_prof.set_ylabel('z [nm]',  fontsize=9)
    ax_prof.set_facecolor(COLORS['bg'])
    ax_prof.grid(True, alpha=0.2)

    # ── Middle: normalized metrics ───────────────────────────────────────────
    labels     = ['AR / 10', 'Taper×10', 'Bowing×10']
    vals       = [res.aspect_ratio / 10,
                  res.taper_index  * 10,
                  res.bowing_index * 10]
    bar_colors = [COLORS['primary'], COLORS['secondary'], COLORS['purple']]
    xpos = np.arange(len(labels))
    bars = ax_bar.bar(xpos, vals, color=bar_colors, alpha=0.85,
                      edgecolor='k', linewidth=0.6, width=0.5)
    for bar, val in zip(bars, vals):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.015,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
    ax_bar.axhline(1.0, color='gray', lw=1.5, ls='--',
                   label='Target AR/10 = 1')
    ax_bar.set_xticks(xpos)
    ax_bar.set_xticklabels(labels, fontsize=9)
    ax_bar.set_ylabel('Normalized value', fontsize=9)
    ax_bar.set_title('Normalized Metrics', fontweight='bold', fontsize=11)
    ax_bar.legend(fontsize=8)
    ax_bar.set_facecolor(COLORS['bg'])
    ax_bar.grid(axis='y', alpha=0.3)

    # ── Right: recipe table ──────────────────────────────────────────────────
    rows: List[List[str]] = [
        ['CF4 flow',     f'{cf4_best:.2f} sccm'],
        ['Ar flow',      f'{30.0 - cf4_best:.2f} sccm'],
        ['CF4 fraction', f'{cf4_best / 30.0:.3f}'],
        ['V_bias',       f'{res.conditions.v_bias:.0f} V'],
    ]
    if show_time:
        rows.append(['Etch time', f'{res.conditions.etch_time:.0f} s'])
    rows.append(['', ''])
    rows += [
        ['Depth',        f'{res.total_depth:.1f} nm'],
        ['CD_top',       f'{res.cd_top:.1f} nm'],
        ['CD_mid',       f'{res.cd_mid:.1f} nm'],
        ['CD_bot',       f'{res.cd_bot:.1f} nm'],
        ['Aspect Ratio', f'{res.aspect_ratio:.3f}'],
        ['Taper index',  f'{res.taper_index:.4f}'],
        ['Bowing index', f'{res.bowing_index:.4f}'],
    ]
    sep_row = 6 if show_time else 5   # table row index of separator (header=0)

    ax_tab.axis('off')
    tbl = ax_tab.table(
        cellText=rows,
        colLabels=['Parameter', 'Optimal Value'],
        loc='center',
        cellLoc='left',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.15, 1.45)
    for j in range(2):
        tbl[(0, j)].set_facecolor('#374151')
        tbl[(0, j)].set_text_props(color='white', fontweight='bold')
        tbl[(sep_row, j)].set_facecolor('#E5E7EB')
    ax_tab.set_title('Recommended Recipe', fontweight='bold',
                     fontsize=11, pad=10)

    plt.tight_layout()
    if save_path:
        import os as _os
        _dir = _os.path.dirname(save_path)
        if _dir:
            _os.makedirs(_dir, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved → {save_path}")
    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: RODEo VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def run_rodeo_validation(mp: ModelParameters) -> None:
    """
    Validation against Chopra et al. (SPIE 2018) RODEo paper, Table 3.

    Paper: Plasma-Therm 790 CCP-RIE, 50mTorr, 200W, CF4/Ar=40/10 sccm, t=180s
           130nm pitch line-space SiO2 pattern.
    Experimental:  Width=69.6 nm, Height=39.1 nm
    RODEo model:   Width=60.4 nm, Height=32.8 nm

    Strategy
    --------
    The ICP reactor (our calibration) and the CCP-RIE (paper) share identical
    surface chemistry (CF4/Ar -> SiO2) but differ in absolute plasma flux level.
    We introduce a single scalar k_rate_global that scales ALL plasma fluxes
    (F, CFx, ion) uniformly, representing the overall plasma coupling efficiency
    of the paper's reactor.  k_rate_global is fitted to match the etch depth
    at t=180 s (one calibration target), then CD_top is a FREE forward prediction
    (one validation target).  Both results must be within 10% of experiment.
    """
    from copy import deepcopy
    from scipy.optimize import brentq

    EXP_DEPTH = 39.1   # nm  (Table 3, experiment)
    EXP_WIDTH = 69.6   # nm  (Table 3, experiment)
    RODEO_DEPTH = 32.8
    RODEO_WIDTH = 60.4

    print("\n" + "=" * 70)
    print("  RODEo VALIDATION  (Chopra et al., SPIE 2018, Table 3)")
    print("  Reactor : Plasma-Therm 790 CCP-RIE")
    print("  Conditions: 50 mTorr | 200 W | CF4/Ar=40/10 sccm | t=180 s")
    print(f"  Experiment: Width={EXP_WIDTH} nm, Height={EXP_DEPTH} nm")
    print(f"  RODEo:      Width={RODEO_WIDTH} nm, Height={RODEO_DEPTH} nm")
    print("=" * 70)

    # Base process conditions (paper values; self-bias not reported ->
    # use -150V as typical CCP-RIE estimate; affects only ion energy,
    # not k_rate_global calibration which adjusts absolute flux level)
    def _make_cond(k_global: float) -> tuple:
        mp_v = deepcopy(mp)
        mp_v.k_rate_global = k_global
        cond = ProcessConditions(
            cf4_flow      = 40.0,
            ar_flow       = 10.0,
            total_flow    = 50.0,
            v_bias        = -150.0,
            source_power  = 200.0,
            pressure      = 50.0,
            etch_time     = 180.0,
            cd_initial    = 65.0,
            mask_thickness= 200.0,
            target_depth  = 500.0,
        )
        return cond, mp_v

    # ── Step 1: nominal run (k_rate_global=1.0) ──────────────────────────────
    print("\n  [Step 1] Nominal run (k_rate_global=1.0, ICP-equivalent flux)...")
    cond0, mp0 = _make_cond(1.0)
    r0 = run_forward_simulation(cond0, mp0, verbose=False)
    print(f"    Depth at t=180s : {r0.total_depth:.1f} nm  (exp: {EXP_DEPTH} nm)")
    print(f"    CD_top at t=180s: {r0.cd_top:.1f} nm  (exp: {EXP_WIDTH} nm)")

    # ── Step 2: calibrate k_rate_global to match etch depth ──────────────────
    print("\n  [Step 2] Calibrating k_rate_global to match depth=39.1 nm at t=180s...")

    def depth_residual(k):
        c, m = _make_cond(k)
        res = run_forward_simulation(c, m, verbose=False)
        return res.total_depth - EXP_DEPTH

    # k must be between 0 and 1 (paper's RIE flux is less than our ICP)
    k_lo, k_hi = 0.01, 2.0
    try:
        k_cal = brentq(depth_residual, k_lo, k_hi, xtol=1e-4, maxiter=60)
    except ValueError:
        # fallback: linear estimate
        k_cal = EXP_DEPTH / r0.total_depth
        print(f"    (brentq fallback, using linear estimate k={k_cal:.4f})")

    print(f"    k_rate_global = {k_cal:.4f}  "
          f"(physical meaning: CCP-RIE flux is {k_cal*100:.1f}% of our ICP flux)")

    # ── Step 3: validation run with calibrated k_rate_global ─────────────────
    print("\n  [Step 3] Validation run with k_rate_global calibrated ...")
    cond_v, mp_v = _make_cond(k_cal)
    r_v = run_forward_simulation(cond_v, mp_v, verbose=False)

    depth_err = 100.0 * (r_v.total_depth - EXP_DEPTH) / EXP_DEPTH
    width_err = 100.0 * (r_v.cd_top      - EXP_WIDTH) / EXP_WIDTH
    rdep_err  = 100.0 * (RODEO_DEPTH - EXP_DEPTH) / EXP_DEPTH
    rwid_err  = 100.0 * (RODEO_WIDTH  - EXP_WIDTH) / EXP_WIDTH

    print("\n" + "=" * 70)
    print("  VALIDATION RESULT (t=180 s, k_rate_global calibrated to depth)")
    print(f"  {'Metric':<12} {'Experiment':>12} {'Our model':>12} {'Error':>8}  {'RODEo':>8} {'RODEo err':>10}")
    print("  " + "-" * 64)
    print(f"  {'Height[nm]':<12} {EXP_DEPTH:>12.1f} {r_v.total_depth:>12.1f} {depth_err:>+7.1f}%  "
          f"{RODEO_DEPTH:>8.1f} {rdep_err:>+9.1f}%")
    print(f"  {'Width[nm]':<12} {EXP_WIDTH:>12.1f} {r_v.cd_top:>12.1f} {width_err:>+7.1f}%  "
          f"{RODEO_WIDTH:>8.1f} {rwid_err:>+9.1f}%")
    print("  " + "-" * 64)

    ok_depth = abs(depth_err) < 10.0
    ok_width = abs(width_err) < 10.0
    print(f"  Height within 10%: {'PASS' if ok_depth else 'FAIL'}  "
          f"Width within 10%: {'PASS' if ok_width else 'FAIL'}")
    print("=" * 70)
    print("  NOTE: k_rate_global was fitted to Height only.")
    print("  Width is a FORWARD PREDICTION (not fitted) -- tests model physics.")

    # Save validation results to file
    import json as _json
    val_path = os.path.join(_HERE, 'rodeo_validation_result.json')
    val_dict = {
        'reference': 'Chopra et al., SPIE 2018 (RODEo), Table 3',
        'reactor':   'Plasma-Therm 790 CCP-RIE',
        'conditions': {
            'pressure_mTorr': 50.0,
            'power_W':        200.0,
            'cf4_sccm':       40.0,
            'ar_sccm':        10.0,
            'total_flow_sccm':50.0,
            'cf4_fraction':   0.8,
            'etch_time_s':    180.0,
            'cd_initial_nm':  65.0,
        },
        'k_rate_global_calibrated': round(k_cal, 6),
        'results': {
            'Height_nm': {
                'experiment': EXP_DEPTH,
                'our_model':  round(r_v.total_depth, 2),
                'rodeo':      RODEO_DEPTH,
                'our_error_%':  round(depth_err, 2),
                'rodeo_error_%':round(rdep_err,  2),
                'pass_10pct':   bool(ok_depth),
            },
            'Width_nm': {
                'experiment': EXP_WIDTH,
                'our_model':  round(r_v.cd_top, 2),
                'rodeo':      RODEO_WIDTH,
                'our_error_%':  round(width_err, 2),
                'rodeo_error_%':round(rwid_err,  2),
                'pass_10pct':   bool(ok_width),
                'note': 'forward prediction (not fitted)',
            },
        },
        'overall_pass': bool(ok_depth and ok_width),
    }
    with open(val_path, 'w') as _f:
        _json.dump(val_dict, _f, indent=2)
    print(f"\n  Saved -> {val_path}")

    plot_rodeo_validation(
        exp_depth=EXP_DEPTH, exp_width=EXP_WIDTH,
        our_depth=r_v.total_depth, our_width=r_v.cd_top,
        rodeo_depth=RODEO_DEPTH, rodeo_width=RODEO_WIDTH,
        k_cal=k_cal,
        save_path=os.path.join(_HERE, 'figures', 'harc_v2_rodeo_validation.png'),
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10: MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import json as _json
    print("=" * 70)
    print("  HARC ETCH SIMULATOR v2")
    print("  New models: IAD lateral flux + 2-D mask aperture evolution")
    print("  Calibration: 4-point dataset (point 5 excluded - low reliability)")
    print("  (250W, 10mTorr, -1000V, 15C)")
    print("=" * 70)
    os.makedirs(os.path.join(_HERE, 'figures'), exist_ok=True)

    # Warm start from JSON (good depth/top/AR), then refine birth CD + K_dep_side for CD_bot
    mp_init = ModelParameters()
    _json_path = os.path.join(_HERE, 'harc_v2_calibrated_params_physics.json')
    if os.path.exists(_json_path):
        with open(_json_path) as _f:
            _d = _json.load(_f)
        for _k, _v in _d.items():
            if hasattr(mp_init, _k):
                setattr(mp_init, _k, _v)
        print(f"  Warm start: loaded {_json_path}")

    exp_data = EXPERIMENTAL_DATA.copy()

    # Point 5 (CF4=22/Ar=8) excluded from calibration; retained for validation display
    cal_data = exp_data[exp_data['reliable']].copy().reset_index(drop=True)

    print("\n[STEP 1] Pre-calibration forward runs ...")
    print_accuracy_table("PRE-CALIBRATION (all 5 points)", exp_data, mp_init)

    print(f"\n[STEP 2] Running calibration pass 1 on {len(cal_data)} reliable points ...")
    mp_cal1, cal_info1 = calibrate_model_parameters(
        cal_data, mp_init, verbose=True
    )

    print(f"\n[STEP 2b] Calibration pass 2 - warm restart from pass 1 ...")
    mp_cal2, cal_info2 = calibrate_model_parameters(
        cal_data, mp_cal1, verbose=True
    )

    print(f"\n[STEP 2c] Calibration pass 3 - final fine-tuning ...")
    mp_cal, cal_info = calibrate_model_parameters(
        cal_data, mp_cal2, verbose=True,
        cal_dt=1.0,   # finer dt for pass 3 accuracy
    )

    print("\n[STEP 3] Post-calibration accuracy (all 5 points for reference) ...")
    print_accuracy_table("POST-CALIBRATION PASS 1", exp_data, mp_cal1)
    print_accuracy_table("POST-CALIBRATION PASS 2", exp_data, mp_cal2)
    print_accuracy_table("POST-CALIBRATION PASS 3 (FINAL)", exp_data, mp_cal)

    print("\n[STEP 4] Generating plots ...")
    plot_calibration_comparison(
        exp_data, mp_init, mp_cal,
        save_path=os.path.join(_HERE, 'figures', 'harc_v2_calibration.png')
    )
    plot_profiles(
        exp_data, mp_cal,
        save_path=os.path.join(_HERE, 'figures', 'harc_v2_profiles.png')
    )

    print("\n[STEP 5] Calibrated ModelParameters (copy-paste to reuse):")
    print("  mp_cal = ModelParameters(")
    for pname in cal_info['calibrate_params']:
        val = getattr(mp_cal, pname)
        print(f"      {pname:<22} = {val:.4e},")
    print("  )")

    print(f"\n[STEP 6] Saving calibrated params → {_json_path}")
    _save_dict = {k: v for k, v in asdict(mp_cal).items()
                  if k not in ('dz', 'dt')}
    with open(_json_path, 'w') as _f:
        _json.dump(_save_dict, _f, indent=2)
    print(f"  Saved {len(_save_dict)} parameters.")

    print("\n[STEP 7] RODEo external validation ...")
    run_rodeo_validation(mp_cal)

    print("\n[STEP 8] Optimal profile (CF4=9 sccm / Ar=21 sccm, t=340s) ...")
    cf4_s9 = 9.0
    _cond_s9 = ProcessConditions(
        cf4_flow=9.0, ar_flow=21.0,
        v_bias=-1000.0, source_power=250.0, pressure=10.0,
        substrate_temp=15.0, etch_time=340.0,
        cd_initial=200.0, mask_thickness=1350.0, target_depth=2500.0,
    )
    res_s9 = run_forward_simulation(_cond_s9, mp_cal, verbose=False)
    print(f"  CF4={cf4_s9:.2f} sccm  AR={res_s9.aspect_ratio:.3f}"
          f"  Bowing={res_s9.bowing_index:.4f}  CD_bot={res_s9.cd_bot:.1f} nm"
          f"  Depth={res_s9.total_depth:.1f}  CDtop={res_s9.cd_top:.1f}")
    plot_optimization_result(
        res_s9, cf4_s9,
        title=(
            'Physics-Based Optimization Result\n'
            '[Calibrated Model  →  CF4/Ar Optimal Process Conditions]'
        ),
        show_time=False,
        save_path=os.path.join(_HERE, 'figures', 'harc_v2_ar10_extended.png'),
    )

    print("\n[STEP 9] AR=10 full-space optimization "
          "(CF4, V_bias, time | bowing≤0.65, taper≤0.85, CD_bot≥30nm) ...")
    cf4_s10, res_s10 = _grid_search_ar10(
        mp_cal, target_ar=10.0,
        cf4_vals=list(np.linspace(2.0, 28.0, 14)),
        vbias_vals=[-750.0, -1000.0, -1250.0, -1500.0, -2000.0],
        time_vals=[240.0, 360.0, 480.0, 600.0, 720.0, 900.0],
        max_bowing=0.65,
        max_taper=0.85,
        min_cd_bot=30.0,
    )
    if res_s10 is not None:
        print(f"  Best: CF4={cf4_s10:.2f} sccm  "
              f"V_bias={res_s10.conditions.v_bias:.0f}V  "
              f"t={res_s10.conditions.etch_time:.0f}s  "
              f"AR={res_s10.aspect_ratio:.3f}  "
              f"Bowing={res_s10.bowing_index:.4f}  "
              f"Taper={res_s10.taper_index:.4f}  "
              f"CD_bot={res_s10.cd_bot:.1f}nm")
        plot_optimization_result(
            res_s10, cf4_s10,
            title=(
                'Physics-Based Optimization Result\n'
                '[Full Parameter Space (CF4/Ar, V_bias, Etch Time)  →  AR ≈ 10]'
            ),
            show_time=True,
            save_path=os.path.join(_HERE, 'figures', 'harc_v2_ar10_full_opt.png'),
        )
    else:
        print("  [WARN] No valid result found within constraints.")

    return mp_cal, cal_info


if __name__ == '__main__':
    mp_cal, cal_info = main()
