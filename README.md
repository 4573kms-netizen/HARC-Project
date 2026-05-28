# HARC Etch Physics-Based Simulator v2

Physics-based forward simulation and inverse optimization for High Aspect Ratio Contact (HARC) etching using CF4/Ar plasma.

## Quick Start

```bash
# 1. Clone
git clone https://github.com/4573kms-netizen/HARC-Project.git
cd HARC-Project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run (generates all 5 figures using calibrated parameters)
python regen_figures.py
```

---

## Overview

This simulator models the full HARC etch process from plasma generation through feature-scale profile evolution. It supports:

- **Forward simulation**: predict etch depth, CD profile, AR given process conditions
- **Calibration**: fit 10 model parameters to experimental SEM data via bounded least-squares (3-pass TRF with hinge-loss penalty)
- **Inverse optimization**: find CF4/Ar recipe that maximizes aspect ratio
- **External validation**: comparison against Chopra et al. (SPIE 2018) RODEo model

Calibrated conditions: Source 250 W, 10 mTorr, V_bias = −1000 V, 15 °C electrode, 240 s etch time.

---

## File Structure

```
harc_etch_simulator_v2.py             Main simulator (forward sim + calibration + plots)
regen_figures.py                      Regenerate all 5 figures from calibrated JSON (fast)
run_inverse_opt.py                    Inverse optimization (maximize AR at fixed V_bias)
make_cal_figure.py                    Calibration bar-chart figure
make_summary_table.py                 Publication-quality summary table (cal + RODEo)
sweep_340s_ar10.py                    CF4/Ar sweep at t=340s targeting AR=10
harc_v2_calibrated_params_physics.json   Calibrated model parameters (39 params)
requirements.txt                      Python dependencies
```

All scripts are location-independent — they can be run from any directory.

---

## Physics Models

### 1. 0-D Global Plasma Model

Estimates species fluxes at the wafer surface from bulk process inputs.

| Species | Formula |
|---------|---------|
| F radical | `Γ_F = A_F · √P · √Q_CF4 · β_F · f_p · f_sat` |
| CFx radical | `Γ_CFx = A_CFx · √P · √Q_tot · β_CFx · f_cf4 · (1 − 0.5·f_cf4) · f_p` |
| Ion | `Γ_ion = A_ion · √P · f_p_ion · (f_Ar + α_CF4·f_CF4)^β_ion` |

- **`f_sat`**: CF4 saturation factor — high CF4 fraction suppresses F-radical yield.  
  `f_sat = max(1 − γ_F_sat · f_CF4, 0.05)`
- **`α_CF4_ion`**: CF4 fragment ionization efficiency relative to Ar. Prevents ion flux collapse at high CF4 since CF3⁺, CF2⁺ still contribute.

---

### 2. Sheath / Ion Energy Model

```
E_ion = α_E · |V_bias| + E_thermal
E_ion_eff = E_ion / (1 + (p − 10) / 200)
```

---

### 3. Ion Angular Distribution (IAD) — Gaussian Model

```
f(θ) ∝ exp(−θ² / 2σ_iad²)
```

`σ_iad ≈ 0.30 rad (~17°)` primarily governs **lateral sidewall flux**. Vertical transport uses a separate Clausing model.

---

### 4. Vertical Ion Transport — Clausing Power-Law

```
T_v(z) = η_ion / (1 + AR_mask^n_clausing)
```

where `AR_mask = z / CD_mask`.

---

### 5. Lateral Ion Transport — IAD Acceptance-Cone Model

```
θ_sw(z) = arctan(r_mask / z)
T_lat(z) = exp(−θ_sw² / 2σ_iad²) · sin(θ_sw) · cos(θ_sw) · shadow(z)
```

---

### 6. Neutral Transport — Exponential Attenuation

```
T_n(z) = exp(−z / (λ_neutral · CD_top))
```

`λ_neutral ≈ 7.16 CD_top` (calibrated).

---

### 7. Surface Reaction Model — Vertical Etch Rate

```
R_v(z) = K_chem · Γ_F(z)
        + K_ie  · Γ_F(z) · Γ_ion(z) · f_IE(E)
        + K_sput · Γ_ion(z) · Y_s(E)
        − K_pass · Γ_CFx(z)
```

---

### 8. Bohdansky Sputtering Yield

```
Y_s(E) = Q_s · [1 − (E_th/E)^0.5]²     (E > E_th)
```

`E_th ≈ 20 eV`, `Q_s ≈ 0.042`.

---

### 9. Lateral Etch Rate — Sidewall Model

```
R_lat(z) = K_lat_neu · Γ_F(z)
          + K_lat_ion · Γ_F(z) · T_lat(z) · f_IE
          + K_lat_sput · T_lat(z) · Y_s
          − K_pass · K_lat_pass · Γ_CFx(z)
          − K_dep_side · Γ_CFx(z)
```

---

### 10. Mask Aperture Evolution

```
dCD_mask/dt = 2 · (R_sput_mask + R_F_mask − R_poly_mask)
```

- **Ar⁺ erosion** widens the mask opening (proportional to Ar fraction).
- **F-radical chemical etch** (`K_F_mask`) explains non-monotonic CD_top at high CF4.
- **CFx polymer** narrows the mask opening.

---

### 11. Birth CD Model — IAD Collimation Effect

```
CD_born(z) = CD_mask · exp(−k_born_eff · AR_birth)

k_born_eff = k_born · exp(−spread · (f_CF4 − f_peak)²)
```

Asymmetric Gaussian: separate spread parameters left/right of peak CF4 fraction (0.33).

---

### 12. Profile Evolution — Explicit Euler Time-Stepping

`dt = 0.5 s`, `dz = 20 nm`. New depth nodes spawned as the etch floor advances.

---

## Calibration Results

**Method**: 3-pass TRF (Trust Region Reflective) with hinge-loss penalty (threshold 4%), 10 free parameters.  
**Dataset**: 4 reliable points (CF4/Ar = 6/24, 10/20, 14/16, 18/12 sccm). Point 5 (CF4=22/Ar=8) excluded — low reliability.

| CF4/Ar [sccm] | Depth err | CD_top err | CD_bot err | AR err |
|---------------|-----------|------------|------------|--------|
| 6 / 24  | +3.3% ✓ | −0.0% ✓ | −2.2% ✓ | +3.3% ✓ |
| 10 / 20 | −3.5% ✓ | +1.4% ✓ | +2.5% ✓ | −4.9% ✓ |
| 14 / 16 | −1.9% ✓ | +1.1% ✓ | −0.5% ✓ | −3.0% ✓ |
| 18 / 12 | +1.1% ✓ | −3.7% ✓ | −2.6% ✓ | +4.8% ✓ |

All 16 metrics within ±5%.

---

## External Validation (RODEo)

Compared against Chopra et al. (SPIE 2018), Table 3 — Plasma-Therm 790 CCP-RIE, 50 mTorr, 200 W, CF4/Ar=40/10 sccm, t=180 s.

| Metric | Experiment | This work | Error | RODEo | RODEo err |
|--------|-----------|-----------|-------|-------|-----------|
| Height [nm] | 39.1 | 39.1 | −0.0% ✓ | 32.8 | −16.1% |
| Width [nm] ★ | 69.6 | 66.5 | −4.5% ✓ | 60.4 | −13.2% |

★ Width is a **forward prediction** (not fitted) — only Height was used to calibrate reactor flux scale.

---

## Inverse Optimization Result

Target: maximize AR at V_bias = −1000 V, total flow = 30 sccm, t = 240 s.

**Optimal: CF4 = 7.50 sccm / Ar = 22.50 sccm → AR = 6.603**

| Metric | Value |
|--------|-------|
| Depth | 1380 nm |
| CD_top | 209 nm |
| CD_bot | 54 nm |
| Taper index | 0.741 |
| Bowing index | 0.625 |

---

## Requirements

```
Python 3.9+
```

```bash
pip install -r requirements.txt
```

---

## Usage

### Regenerate all figures (recommended — uses calibrated JSON, skips re-calibration)
```bash
python regen_figures.py
```
Outputs saved to `figures/`.

### Run full calibration from scratch
```bash
python harc_etch_simulator_v2.py
```

### Inverse optimization
```bash
python run_inverse_opt.py
```

### Forward simulation in Python
```python
from harc_etch_simulator_v2 import ProcessConditions, ModelParameters, run_forward_simulation
import json

with open("harc_v2_calibrated_params_physics.json") as f:
    params = json.load(f)

mp = ModelParameters(**{k: v for k, v in params.items()
                        if hasattr(ModelParameters(), k)})
cond = ProcessConditions(
    cf4_flow=7.5, ar_flow=22.5,
    v_bias=-1000.0, source_power=250.0,
    pressure=10.0, etch_time=240.0,
    cd_initial=200.0
)
result = run_forward_simulation(cond, mp, verbose=True)

print(f"Depth:        {result.total_depth:.1f} nm")
print(f"Aspect Ratio: {result.aspect_ratio:.3f}")
print(f"CD_top:       {result.cd_top:.1f} nm")
print(f"CD_bot:       {result.cd_bot:.1f} nm")
```

---

> **Note:** All default `ModelParameters` values are initial literature estimates, **not calibrated**.
> Use `harc_v2_calibrated_params_physics.json` for quantitative predictions.
