# HARC Etch Physics-Based Simulator v2

Physics-based forward simulation and inverse optimization for High Aspect Ratio Contact (HARC) etching using CF4/Ar plasma.

## Overview

This simulator models the full HARC etch process from plasma generation through feature-scale profile evolution. It supports:

- **Forward simulation**: predict etch depth, CD profile, AR given process conditions
- **Inverse optimization**: find CF4/Ar recipe that maximizes aspect ratio
- **Calibration**: fit 20 model parameters to experimental SEM data via bounded least-squares

Calibrated conditions: Source 250 W, 10 mTorr, V_bias = −1000 V, 15 °C electrode, 240 s etch time.

---

## Physics Models

### 1. 0-D Global Plasma Model

Estimates species fluxes at the wafer surface from bulk process inputs.

| Species | Formula |
|---------|---------|
| F radical | `Γ_F = A_F · √P · √Q_CF4 · β_F · f_p · f_sat` |
| CFx radical | `Γ_CFx = A_CFx · √P · √Q_tot · β_CFx · f_cf4 · (1 − 0.5·f_cf4) · f_p` |
| Ion | `Γ_ion = A_ion · √P · f_p_ion · (f_Ar + α_CF4·f_CF4)^β_ion` |

- **`f_sat`**: CF4 saturation factor — high CF4 fraction suppresses F-radical yield (self-limitation by recombination).  
  `f_sat = max(1 − γ_F_sat · f_CF4, 0.05)`
- **`f_p_neut`**: pressure correction for neutrals (higher pressure → more gas-phase recombination).  
  `f_p_neut = 1 / (1 + p/50)`
- **`f_p_ion`**: pressure correction for ions (higher pressure → more ion–neutral collisions, lower ion flux).  
  `f_p_ion = √(10/p)`
- **`α_CF4_ion`**: CF4 fragment ionization efficiency relative to Ar. Prevents ion flux collapse at high CF4 since CF3⁺, CF2⁺ still contribute. Key for predicting depth at CF4 ≈ 10 sccm.

---

### 2. Sheath / Ion Energy Model

Estimates mean ion bombardment energy using a simplified Child–Langmuir sheath.

```
E_ion = α_E · |V_bias| + E_thermal
E_ion_eff = E_ion / (1 + (p − 10) / 200)
```

- `α_E ≈ 0.61`: fraction of bias potential delivered to ions (accounts for sheath potential drop and collisionality).
- `E_thermal ≈ 5 eV`: plasma potential (minimum ion energy even at zero bias).
- Pressure correction reduces effective energy at higher pressures due to ion–neutral charge-exchange collisions in the sheath.

---

### 3. Ion Angular Distribution (IAD) — Gaussian Model

Ions entering the feature are not perfectly vertical; they follow a Gaussian angular distribution about the surface normal:

```
f(θ) ∝ exp(−θ² / 2σ_iad²)
```

- `σ_iad ≈ 0.30 rad (~17°)`: IAD 1-σ spread, calibrated from sidewall taper data.
- At −1000 V bias and 10 mTorr, the sheath collimates ions strongly. σ_iad primarily governs the **lateral sidewall flux**; vertical transport uses the separate Clausing model (see below).
- This physical decoupling is justified because sheath electric fields collimate vertical ions far beyond what the bulk IAD alone predicts.

---

### 4. Vertical Ion Transport — Clausing Power-Law

Ion transmission to the etch floor decreases with aspect ratio (AR = depth / CD_mask) due to geometric line-of-sight shadowing.

```
T_v(z) = η_ion / (1 + AR_mask^n_clausing)
```

where `AR_mask = z / CD_mask` (depth normalized to mask opening), `η_ion` is the overall beam efficiency, and `n_clausing` is a calibrated exponent.

- At 10 mTorr, the ion mean free path (MFP) ≫ feature depth → ballistic transport; no gas-phase scattering correction needed.
- The Clausing power-law is an empirical geometric model originally derived for effusion through tubes, adapted here for ion beams.

---

### 5. Lateral Ion Transport — IAD Acceptance-Cone Model

Ions that graze the mask edge at angle `θ_sw` relative to the surface normal impinge on the sidewall at depth z:

```
θ_sw(z) = arctan(r_mask / z)

T_lat(z) = exp(−θ_sw² / 2σ_iad²) · sin(θ_sw) · cos(θ_sw) · shadow(z)
```

- `exp(−θ_sw²/2σ²)`: IAD probability that ions exist at this grazing angle.
- `sin(θ_sw)`: flux projected onto the sidewall surface.
- `cos(θ_sw)`: geometric foreshortening (incidence angle factor).
- `shadow(z)`: re-shadowing factor ≤ 1 if the profile narrows above depth z (prevents unphysical "reverse taper" lateral flux).

This model couples IAD spread (`σ_iad`) and mask geometry to predict the depth-dependence of sidewall ion bombardment.

---

### 6. Neutral Transport — Exponential Attenuation

F-radical and CFx fluxes attenuate with depth due to wall sticking. Modeled as an exponential decay along the local (narrowing) profile:

```
T_n(z) = exp(−z / (λ_neutral · CD_top))
```

- `λ_neutral ≈ 14.7 CD_top`: characteristic penetration depth in units of the mask opening.
- Unlike ions, neutrals are isotropic and scatter diffusely off walls (Knudsen diffusion regime at 10 mTorr). The exponential form approximates Knudsen cosine-law transmission through a cylinder.

---

### 7. Surface Reaction Model — Vertical Etch Rate

The vertical etch rate at the feature floor combines three mechanisms minus passivation:

```
R_v(z) = K_chem · Γ_F(z)                          [chemical]
        + K_ie  · Γ_F(z) · Γ_ion(z) · f_IE(E)     [ion-enhanced]
        + K_sput · Γ_ion(z) · Y_s(E)               [physical sputtering]
        − K_pass · Γ_CFx(z) · f_poly(h_poly)       [CFx passivation]
```

- **Chemical**: F radicals spontaneously etch Si via SiF₄ formation (isotropic).
- **Ion-enhanced (IE)**: ion bombardment breaks Si–Si bonds and activates fluorination sites. Rate ∝ Γ_F × Γ_ion (synergy).
- **Sputtering**: physical momentum transfer from Ar⁺/CF⁺ impacts. Governed by Bohdansky yield (see below).
- **Passivation**: CFx radicals deposit a fluorocarbon polymer film that suppresses etch. Modeled via a saturating coverage term.
- **`f_IE(E)`**: ion-energy threshold function — zero below E_threshold (≈15 eV), grows with `√(E/E_th − 1)` above.

---

### 8. Bohdansky Sputtering Yield

Physical sputtering yield Y_s for Si bombarded by Ar⁺ (or CF⁺ fragment ions):

```
Y_s(E) = Q_s · S_n(ε) · [1 − (E_th/E)^(2/3)] · [1 − E_th/E]²
```

where `S_n(ε)` is the nuclear stopping cross-section (Thomas–Fermi reduced energy), and `E_th ≈ 20 eV` is the sputter threshold energy.

- The Bohdansky model is the standard semi-empirical formula for light-ion/heavy-ion sputtering at keV-range energies and below.
- `Q_s ≈ 0.042` is the yield calibration coefficient.

---

### 9. Lateral Etch Rate — Sidewall Model

Sidewall (lateral) etch narrows or widens the CD profile with depth:

```
R_lat(z) = K_lat_neu · Γ_F(z)                           [chemical isotropic]
          + K_lat_ion · Γ_F(z) · T_lat(z) · f_IE        [IAD ion-enhanced]
          − K_pass    · 0.5 · Γ_CFx(z)                  [CFx passivation]
          − K_dep_side · Γ_CFx(z)                        [polymer deposition]
```

- Chemical lateral etch is isotropic; no angular dependence.
- IAD ion-enhanced lateral uses `T_lat(z)` from the acceptance-cone model.
- CFx polymer deposited on sidewalls narrows the CD (negative contribution), and is partially removed by lateral ion bombardment.

---

### 10. Mask Aperture Evolution

The mask opening CD (`CD_mask`) evolves independently from the substrate hole due to competing erosion and deposition:

```
dCD_mask/dt = 2 · (R_lat_mask − R_poly_mask)

R_lat_mask  = K_mask_lat · K_sput · Γ_ion · Y_s · Y_mask_ratio · f_Ar   [Ar+ erosion]
            + K_F_mask · Γ_F                                              [F-radical chemical]
R_poly_mask = K_dep_poly · K_mask_poly · Γ_CFx                           [CFx narrowing]
```

- **Ar⁺ erosion** widens the mask opening; proportional to Ar fraction (Ar is the dominant sputtering species).
- **F-radical chemical etch** (`K_F_mask`) explains the non-monotonic CD_top vs CF4 behavior: at high CF4 (18 sccm), F radicals chemically widen the mask even as Ar⁺ sputtering drops. Without this term, CD_top is underpredicted by ~6% at CF4 = 18 sccm.
- **CFx polymer** narrows the mask opening.
- CD_top reported in the output equals `CD_mask` at the final time — this is what SEM measures at the mask–substrate interface.

---

### 11. Birth CD Model — IAD Collimation Effect

When the etch front advances to a new depth node, the freshly exposed surface has an initial CD determined by the ion beam footprint at that AR:

```
CD_born(z) = CD_mask · exp(−k_born · AR_birth)
```

where `AR_birth = z / CD_mask` at the moment of birth.

- `k_born = 0`: born at full mask width (no birth-taper).
- `k_born > 0`: taper increases with AR because the IAD acceptance cone narrows, so the ion beam footprint on the fresh surface is smaller than the mask opening.
- Physically represents the fact that at high AR, only ions within a narrow angular cone reach the floor and carve a narrower "birth width."

---

### 12. Profile Evolution — Explicit Euler Time-Stepping

The etch profile is advanced in time on a staggered z-grid:

```
CD(z, t+dt) = CD(z, t) − 2 · R_lat(z, t) · dt        [for existing nodes]
depth(t+dt) = depth(t) + R_v_floor(t) · dt            [floor advances]
```

- New depth nodes are spawned when the floor advances past the next grid point; their initial CD follows the birth CD model.
- Time step `dt = 0.5 s`, grid spacing `dz = 20 nm`.
- Explicit Euler is stable here because the etch rates are smooth and the CFL-equivalent condition is satisfied for typical etch speeds (~5–6 nm/s).

---

## Process Variables

| Variable | Range | Unit |
|----------|-------|------|
| CF4 flow | 0 – 30 | sccm |
| Ar flow | 0 – 30 | sccm (CF4 + Ar = 30 sccm total) |
| Bias voltage | −3000 – 0 | V |
| Source power | 100 – 3000 | W |
| Pressure | 1 – 500 | mTorr |
| Etch time | > 0 | s |

---

## Calibration Results

Parameters calibrated from 4 experimental points (CF4/Ar = 6/24, 10/20, 14/16, 18/12 sccm):

| CF4/Ar [sccm] | Depth err | CD_top err | CD_bot err | AR (sim) | AR (exp) |
|---------------|-----------|------------|------------|----------|----------|
| 6/24  | +0.7% | −0.1% | −26.6% | 6.571 | 6.519 |
| 10/20 | −2.2% | +1.4% | +23.2% | 6.543 | 6.780 |
| 14/16 | −1.1% | +1.1% | +6.7%  | 6.230 | 6.363 |
| 18/12 | +0.4% | −3.8% | −13.8% | 5.679 | 5.441 |

Calibrated parameters stored in `harc_v2_calibrated_params_physics.json`.

> **Note:** CF4/Ar = 22/8 point is excluded (low reliability — likely mask damage at high F-flux).

---

## Inverse Optimization Result

Target: maximize AR at V_bias = −1000 V, total flow = 30 sccm.

**Optimal: CF4 = 7.50 sccm / Ar = 22.50 sccm → AR = 6.603**

| Metric | Value |
|--------|-------|
| Depth | 1380 nm |
| CD_top | 209 nm |
| CD_bot | 54 nm |
| Taper index | 0.741 |
| Bowing index | 0.625 |

---

## File Structure

```
harc_etch_simulator_v2.py       Main simulator (forward sim + calibration)
run_inverse_opt.py              Inverse optimization (maximize AR)
analyze_point5.py               Outlier analysis for CF4=22 sccm condition
find_max_ar.py                  AR sweep visualization
harc_v2_calibrated_params_physics.json  Calibrated model parameters
inverse_opt_result.txt          Inverse optimization output
make_opt_figure.py              Optimization result figure
make_report_tables.py           Report tables
draw_flowchart.py               Simulator flowchart
```

---

## Requirements

```
Python 3.9+
numpy
scipy
pandas
matplotlib
```

```bash
pip install numpy scipy pandas matplotlib
```

---

## Usage

### Forward simulation
```python
from harc_etch_simulator_v2 import ProcessConditions, ModelParameters, run_forward_simulation
import json

with open("harc_v2_calibrated_params_physics.json") as f:
    params = json.load(f)

cond = ProcessConditions(
    cf4_flow=7.5, ar_flow=22.5,
    v_bias=-1000.0, source_power=250.0,
    pressure=10.0, etch_time=240.0,
    cd_initial=200.0
)
mp = ModelParameters(**params)
result = run_forward_simulation(cond, mp, verbose=True)

print(f"Depth:        {result.total_depth:.1f} nm")
print(f"Aspect Ratio: {result.aspect_ratio:.3f}")
print(f"CD_top:       {result.cd_top:.1f} nm")
print(f"CD_bot:       {result.cd_bot:.1f} nm")
```

### Inverse optimization
```bash
python run_inverse_opt.py
```

### Calibration
```bash
python harc_etch_simulator_v2.py
```

---

## Warning

> All default `ModelParameters` values are initial literature estimates, **not calibrated**.
> Quantitative predictions require calibration with real SEM data.
> Use `harc_v2_calibrated_params_physics.json` for the calibrated parameter set.
