# HARC Etch Physics-Based Simulator

Physics-based forward simulation and inverse optimization for High Aspect Ratio Contact (HARC) etching using CF4/Ar plasma.

## Overview

This simulator models the HARC etch process from plasma generation to feature-scale profile evolution. It supports forward simulation, inverse optimization, parameter calibration from experimental data, and process window visualization.

## Physical Models

| Model | Description |
|-------|-------------|
| 0-D Global Plasma | F-radical, CFx, and ion flux estimation from CF4 fraction, source power, and pressure |
| Sheath / Ion Energy | Mean ion energy via simplified Child-Langmuir sheath approximation |
| Feature-Scale Transport | Ion (Clausing + exponential) and neutral (Knudsen) transmission vs. depth |
| Surface Reaction | Chemical + ion-enhanced + sputtering − passivation etch rate model |
| Bohdansky Sputtering | Sputtering yield for Si by Ar⁺/CF⁺ ions |
| Profile Evolution | Explicit Euler time-stepping on staggered z-grid |

## Process Variables

| Variable | Range | Unit |
|----------|-------|------|
| CF4 flow | 0 – 30 | sccm |
| Ar flow | 0 – 30 | sccm (CF4 + Ar = 30 sccm) |
| Bias voltage | −3000 – 0 | V |
| Source power | 100 – 3000 | W |
| Pressure | 1 – 500 | mTorr |
| Etch time | > 0 | s |

## Requirements

```
Python 3.9+
numpy
scipy
pandas
matplotlib
```

Install dependencies:
```bash
pip install numpy scipy pandas matplotlib
```

## Usage

### Run demo
```bash
python harc_etch_simulator.py
```

### Forward simulation
```python
from harc_etch_simulator import ProcessConditions, ModelParameters, run_forward_simulation

cond = ProcessConditions(
    cf4_flow=6.0, ar_flow=24.0,
    v_bias=-1000.0, source_power=250.0,
    pressure=10.0, etch_time=300.0,
    cd_initial=200.0
)
mp = ModelParameters()
result = run_forward_simulation(cond, mp, verbose=True)

print(f"Depth: {result.total_depth:.1f} nm")
print(f"Aspect Ratio: {result.aspect_ratio:.2f}")
```

### Inverse optimization (find recipe for target AR)
```python
from harc_etch_simulator import optimize_process_conditions

best_cond, best_result, opt_info = optimize_process_conditions(
    mp=mp, base_cond=cond, target_AR=10.0
)
```

### Calibration with experimental data
```python
import pandas as pd
from harc_etch_simulator import calibrate_model_parameters

exp_data = pd.DataFrame({
    'cf4_flow':    [...],
    'ar_flow':     [...],
    'v_bias':      [...],
    'source_power':[...],
    'pressure':    [...],
    'etch_time':   [...],
    'cd_initial':  [...],
    'depth_meas':  [...],
    'cd_top_meas': [...],
    'cd_bot_meas': [...],
})
mp_calibrated = calibrate_model_parameters(exp_data, mp)
```

## Output Metrics

- `total_depth` : Etch depth [nm]
- `cd_top / cd_mid / cd_bot` : Critical dimension at top / middle / bottom [nm]
- `aspect_ratio` : depth / cd_bot
- `taper_index` : (cd_top − cd_bot) / cd_top — positive = tapered
- `bowing_index` : (cd_max − cd_top) / cd_top — positive = bowing

## Warning

> All default model parameters are initial literature estimates, **not calibrated values**.
> Quantitative predictions require calibration with real experimental SEM data (~10 runs minimum).
>
> Calibration priority: `K_chem`, `K_ie`, `K_sput` → `A_F`, `A_ion` → `lambda_ion`, `lambda_neutral` → `alpha_E` → `K_pass`
