# Regenerate all 5 figures using calibrated JSON parameters (skip re-calibration).
import sys, os, json
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from harc_etch_simulator_v2 import (
    ModelParameters, ProcessConditions,
    EXPERIMENTAL_DATA, _build_experiments,
    run_forward_simulation,
    plot_calibration_comparison,
    plot_profiles,
    run_rodeo_validation,
    plot_optimization_result,
    _grid_search_ar10,
    print_accuracy_table,
)
from dataclasses import asdict

os.makedirs(os.path.join(_HERE, 'figures'), exist_ok=True)

# ── Load calibrated params ────────────────────────────────────────────────────
json_path = os.path.join(_HERE, 'harc_v2_calibrated_params_physics.json')
mp = ModelParameters()
if os.path.exists(json_path):
    with open(json_path) as f:
        d = json.load(f)
    for k, v in d.items():
        if hasattr(mp, k):
            setattr(mp, k, v)
    print(f"Loaded {json_path}")
else:
    print(f"[WARN] {json_path} not found — using defaults")

exp_data = EXPERIMENTAL_DATA.copy()

# ── Fig 1: Calibration comparison (pre = post = same JSON params here) ────────
print("\n[Fig 1] Calibration comparison …")
plot_calibration_comparison(
    exp_data, mp, mp,
    save_path=os.path.join(_HERE, 'figures', 'harc_v2_calibration.png')
)

# ── Fig 2: Calibrated hole profiles ───────────────────────────────────────────
print("\n[Fig 2] Hole profiles …")
plot_profiles(exp_data, mp, save_path=os.path.join(_HERE, 'figures', 'harc_v2_profiles.png'))

# ── Fig 3: RODEo external validation ──────────────────────────────────────────
print("\n[Fig 3] RODEo validation …")
run_rodeo_validation(mp)

# ── Fig 4: STEP 8 – CF4=9/Ar=21, t=340s ──────────────────────────────────────
print("\n[Fig 4] CF4=9 / Ar=21 / t=340s optimization profile …")
cond_s9 = ProcessConditions(
    cf4_flow=9.0, ar_flow=21.0,
    v_bias=-1000.0, source_power=250.0, pressure=10.0,
    substrate_temp=15.0, etch_time=340.0,
    cd_initial=200.0, mask_thickness=1350.0, target_depth=2500.0,
)
res_s9 = run_forward_simulation(cond_s9, mp, verbose=True)
print(f"  AR={res_s9.aspect_ratio:.3f}  Bowing={res_s9.bowing_index:.4f}"
      f"  CD_top={res_s9.cd_top:.1f}  CD_bot={res_s9.cd_bot:.1f}")
plot_optimization_result(
    res_s9, 9.0,
    title='Physics-Based Optimization Result\n'
          '[Calibrated Model  →  CF4/Ar Optimal Process Conditions]',
    show_time=False,
    save_path=os.path.join(_HERE, 'figures', 'harc_v2_ar10_extended.png'),
)

# ── Fig 5: STEP 9 – Full-space AR=10 grid search ─────────────────────────────
print("\n[Fig 5] Full-space AR=10 grid search …")
cf4_s10, res_s10 = _grid_search_ar10(
    mp, target_ar=10.0,
    cf4_vals=list(np.linspace(6.0, 18.0, 7)),   # calibrated CF4 range only
    vbias_vals=[-750.0, -1000.0, -1250.0, -1500.0],
    time_vals=[240.0, 360.0, 480.0, 600.0],
    max_bowing=0.65,
    max_taper=0.85,
    min_cd_bot=30.0,
)
if res_s10 is not None:
    print(f"  Best: CF4={cf4_s10:.2f}  V_bias={res_s10.conditions.v_bias:.0f}V"
          f"  t={res_s10.conditions.etch_time:.0f}s  AR={res_s10.aspect_ratio:.3f}"
          f"  Bowing={res_s10.bowing_index:.4f}")
    plot_optimization_result(
        res_s10, cf4_s10,
        title='Physics-Based Optimization Result\n'
              '[Full Parameter Space (CF4/Ar, V_bias, Etch Time)  ->  AR ~ 10]',
        show_time=True,
        save_path=os.path.join(_HERE, 'figures', 'harc_v2_ar10_full_opt.png'),
    )
else:
    print("  [WARN] No valid result within constraints.")

print("\nDone - all figures regenerated in ./figures/")
