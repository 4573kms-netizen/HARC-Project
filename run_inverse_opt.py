# -*- coding: utf-8 -*-
"""
역최적화: V_bias=-1000V, 식각 시간 240s 고정, AR 최대화 공정 조건 탐색
최적화 변수: CF4 유량만 (Ar = 30 - CF4)
방법: 15점 그리드 서치 → Nelder-Mead 로컬 정밀화
"""
import sys, json, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, r'C:\Users\4573k\Desktop\HARC_simulation_Claude')

import numpy as np
from scipy.optimize import minimize
from harc_etch_simulator_v2 import ProcessConditions, ModelParameters, run_forward_simulation

with open(r'C:\Users\4573k\Desktop\HARC_simulation_Claude\harc_v2_calibrated_params_physics.json') as f:
    params = json.load(f)
mp = ModelParameters(**{k: v for k, v in params.items()
                         if k in ModelParameters.__dataclass_fields__})

VBIAS_FIXED = -1000.0  # 실험 조건 고정

def make_cond(cf4):
    cf4 = float(np.clip(cf4, 2.0, 28.0))
    return ProcessConditions(
        cf4_flow=cf4, ar_flow=30.0 - cf4, v_bias=VBIAS_FIXED,
        source_power=250.0, pressure=10.0, substrate_temp=15.0,
        etch_time=240.0, cd_initial=200.0,
        mask_thickness=1350.0, target_depth=2000.0,
    )

def objective(x):
    try:
        res = run_forward_simulation(make_cond(x[0]), mp, verbose=False)
        return -res.aspect_ratio  # 최대화 → 부호 반전
    except Exception:
        return 1e6

# ── 1단계: 그리드 서치 (CF4 = 2 ~ 28, step 2) ──────────────
print("Step 1: 그리드 서치 (CF4 2~28 sccm, step=2) ...")
cf4_grid = list(range(2, 29, 2))

best_ar, best_cf4 = -1.0, 8.0
t0 = time.time()
for cf4 in cf4_grid:
    try:
        res = run_forward_simulation(make_cond(cf4), mp, verbose=False)
        print(f"  CF4={cf4:2d}/Ar={30-cf4:2d}: AR={res.aspect_ratio:.3f}  "
              f"depth={res.total_depth:.0f}nm  CD_top={res.cd_top:.1f}nm  CD_bot={res.cd_bot:.1f}nm")
        if res.aspect_ratio > best_ar:
            best_ar, best_cf4 = res.aspect_ratio, float(cf4)
    except Exception as e:
        print(f"  CF4={cf4} -> ERROR: {e}")

print(f"\n그리드 최적: CF4={best_cf4:.1f}  AR={best_ar:.3f}  ({time.time()-t0:.0f}s)")

# ── 2단계: Nelder-Mead 로컬 정밀화 ───────────────────────────
print("\nStep 2: Nelder-Mead 정밀화 ...")
t1 = time.time()
opt = minimize(objective, x0=[best_cf4], method='Nelder-Mead',
               options={'maxiter': 200, 'xatol': 0.1, 'fatol': 1e-4})
cf4_best = float(np.clip(opt.x[0], 2.0, 28.0))
print(f"정밀화 완료 ({time.time()-t1:.0f}s)")

# ── 최종 결과 ─────────────────────────────────────────────────
res_best = run_forward_simulation(make_cond(cf4_best), mp, verbose=False)

lines = [
    "=" * 60,
    "  최적 공정 조건  (V_bias=-1000V, t=240s 고정)",
    "=" * 60,
    f"  CF4 유량  : {cf4_best:.2f} sccm",
    f"  Ar  유량  : {30.0 - cf4_best:.2f} sccm",
    f"  CF4 분율  : {cf4_best/30.0:.3f}",
    f"  바이어스  : {VBIAS_FIXED:.0f} V  (고정)",
    f"  소스 파워 : 250 W  (고정)",
    f"  압력      : 10 mTorr  (고정)",
    f"  식각 시간 : 240 s = 4 min  (고정)",
    "",
    f"  예측 식각 깊이  : {res_best.total_depth:.1f} nm",
    f"  예측 Top CD     : {res_best.cd_top:.1f} nm",
    f"  예측 Bot CD     : {res_best.cd_bot:.1f} nm",
    f"  예측 종횡비 AR  : {res_best.aspect_ratio:.3f}",
    f"  예측 Taper      : {res_best.taper_index:.4f}",
    f"  예측 Bowing     : {res_best.bowing_index:.4f}",
    "=" * 60,
]
print("\n" + "\n".join(lines))
with open(r'C:\Users\4573k\Desktop\HARC_simulation_Claude\inverse_opt_result.txt', 'w', encoding='utf-8') as _f:
    _f.write("\n".join(lines) + "\n")
print("-> 저장: inverse_opt_result.txt")
