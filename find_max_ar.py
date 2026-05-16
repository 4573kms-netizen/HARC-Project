"""
t=240s, V_bias=-1000V 고정 조건에서 AR을 최대화하는 CF4/Ar 비율 역계산
캘리브레이션된 파라미터(harc_v2_calibrated_params.json) 사용
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from harc_etch_simulator_v2 import (
    ModelParameters, ProcessConditions, run_forward_simulation
)

# ── 캘리브레이션된 파라미터 로드 ──────────────────────────────────────────────
with open("harc_v2_calibrated_params.json", "r") as f:
    cal_dict = json.load(f)

mp = ModelParameters()
for key, val in cal_dict.items():
    if hasattr(mp, key):
        setattr(mp, key, float(val))

# ── 고정 공정 조건 ─────────────────────────────────────────────────────────────
FIXED = dict(
    v_bias         = -1000.0,
    source_power   = 250.0,
    pressure       = 10.0,
    substrate_temp = 15.0,
    etch_time      = 240.0,
    cd_initial     = 200.0,
    mask_thickness = 1350.0,
    target_depth   = 3000.0,   # 충분히 크게 설정해 식각이 제한되지 않도록
)

def simulate_ar(cf4_flow: float) -> dict:
    """주어진 CF4 유량에서 시뮬레이션 실행 후 주요 결과 반환."""
    ar_flow = 30.0 - cf4_flow
    cond = ProcessConditions(
        cf4_flow=cf4_flow,
        ar_flow=ar_flow,
        **FIXED
    )
    try:
        res = run_forward_simulation(cond, mp, verbose=False)
        return {
            "cf4": cf4_flow,
            "ar_gas": ar_flow,
            "depth": res.total_depth,
            "cd_top": res.cd_top,
            "cd_bot": res.cd_bot,
            "AR": res.aspect_ratio,
            "taper": res.taper_index,
            "ok": True,
        }
    except Exception as e:
        return {"cf4": cf4_flow, "ar_gas": ar_flow, "AR": 0.0, "ok": False, "err": str(e)}

# ── 1. 전체 스윕 (CF4 = 1 ~ 29 sccm, step 0.5) ───────────────────────────────
print("=" * 60)
print("  AR 최대화 역계산  (t=240s, Vbias=-1000V, P=250W)")
print("=" * 60)
print(f"\n  CF4 스윕 중 (1 ~ 29 sccm, step 0.5 sccm) ...")

cf4_sweep = np.arange(1.0, 29.5, 0.5)
results = [simulate_ar(c) for c in cf4_sweep]
ok_results = [r for r in results if r["ok"]]

cf4_arr   = np.array([r["cf4"]  for r in ok_results])
ar_arr    = np.array([r["AR"]   for r in ok_results])
depth_arr = np.array([r["depth"] for r in ok_results])
cdtop_arr = np.array([r["cd_top"] for r in ok_results])
cdbot_arr = np.array([r["cd_bot"] for r in ok_results])

# ── 2. 최적점 탐색 (스윕 결과 기반 → 정밀 최적화) ────────────────────────────
best_idx  = int(np.argmax(ar_arr))
cf4_best_coarse = cf4_arr[best_idx]

# 스윕 최적 근방에서 minimize_scalar로 정밀화
def neg_ar(cf4):
    r = simulate_ar(float(cf4))
    return -r["AR"] if r["ok"] else 0.0

lo = max(cf4_best_coarse - 2.0, 1.0)
hi = min(cf4_best_coarse + 2.0, 29.0)
opt = minimize_scalar(neg_ar, bounds=(lo, hi), method='bounded',
                      options={"xatol": 0.05})

cf4_opt = opt.x
r_opt   = simulate_ar(cf4_opt)

print(f"\n{'─'*60}")
print(f"  [최적 공정 조건]")
print(f"    CF4 유량   = {r_opt['cf4']:.2f} sccm")
print(f"    Ar  유량   = {r_opt['ar_gas']:.2f} sccm")
print(f"    CF4 분율   = {r_opt['cf4']/30*100:.1f} %")
print(f"{'─'*60}")
print(f"  [최적 결과]")
print(f"    식각 깊이  = {r_opt['depth']:.1f} nm")
print(f"    CD_top     = {r_opt['cd_top']:.1f} nm")
print(f"    CD_bot     = {r_opt['cd_bot']:.1f} nm")
print(f"    AR (최대)  = {r_opt['AR']:.4f}")
print(f"    Taper index= {r_opt['taper']:.4f}")
print(f"{'─'*60}")

# ── 3. 실험 5개 조건과 비교 ───────────────────────────────────────────────────
EXP = [
    (6,  24, 1369, 210.0, 74.4),
    (10, 20, 1390, 205.0, 44.2),
    (14, 16, 1298, 204.0, 53.4),
    (18, 12, 1159, 213.0, 73.6),
    (22,  8,  616, 140.1, 65.6),
]
print(f"\n  실험 조건 AR 비교")
print(f"  {'CF4/Ar':>8}  {'깊이(nm)':>10}  {'AR_exp':>8}  {'AR_sim':>8}")
print(f"  {'─'*48}")
for cf4e, are, dep, cdt, cdb in EXP:
    r_e = simulate_ar(float(cf4e))
    ar_exp = dep / max(cdt, 1)
    flag = " ← 실험 최대" if cf4e == 10 else ""
    print(f"  {cf4e:>3}/{are:<3} sccm  {dep:>10.0f}  {ar_exp:>8.3f}  {r_e['AR']:>8.3f}{flag}")

print(f"\n  → 역계산 최적 AR = {r_opt['AR']:.4f}  (CF4={cf4_opt:.1f}/Ar={30-cf4_opt:.1f} sccm)")

# ── 4. 시각화 ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(
    f'AR 최대화 역계산  (t=240s, Vbias=−1000V, P=250W, 10mTorr)\n'
    f'최적: CF4={cf4_opt:.1f}/Ar={30-cf4_opt:.1f} sccm  →  AR={r_opt["AR"]:.3f}',
    fontsize=12, fontweight='bold'
)

ax1, ax2, ax3, ax4 = axes.flat

# (a) AR vs CF4
ax1.plot(cf4_arr, ar_arr, 'b-', lw=2, label='시뮬레이션')
ax1.axvline(cf4_opt, color='red', ls='--', lw=1.5, label=f'최적 CF4={cf4_opt:.1f} sccm')
ax1.scatter([r[0] for r in EXP],
            [r[2]/max(r[3],1) for r in EXP],
            color='k', s=60, zorder=5, label='실험 AR')
ax1.set_xlabel('CF4 유량 [sccm]'); ax1.set_ylabel('Aspect Ratio')
ax1.set_title('(a) AR vs CF4/Ar 비율')
ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
ax1.set_xlim(0, 30)

# (b) 식각 깊이 vs CF4
ax2.plot(cf4_arr, depth_arr, 'g-', lw=2)
ax2.axvline(cf4_opt, color='red', ls='--', lw=1.5)
ax2.scatter([r[0] for r in EXP], [r[2] for r in EXP],
            color='k', s=60, zorder=5, label='실험 깊이')
ax2.set_xlabel('CF4 유량 [sccm]'); ax2.set_ylabel('식각 깊이 [nm]')
ax2.set_title('(b) 식각 깊이 vs CF4/Ar 비율')
ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
ax2.set_xlim(0, 30)

# (c) CD_top, CD_bot vs CF4
ax3.plot(cf4_arr, cdtop_arr, 'b-', lw=2, label='CD_top (sim)')
ax3.plot(cf4_arr, cdbot_arr, 'r-', lw=2, label='CD_bot (sim)')
ax3.axvline(cf4_opt, color='gray', ls='--', lw=1.5)
ax3.scatter([r[0] for r in EXP], [r[3] for r in EXP],
            color='b', s=50, marker='D', zorder=5, label='CD_top exp')
ax3.scatter([r[0] for r in EXP], [r[4] for r in EXP],
            color='r', s=50, marker='D', zorder=5, label='CD_bot exp')
ax3.set_xlabel('CF4 유량 [sccm]'); ax3.set_ylabel('CD [nm]')
ax3.set_title('(c) CD_top / CD_bot vs CF4/Ar 비율')
ax3.legend(fontsize=8); ax3.grid(alpha=0.3)
ax3.set_xlim(0, 30)

# (d) 최적 조건의 홀 프로파일
r_best_full = simulate_ar(cf4_opt)
cond_best = ProcessConditions(cf4_flow=cf4_opt, ar_flow=30-cf4_opt, **FIXED)
res_best  = run_forward_simulation(cond_best, mp, verbose=False)
z_b  = res_best.z_grid
cd_b = res_best.cd_profile
ax4.plot(-cd_b/2, -z_b, 'b-', lw=2)
ax4.plot( cd_b/2, -z_b, 'b-', lw=2)
ax4.fill_betweenx(-z_b, -cd_b/2, cd_b/2, alpha=0.15, color='blue')
ax4.axhline(-res_best.total_depth, color='red', ls='--', lw=1.2,
            label=f'깊이={res_best.total_depth:.0f} nm')
ax4.set_xlabel('x [nm]'); ax4.set_ylabel('깊이 [nm]')
ax4.set_title(f'(d) 최적 홀 단면 프로파일\n'
              f'CF4={cf4_opt:.1f}/Ar={30-cf4_opt:.1f} sccm  AR={res_best.aspect_ratio:.3f}')
ax4.legend(fontsize=8); ax4.grid(alpha=0.3)
ax4.set_facecolor('#F8FAFC')

plt.tight_layout()
plt.savefig('max_ar_result.png', dpi=150, bbox_inches='tight')
print(f"\n  그래프 저장 → max_ar_result.png")
plt.show()
