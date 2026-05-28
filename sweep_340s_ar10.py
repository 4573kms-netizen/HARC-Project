"""
t=340s, calibrated params 고정, CF4/Ar sweep → AR=10에 가장 가까운 조건 찾기 + Figure 5 스타일 프로파일
"""
import json, dataclasses, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_HERE = os.path.dirname(os.path.abspath(__file__))
import sys; sys.path.insert(0, _HERE)

from harc_etch_simulator_v2 import (
    ModelParameters, ProcessConditions, run_forward_simulation, COLORS
)

# ── 1. calibrated params 로드 ──────────────────────────────────────────────
with open(os.path.join(_HERE, 'harc_v2_calibrated_params_physics.json')) as f:
    cal = json.load(f)
mp_cal = ModelParameters(**{k: v for k, v in cal.items()
                             if k in {f.name for f in dataclasses.fields(ModelParameters)}})

TOTAL_FLOW = 30.0   # sccm (CF4 + Ar = 30 고정)
ETCH_TIME  = 340.0  # s
TARGET_AR  = 10.0

BASE_COND = dict(
    v_bias         = -1000.0,
    source_power   = 250.0,
    pressure       = 10.0,
    substrate_temp = 15.0,
    cd_initial     = 200.0,
    mask_thickness = 1350.0,
    target_depth   = 1400.0,
    etch_time      = ETCH_TIME,
)

# ── 2. CF4 sweep 4~26 sccm (step 1) ───────────────────────────────────────
cf4_vals = np.arange(4, 27, 1, dtype=float)
results  = []

print(f"{'CF4':>5} {'Ar':>5} | {'Depth':>8} {'CD_top':>8} {'CD_bot':>8} {'AR':>7}")
print("-" * 52)
for cf4 in cf4_vals:
    ar_flow = TOTAL_FLOW - cf4
    cond = ProcessConditions(cf4_flow=cf4, ar_flow=ar_flow, **BASE_COND)
    try:
        r = run_forward_simulation(cond, mp_cal, verbose=False)
        results.append((cf4, ar_flow, r))
        print(f"{cf4:5.1f} {ar_flow:5.1f} | "
              f"{r.total_depth:8.1f} {r.cd_top:8.1f} {r.cd_bot:8.1f} {r.aspect_ratio:7.3f}")
    except Exception as e:
        print(f"{cf4:5.1f} {ar_flow:5.1f} | ERROR: {e}")
        results.append((cf4, ar_flow, None))

# ── 3. AR=10에 가장 가까운 조건 찾기 ─────────────────────────────────────
valid = [(cf4, ar, r) for cf4, ar, r in results if r is not None]
best  = min(valid, key=lambda x: abs(x[2].aspect_ratio - TARGET_AR))
b_cf4, b_ar, b_res = best

print(f"\n★ AR=10 최근접 조건: CF4={b_cf4:.0f} / Ar={b_ar:.0f} sccm")
print(f"  Depth  = {b_res.total_depth:.1f} nm")
print(f"  CD_top = {b_res.cd_top:.1f} nm")
print(f"  CD_bot = {b_res.cd_bot:.1f} nm")
print(f"  AR     = {b_res.aspect_ratio:.3f}")

# ── 4. Figure 5 스타일: 5개 대표 조건 프로파일 ───────────────────────────
# 캘리브레이션 4개 원본 조건 + 최적 조건 (t=340s)
plot_cf4 = [6.0, 10.0, 14.0, 18.0, b_cf4]
plot_labels = [
    f"CF4={int(c)}/Ar={int(TOTAL_FLOW-c)}" + (" ★AR≈10" if c == b_cf4 else "")
    for c in plot_cf4
]

fig, axes = plt.subplots(1, len(plot_cf4), figsize=(4 * len(plot_cf4), 8))
fig.suptitle(
    f'HARC v2 — t=340 s  (calibrated params fixed)\n'
    f'Source 250 W, 10 mTorr, Vbias −1000 V, T=15 °C',
    fontsize=13, fontweight='bold'
)

for ax, cf4, label in zip(axes, plot_cf4, plot_labels):
    ar_flow = TOTAL_FLOW - cf4
    cond = ProcessConditions(cf4_flow=cf4, ar_flow=ar_flow, **BASE_COND)
    try:
        r = run_forward_simulation(cond, mp_cal, verbose=False)
    except Exception as e:
        ax.set_title(f"{label}\nERROR: {e}", fontsize=8)
        continue

    z  = r.z_grid
    cd = r.cd_profile
    is_best = (cf4 == b_cf4)
    color   = COLORS['secondary'] if is_best else COLORS['primary']

    ax.plot(-cd/2, -z, color=color, lw=2.2)
    ax.plot( cd/2, -z, color=color, lw=2.2)
    ax.fill_betweenx(-z, -cd/2, cd/2, alpha=0.13, color=color)

    # 마스크
    mask_cd = r.cd_top
    ax.fill_betweenx([0, cond.mask_thickness * 0.12],
                     [-mask_cd/2, -mask_cd/2], [mask_cd/2, mask_cd/2],
                     alpha=0.25, color='gray')

    ax.set_title(
        f"{label}\n"
        f"Depth={r.total_depth:.0f} nm\n"
        f"CD_top={r.cd_top:.0f}  CD_bot={r.cd_bot:.0f} nm\n"
        f"AR = {r.aspect_ratio:.3f}" + (" ← Target" if is_best else ""),
        fontsize=8, fontweight='bold',
        color=COLORS['secondary'] if is_best else 'black'
    )
    ax.set_xlabel('x [nm]', fontsize=8)
    ax.set_ylabel('Depth [nm]', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)
    ax.set_facecolor(COLORS['bg'])

    # AR 텍스트 박스
    ax.text(0, -r.total_depth * 0.5,
            f"AR={r.aspect_ratio:.2f}",
            ha='center', va='center', fontsize=10, fontweight='bold',
            color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.85))

plt.tight_layout()
os.makedirs(os.path.join(_HERE, 'figures'), exist_ok=True)
save_path = os.path.join(_HERE, 'figures', 'harc_340s_ar10_profiles.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\n  그림 저장 → {save_path}")
plt.show()

# ── 5. AR vs CF4 곡선 ─────────────────────────────────────────────────────
cf4_plot = [x[0] for x in valid]
ar_plot  = [x[2].aspect_ratio for x in valid]

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(cf4_plot, ar_plot, 'o-', color=COLORS['primary'], lw=2, ms=6, label='t=340 s (sim)')
ax2.axhline(TARGET_AR, color=COLORS['secondary'], lw=1.5, ls='--', label='AR = 10 target')
ax2.axvline(b_cf4, color=COLORS['accent'], lw=1.5, ls=':', label=f'Best: CF4={b_cf4:.0f} sccm')
ax2.scatter([b_cf4], [b_res.aspect_ratio], s=120, color=COLORS['secondary'], zorder=5)
ax2.annotate(f"CF4={b_cf4:.0f}/Ar={b_ar:.0f}\nAR={b_res.aspect_ratio:.3f}",
             xy=(b_cf4, b_res.aspect_ratio), xytext=(b_cf4+1.5, b_res.aspect_ratio+0.3),
             fontsize=9, color=COLORS['secondary'], fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=COLORS['secondary']))
ax2.set_xlabel('CF4 flow [sccm]  (Ar = 30 − CF4)', fontsize=11)
ax2.set_ylabel('Aspect Ratio', fontsize=11)
ax2.set_title(f't=340 s, calibrated params — AR vs CF4/Ar sweep', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_facecolor(COLORS['bg'])
plt.tight_layout()
save2 = os.path.join(_HERE, 'figures', 'harc_340s_ar_sweep.png')
plt.savefig(save2, dpi=150, bbox_inches='tight')
print(f"  AR 곡선 저장 → {save2}")
plt.show()
