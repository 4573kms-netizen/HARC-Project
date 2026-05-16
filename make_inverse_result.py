"""
역최적화 (Differential Evolution, ~1247 평가) + 3패널 결과 그림 + AR sweep 업데이트
"""
import sys, json, warnings, time
warnings.filterwarnings('ignore')

BASEDIR = r'C:\Users\4573k\Desktop\HARC_simulation_Claude'
sys.path.insert(0, BASEDIR)

import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import differential_evolution

from harc_etch_simulator_v2 import ProcessConditions, ModelParameters, run_forward_simulation

# ── 캘리브레이션 파라미터 로드 ─────────────────────────────────────────────
with open(f'{BASEDIR}\\harc_v2_calibrated_params_physics.json') as f:
    params = json.load(f)
mp = ModelParameters(**{k: v for k, v in params.items()
                         if k in ModelParameters.__dataclass_fields__})

VBIAS     = -1000.0
eval_cnt  = [0]

def make_cond(cf4):
    cf4 = float(np.clip(cf4, 2.0, 28.0))
    return ProcessConditions(
        cf4_flow=cf4, ar_flow=30.0 - cf4,
        v_bias=VBIAS, source_power=250.0,
        pressure=10.0, substrate_temp=15.0,
        etch_time=240.0, cd_initial=200.0,
        mask_thickness=1350.0, target_depth=2000.0,
    )

def objective(x):
    eval_cnt[0] += 1
    try:
        res = run_forward_simulation(make_cond(x[0]), mp, verbose=False)
        if eval_cnt[0] % 100 == 0:
            print(f"  [{eval_cnt[0]:4d}] CF4={x[0]:.2f}  AR={res.aspect_ratio:.3f}")
        return 1.0 - res.aspect_ratio / 10.0   # 목적함수 J = 1 - AR/10  (최소화)
    except Exception:
        return 1.0

# ── 1. Differential Evolution 최적화 ──────────────────────────────────────
# popsize=20, maxiter=60 → 약 20*(60+1)+polish ≈ 1247 평가
print("=" * 62)
print("  HARC Inverse Optimization  [Differential Evolution]")
print(f"  CF4: 2~28 sccm | Ar=30-CF4 | Vbias={VBIAS:.0f}V 고정")
print("  예상 평가 횟수: ~1247   (popsize=20 × maxiter=60 + polish)")
print("=" * 62)
t0 = time.time()

result = differential_evolution(
    objective,
    bounds=[(2.0, 28.0)],
    popsize=20,
    maxiter=60,
    tol=1e-7,
    seed=42,
    polish=True,
    disp=False,
    workers=1,
)
elapsed = time.time() - t0
print(f"\n  완료: {elapsed:.1f}s  |  평가 횟수: {eval_cnt[0]}")

# ── 2. 최적 결과 계산 ──────────────────────────────────────────────────────
CF4  = float(np.clip(result.x[0], 2.0, 28.0))
res_opt = run_forward_simulation(make_cond(CF4), mp, verbose=False)

AR_F = 30.0 - CF4
FRAC = CF4 / 30.0
D    = res_opt.total_depth
T    = res_opt.cd_top
B    = res_opt.cd_bot
M    = (T + B) / 2.0
AR   = res_opt.aspect_ratio
TAP  = res_opt.taper_index
BOW  = res_opt.bowing_index
OBJ  = 1.0 - AR / 10.0
NEV  = eval_cnt[0]
MH   = 350.0

print(f"\n  ★ 최적 결과")
print(f"  CF4={CF4:.2f} sccm / Ar={AR_F:.2f} sccm  (CF4 분율={FRAC:.3f})")
print(f"  Depth={D:.1f} nm | CD_top={T:.1f} nm | CD_mid={M:.1f} nm | CD_bot={B:.1f} nm")
print(f"  AR={AR:.3f} | Taper={TAP:.4f} | Bowing={BOW:.4f}")
print(f"  Obj. j={OBJ:.4e} | # Evaluations={NEV}")

# txt 저장
with open(f'{BASEDIR}\\inverse_opt_result.txt', 'w', encoding='utf-8') as fp:
    fp.write(f"CF4={CF4:.2f} sccm, Ar={AR_F:.2f} sccm\n"
             f"Depth={D:.1f}nm, CD_top={T:.1f}nm, CD_mid={M:.1f}nm, CD_bot={B:.1f}nm\n"
             f"AR={AR:.3f}, Taper={TAP:.4f}, Bowing={BOW:.4f}\n"
             f"Obj.j={OBJ:.4e}, Evaluations={NEV}\n")

# ── 3. 3패널 결과 그림 ────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 7))
fig.patch.set_facecolor('white')
gs  = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1.0, 1.1], wspace=0.38)

# ── 패널 1: 홀 단면 프로파일 ──────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
z    = np.linspace(0, -D, 300)
half = T/2 + (B/2 - T/2) * (np.abs(z) / D)

ax1.fill_betweenx(z, -half, half, alpha=0.30, color='#4472C4')
ax1.plot(-half, z, color='#1F4E9A', lw=2.0)
ax1.plot( half, z, color='#1F4E9A', lw=2.0)

# 마스크 (사다리꼴)
m_bot = T / 2
m_top = T / 2 * 1.9
mask_px = [-m_top, -m_bot, m_bot, m_top]
mask_py = [MH, 0, 0, MH]
ax1.fill(mask_px, mask_py, color='gray', alpha=0.38, zorder=2)
ax1.plot([-m_top, -m_bot], [MH, 0], 'gray', lw=1.5, zorder=3)
ax1.plot([ m_top,  m_bot], [MH, 0], 'gray', lw=1.5, zorder=3)

ax1.set_xlim(-230, 230)
ax1.set_ylim(-D * 1.08, MH * 1.3)
ax1.set_xlabel('x [nm]', fontsize=11)
ax1.set_ylabel('z [nm]', fontsize=11)
ax1.set_title(f'Optimal Profile\nAR={AR:.2f}', fontsize=12, fontweight='bold')
ax1.axhline(0, color='k', lw=0.6, ls='--', alpha=0.4)
ax1.grid(True, alpha=0.25)
ax1.tick_params(labelsize=9)

# CD 치수 표시
for z_frac, label, cd in [(0.0, 'CD_top', T), (0.5, 'CD_mid', M), (1.0, 'CD_bot', B)]:
    z_val = -D * z_frac
    hv    = T/2 + (B/2 - T/2) * z_frac
    ax1.annotate('', xy=(hv, z_val), xytext=(-hv, z_val),
                 arrowprops=dict(arrowstyle='<->', color='#DC2626', lw=1.2))
    ax1.text(0, z_val + D*0.03, f'{label}={cd:.0f}nm',
             ha='center', va='bottom', fontsize=7.5, color='#DC2626')

# ── 패널 2: Normalized Metrics ────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
labels = ['AR / 10', 'Taper×10', 'Bowing×10']
vals   = [AR / 10.0, TAP * 10.0, BOW * 10.0]
colors = ['#4472C4', '#ED7D31', '#ED7D31']

bars = ax2.bar(labels, vals, color=colors, edgecolor='white', linewidth=0.5, width=0.5)
for bar, v in zip(bars, vals):
    yt = bar.get_height() + 0.06 if v > 0.05 else 0.06
    ax2.text(bar.get_x() + bar.get_width() / 2, yt,
             f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2.axhline(1.0, color='#1F77B4', ls='--', lw=1.8, label='Target AR/10=1')
ax2.legend(fontsize=9, loc='upper right')
ax2.set_ylabel('Normalized value', fontsize=11)
ax2.set_title('Normalized Metrics', fontsize=12, fontweight='bold')
ax2.set_ylim(0, max(max(vals) * 1.18 + 0.3, 1.5))
ax2.grid(axis='y', alpha=0.25)
ax2.tick_params(labelsize=9)

# ── 패널 3: Recommended Recipe 표 ────────────────────────────────────────
ax3 = fig.add_subplot(gs[2])
ax3.axis('off')

rows = [
    ['CF4 flow',       f'{CF4:.2f} sccm'],
    ['Ar  flow',       f'{AR_F:.2f} sccm'],
    ['CF4 fraction',   f'{FRAC:.3f}'],
    ['V_bias',         f'{VBIAS:.1f} V'],
    ['─' * 14,         '─' * 14],
    ['Depth',          f'{D:.1f} nm'],
    ['CD_top',         f'{T:.2f} nm'],
    ['CD_mid',         f'{M:.2f} nm'],
    ['CD_bot',         f'{B:.2f} nm'],
    ['Aspect Ratio',   f'{AR:.3f}'],
    ['Taper index',    f'{TAP:.4f}'],
    ['Bowing index',   f'{BOW:.4f}'],
    ['─' * 14,         '─' * 14],
    ['Obj. j (final)', f'{OBJ:.4e}'],
    ['# Evaluations',  f'{NEV}'],
]

tbl = ax3.table(
    cellText=rows,
    colLabels=['Parameter', 'Optimal Value'],
    cellLoc='left', loc='center',
    bbox=[0.0, 0.02, 1.0, 0.96],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)

for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor('#bbbbbb')
    cell.set_linewidth(0.5)
    if r == 0:
        cell.set_facecolor('#4472C4')
        cell.set_text_props(color='white', fontweight='bold')
    elif r > 0 and rows[r-1][0].startswith('─'):
        cell.set_facecolor('#e8e8e8')
        cell.set_height(0.026)
    else:
        cell.set_facecolor('#f5f8ff' if r % 2 == 0 else 'white')

ax3.set_title('Recommended Recipe', fontsize=12, fontweight='bold', pad=16)

fig.suptitle(
    'Physics-Based Inverse Optimization Result\n'
    '[Calibrated Model  —  CF4/Ar Optimal Process Conditions]',
    fontsize=13, fontweight='bold', color='red', y=1.03
)

plt.tight_layout()
out1 = f'{BASEDIR}\\harc_optimization_result_new.png'
plt.savefig(out1, dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f'\n  저장 → {out1}')

# ── 4. AR Sweep 그림 업데이트 ─────────────────────────────────────────────
print('\nAR sweep 재계산 중 (CF4=2~28, step=2)...')
cf4_pts = list(range(2, 29, 2))
ar_pts  = []
for c in cf4_pts:
    try:
        r = run_forward_simulation(make_cond(c), mp, verbose=False)
        ar_pts.append(r.aspect_ratio)
        print(f"  CF4={c:2d}/Ar={30-c:2d}: AR={r.aspect_ratio:.3f}")
    except:
        ar_pts.append(np.nan)

EXP_CF4 = [6,    10,    14,    18]
EXP_AR  = [6.519, 6.780, 6.363, 5.441]

fig2, ax = plt.subplots(figsize=(7.5, 4.8))
fig2.patch.set_facecolor('white')

ax.plot(cf4_pts, ar_pts, 'o-', color='#2563EB', lw=2, ms=6,
        markerfacecolor='white', markeredgewidth=2, label='시뮬레이션')
ax.axvline(CF4, color='#DC2626', ls='--', lw=1.8, alpha=0.8)
ax.scatter([CF4], [AR], s=130, color='#DC2626', zorder=5)
ax.text(CF4 + 0.4, AR + 0.05,
        f'최적\nCF4={CF4:.2f} sccm\nAR={AR:.3f}',
        fontsize=8.5, color='#DC2626', fontweight='bold', va='bottom')

ax.scatter(EXP_CF4, EXP_AR, s=80, color='#374151', zorder=5,
           marker='D', label='실험값')

ax.set_xlabel('CF4 유량 [sccm]  (Ar = 30 − CF4)', fontsize=10)
ax.set_ylabel('Aspect Ratio (AR)', fontsize=10)
ax.set_title('AR vs CF4/Ar  (Vbias=−1000V, 250W, 10mTorr, 240s)',
             fontsize=10, fontweight='bold', color='#1E3A5F')
ax.legend(fontsize=9, loc='upper right')
ax.set_xlim(0, 30)
ax.set_ylim(1.5, 8.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(alpha=0.3)
ax.set_facecolor('#FAFAFA')
plt.tight_layout()

out2 = f'{BASEDIR}\\ppt_ar_sweep.png'
plt.savefig(out2, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f'  저장 → {out2}')
print('\n모두 완료!')
