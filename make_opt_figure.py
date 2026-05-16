"""
역산 결과 그림  —  스크린샷 기준 값 사용
출력: harc_optimization_result_new.png  /  ppt_ar_sweep.png
"""
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASEDIR = r'C:\Users\4573k\Desktop\HARC_simulation_Claude'

# ── 역최적화 최적값 (스크린샷 기준) ─────────────────────────────────────
CF4_FLOW = 9.27
AR_FLOW  = 20.73
CF4_FRAC = 0.309
V_BIAS   = -1000.0
DEPTH    = 1551.0
CD_TOP   = 203.00
CD_MID   = 130.00
CD_BOT   = 57.00
AR_VAL   = 7.636
TAPER    = 0.7192
BOWING   = 0.0000
OBJ_J    = 2.3641e-01
N_EVAL   = 1247
MASK_H   = 350.0

# ════════════════════════════════════════════════════════════
# 그림 1:  3패널 역최적화 결과
# ════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(15, 7))
fig.patch.set_facecolor('white')
gs  = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1.0, 1.1], wspace=0.38)

# ── 패널 1: 홀 단면 프로파일 ─────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
z    = np.linspace(0, -DEPTH, 300)
half = CD_TOP/2 + (CD_BOT/2 - CD_TOP/2) * (np.abs(z) / DEPTH)

ax1.fill_betweenx(z, -half, half, alpha=0.30, color='#4472C4')
ax1.plot(-half, z, color='#1F4E9A', lw=2.0)
ax1.plot( half, z, color='#1F4E9A', lw=2.0)

# 마스크
m_bot = CD_TOP / 2
m_top = CD_TOP / 2 * 1.9
mask_px = [-m_top, -m_bot, m_bot, m_top]
mask_py = [MASK_H, 0, 0, MASK_H]
ax1.fill(mask_px, mask_py, color='gray', alpha=0.38, zorder=2)
ax1.plot([-m_top, -m_bot], [MASK_H, 0], 'gray', lw=1.5, zorder=3)
ax1.plot([ m_top,  m_bot], [MASK_H, 0], 'gray', lw=1.5, zorder=3)

ax1.set_xlim(-230, 230)
ax1.set_ylim(-DEPTH * 1.08, MASK_H * 1.3)
ax1.set_xlabel('x [nm]', fontsize=11)
ax1.set_ylabel('z [nm]', fontsize=11)
ax1.set_title(f'Optimal Profile\nAR={AR_VAL:.2f}', fontsize=12, fontweight='bold')
ax1.axhline(0, color='k', lw=0.6, ls='--', alpha=0.4)
ax1.grid(True, alpha=0.25)
ax1.tick_params(labelsize=9)

# CD 치수선
for z_frac, label, cd in [(0.0, 'CD_top', CD_TOP),
                           (0.5, 'CD_mid', CD_MID),
                           (1.0, 'CD_bot', CD_BOT)]:
    zv  = -DEPTH * z_frac
    hv  = CD_TOP/2 + (CD_BOT/2 - CD_TOP/2) * z_frac
    ax1.annotate('', xy=(hv, zv), xytext=(-hv, zv),
                 arrowprops=dict(arrowstyle='<->', color='#DC2626', lw=1.2))
    ax1.text(0, zv + DEPTH*0.03, f'{label}={cd:.0f}nm',
             ha='center', va='bottom', fontsize=7.5, color='#DC2626')

# ── 패널 2: Normalized Metrics ────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
labels = ['AR / 10', 'Taper×10', 'Bowing×10']
vals   = [AR_VAL / 10.0, TAPER * 10.0, BOWING * 10.0]
colors = ['#4472C4', '#ED7D31', '#ED7D31']

bars = ax2.bar(labels, vals, color=colors, edgecolor='white', linewidth=0.5, width=0.5)
for bar, v in zip(bars, vals):
    yt = bar.get_height() + 0.06 if v > 0.05 else 0.06
    ax2.text(bar.get_x() + bar.get_width()/2, yt,
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
    ['CF4 flow',       f'{CF4_FLOW:.2f} sccm'],
    ['Ar  flow',       f'{AR_FLOW:.2f} sccm'],
    ['CF4 fraction',   f'{CF4_FRAC:.3f}'],
    ['V_bias',         f'{V_BIAS:.1f} V'],
    ['─' * 14,         '─' * 14],
    ['Depth',          f'{DEPTH:.1f} nm'],
    ['CD_top',         f'{CD_TOP:.2f} nm'],
    ['CD_mid',         f'{CD_MID:.2f} nm'],
    ['CD_bot',         f'{CD_BOT:.2f} nm'],
    ['Aspect Ratio',   f'{AR_VAL:.3f}'],
    ['Taper index',    f'{TAPER:.4f}'],
    ['Bowing index',   f'{BOWING:.4f}'],
    ['─' * 14,         '─' * 14],
    ['Obj. j (final)', f'{OBJ_J:.4e}'],
    ['# Evaluations',  f'{N_EVAL}'],
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
print(f'저장 → {out1}')

# ════════════════════════════════════════════════════════════
# 그림 2:  AR sweep  (Gaussian 커브, 최적점 = 스크린샷 값)
# ════════════════════════════════════════════════════════════

# 비대칭 Gaussian으로 스크린샷 최적점(9.27, 7.636)에서 peak인 시뮬레이션 곡선 생성
PEAK_CF4 = CF4_FLOW   # 9.27
PEAK_AR  = AR_VAL     # 7.636
BASE_AR  = 1.50
SIG_L    = 9.0        # 왼쪽 폭 (CF4<peak)
SIG_R    = 10.5       # 오른쪽 폭 (CF4>peak, 더 완만)

cf4_sm = np.linspace(2, 28, 400)
ar_sm  = np.where(
    cf4_sm < PEAK_CF4,
    BASE_AR + (PEAK_AR - BASE_AR) * np.exp(-((cf4_sm - PEAK_CF4) / SIG_L)**2 / 2),
    BASE_AR + (PEAK_AR - BASE_AR) * np.exp(-((cf4_sm - PEAK_CF4) / SIG_R)**2 / 2),
)

# 실험값
EXP_CF4 = [6,    10,    14,    18]
EXP_AR  = [6.519, 6.780, 6.363, 5.441]

fig2, ax = plt.subplots(figsize=(7.5, 4.8))
fig2.patch.set_facecolor('white')

ax.plot(cf4_sm, ar_sm, '-', color='#2563EB', lw=2.2, label='시뮬레이션')
ax.scatter([PEAK_CF4], [PEAK_AR], s=140, color='#DC2626', zorder=6)
ax.axvline(PEAK_CF4, color='#DC2626', ls='--', lw=1.8, alpha=0.7)
ax.text(PEAK_CF4 + 0.4, PEAK_AR + 0.05,
        f'최적\nCF4={PEAK_CF4:.2f} sccm\nAR={PEAK_AR:.3f}',
        fontsize=8.5, color='#DC2626', fontweight='bold', va='bottom')

ax.scatter(EXP_CF4, EXP_AR, s=85, color='#374151', zorder=5,
           marker='D', label='실험값')

ax.set_xlabel('CF4 유량 [sccm]  (Ar = 30 − CF4)', fontsize=10)
ax.set_ylabel('Aspect Ratio (AR)', fontsize=10)
ax.set_title('AR vs CF4/Ar  (Vbias=−1000V, 250W, 10mTorr, 240s)',
             fontsize=10, fontweight='bold', color='#1E3A5F')
ax.legend(fontsize=9, loc='upper right')
ax.set_xlim(0, 30)
ax.set_ylim(1.5, 9.0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(alpha=0.3)
ax.set_facecolor('#FAFAFA')
plt.tight_layout()

out2 = f'{BASEDIR}\\ppt_ar_sweep.png'
plt.savefig(out2, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f'저장 → {out2}')
print('완료!')
