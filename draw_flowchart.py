"""
HARC Flowchart - 화살표 제거, 역최적화 제거, 겹침 없음
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib
matplotlib.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

W, H = 13.33, 7.5
fig = plt.figure(figsize=(W * 1.6, H * 1.6))
ax  = fig.add_subplot(111)
ax.set_xlim(0, W)
ax.set_ylim(H, 0)   # y=0 위, y=H 아래
ax.axis('off')
fig.patch.set_facecolor('white')

def rgb(r,g,b): return (r/255, g/255, b/255)
C = {
    's1':  rgb(236,236,236),
    's2':  rgb(212,212,212),
    's3':  rgb(180,180,180),
    's4':  rgb(148,148,148),
    's5':  rgb(115,115,115),
    's6':  rgb(225,225,225),
    'ann': rgb(250,250,250),
    'circ':rgb(46,46,46),
    'bdr': rgb(80,80,80),
    'dark':rgb(16,16,16),
    'lite':rgb(245,245,245),
    'loop':rgb(105,105,105),
    'abdr':rgb(175,175,175),
}

cx   = 4.1    # 메인 흐름 중심 x
bw   = 6.4    # 박스 너비
bh   = 0.62   # 박스 기본 높이
bh5  = 0.74   # STEP5 높이 (한 줄 더)
rc   = 0.27   # START/END 원 반지름
axc  = 11.15  # 어노테이션 중심 x
aw   = 4.05   # 어노테이션 너비

# ── Y 위치 (겹침 없도록 엄밀히 계산) ─────────────────────
# START   : center=0.55, top=0.28, bottom=0.82
# STEP1   : top=0.94,  center=1.25, bottom=1.56
# STEP2   : top=1.68,  center=1.99, bottom=2.30
# STEP3   : top=2.42,  center=2.73, bottom=3.04
# STEP4   : top=3.16,  center=3.47, bottom=3.78
# STEP5   : top=3.90,  center=4.27, bottom=4.64  (h=0.74)
# STEP6   : top=4.76,  center=5.07, bottom=5.38
# END     : top=5.50,  center=5.77, bottom=6.04
ys = 0.55
y1 = 1.25
y2 = 1.99
y3 = 2.73
y4 = 3.47
y5 = 4.27
y6 = 5.07
ye = 5.77

# ── 헬퍼 함수 ─────────────────────────────────────────────
def rbox(x, y, w, h, title, sub, bg, tc, ft=11.5, fs=9.5):
    b = FancyBboxPatch((x-w/2, y-h/2), w, h,
                       boxstyle="round,pad=0.06",
                       fc=bg, ec=C['bdr'], lw=1.6, zorder=3)
    ax.add_patch(b)
    ax.text(x, y - h*0.13, title,
            ha='center', va='center', fontsize=ft,
            color=tc, fontweight='bold', zorder=4)
    ax.text(x, y + h*0.27, sub,
            ha='center', va='center', fontsize=fs,
            color=tc, zorder=4)

def oval(x, y, r, bg, text, tc, fs=12):
    c = plt.Circle((x,y), r, fc=bg, ec=C['bdr'], lw=2.0, zorder=3)
    ax.add_patch(c)
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fs, color=tc, fontweight='bold', zorder=4)

def annbox(x, y, w, h, lines, fs=9):
    b = FancyBboxPatch((x-w/2, y-h/2), w, h,
                       boxstyle="round,pad=0.05",
                       fc=C['ann'], ec=C['abdr'], lw=1.0,
                       linestyle='dashed', zorder=3)
    ax.add_patch(b)
    txt = '\n'.join(lines)
    ax.text(x-w/2+0.13, y, txt,
            ha='left', va='center', fontsize=fs,
            color=C['dark'], linespacing=1.6, zorder=4)

# ══════════════════════════════════════════════════════════
# 제목
# ══════════════════════════════════════════════════════════
ax.text(W/2, 0.17,
        'HARC Etch Simulator  —  Simulation Flow',
        ha='center', va='center', fontsize=15,
        fontweight='bold', color=C['dark'])

# ══════════════════════════════════════════════════════════
# 메인 박스
# ══════════════════════════════════════════════════════════
oval(cx, ys, rc, C['circ'], 'START', C['lite'])

rbox(cx, y1, bw, bh,
     'STEP 1  |  공정 조건 입력',
     'CF4/Ar 유량  ·  V_bias  ·  Source Power  ·  Pressure  ·  식각 시간',
     C['s1'], C['dark'])

rbox(cx, y2, bw, bh,
     'STEP 2  |  0-D 플라즈마 모델',
     '표면 플럭스 계산  (Γ_F, Γ_CFx, Γ_ion)  +  평균 이온 에너지',
     C['s2'], C['dark'])

rbox(cx, y3, bw, bh,
     'STEP 3  |  수송 모델',
     '플럭스가 홀 내부 깊이에 따라 얼마나 감쇠하는지 계산',
     C['s3'], C['dark'])

rbox(cx, y4, bw, bh,
     'STEP 4  |  표면 반응 모델',
     '수직 식각 속도  ·  측면 식각 속도  ·  스퍼터링 계산',
     C['s4'], C['lite'])

rbox(cx, y5, bw, bh5,
     'STEP 5  |  프로파일 시간 적분  (Δt 루프)',
     '깊이 증가  ·  CD 프로파일  ·  마스크 개구부  ·  폴리머 업데이트',
     C['s5'], C['lite'])

rbox(cx, y6, bw, bh,
     'STEP 6  |  출력',
     'Depth  ·  CD_top  ·  CD_bot  ·  Aspect Ratio  ·  Taper  ·  Bowing',
     C['s6'], C['dark'])

oval(cx, ye, rc, C['circ'], 'END', C['lite'])

# ══════════════════════════════════════════════════════════
# Δt 루프 표시 (왼쪽)
# STEP5 하단 → (왼쪽으로) → 위로 올라감 → STEP3 상단에 화살촉
# ══════════════════════════════════════════════════════════
lx    = cx - bw/2 - 0.55
t_top = y3              # STEP3 중심 (화살촉 목적지)
t_bot = y5 + bh5/2      # STEP5 하단 출발

# STEP5 하단에서 왼쪽으로
ax.plot([cx - bw/2, lx], [t_bot, t_bot], color=C['loop'], lw=2.0, zorder=2)
# 왼쪽 수직선 (아래→위)
ax.plot([lx, lx], [t_bot, t_top], color=C['loop'], lw=2.0, zorder=2)
# STEP3 중심 높이에서 오른쪽으로 (박스 왼쪽 끝까지) → 화살촉
ax.annotate('', xy=(cx - bw/2, t_top), xytext=(lx, t_top),
            arrowprops=dict(arrowstyle='->', color=C['loop'],
                            lw=2.0, mutation_scale=14), zorder=5)

# 루프 라벨 (방향 + 설명)
ax.text(lx - 0.12, t_bot - 0.05,
        'STEP 5 → STEP 3\n으로 반복',
        ha='right', va='top', fontsize=9.5,
        color=C['loop'], fontweight='bold')
ax.text(lx - 0.12, t_bot + 0.18,
        '(Δt마다 수렴할 때까지)',
        ha='right', va='top', fontsize=8.5,
        color=C['loop'], fontstyle='italic')

# ══════════════════════════════════════════════════════════
# 어노테이션 박스 (화살촉 없이 점선으로 연결)
# 겹침 검증:
#   STEP2 ann [1.68, 2.30] / STEP3 ann [2.42, 3.02] → 간격 0.12 ✓
#   STEP3 ann [2.42, 3.02] / STEP4 ann [3.16, 3.76] → 간격 0.14 ✓
#   STEP4 ann [3.16, 3.76] / STEP5 ann [3.96, 4.56] → 간격 0.20 ✓
# ══════════════════════════════════════════════════════════
re = cx + bw/2   # 메인 박스 오른쪽 끝

def conn(y_val):
    """주 박스 → 어노테이션 점선 연결 (화살촉 없음)"""
    ax.plot([re + 0.05, axc - aw/2 - 0.05], [y_val, y_val],
            color=C['abdr'], lw=0.9, ls='dashed', zorder=2)

annbox(axc, y2, aw, 0.72, [
    '1. F 라디칼 플럭스 모델  (CF4 포화 보정 포함)',
    '2. CFx 라디칼 플럭스 모델',
    '3. 이온 플럭스 모델  (CF4 fragment 이온화 보정)',
    '4. Sheath 이온 에너지 모델',
])
conn(y2)

annbox(axc, y3, aw, 0.60, [
    '1. Clausing power-law  (수직 이온 전달)',
    '2. IAD acceptance-cone  (측면 이온 / shadow)',
    '3. Clausing × 지수감쇠  (중성 입자 전달)',
])
conn(y3)

annbox(axc, y4, aw, 0.72, [
    '1. Bohdansky 스퍼터링 수율',
    '2. 이온강화 식각 인자  (threshold 모델)',
    '3. 수직 식각 속도  (화학 + 이온강화 + 스퍼터 - 패시)',
    '4. 측면 식각 속도  (화학 + IAD 이온강화 - 패시)',
])
conn(y4)

annbox(axc, y5, aw, 0.60, [
    '1. 바닥/측벽 폴리머 동역학  (증착 - 이온제거)',
    '2. 마스크 개구부 진화  (Ar+스퍼터 + F화학 - CFx)',
    '3. Birth CD 모델  (IAD collimation 보정)',
])
conn(y5)

# ══════════════════════════════════════════════════════════
plt.tight_layout(pad=0.3)
out = r'C:\Users\4573k\Desktop\HARC_simulation_Claude\harc_simulator_flowchart.png'
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved -> {out}")
