"""
Bowing을 줄인 (측벽이 곧은) 고AR 프로파일 탐색 & 그리기.
- 시뮬레이터 직접 사용 (캘리브 파라미터 고정)
- LHS로 5D 조건 탐색 → AR 높고 bowing 낮은 조건 선택
- 기존 고bowing(AR=9.96) 조건과 나란히 비교
"""
import os, json, warnings, sys
warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np
from scipy.stats import qmc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from harc_etch_simulator_v2 import ModelParameters, ProcessConditions, run_forward_simulation

_HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({'figure.dpi':120,'savefig.dpi':170,'font.size':11,
    'axes.titlesize':12.5,'axes.titleweight':'bold','axes.grid':True,'grid.alpha':0.25,
    'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'#FAFAFA'})

mp = ModelParameters()
with open(os.path.join(_HERE,'harc_v2_calibrated_params_physics.json')) as f:
    for k,v in json.load(f).items():
        if hasattr(mp,k): setattr(mp,k,v)

LO = np.array([  2.0,   5.0, -1500.0, 120.0,  4.0])
HI = np.array([ 28.0,  40.0,  -200.0, 600.0, 40.0])

def make_cond(x):
    cf4,ar,vb,t,p = x
    return ProcessConditions(cf4_flow=float(cf4),ar_flow=float(ar),total_flow=float(cf4+ar),
        v_bias=float(vb),etch_time=float(t),pressure=float(p),
        source_power=250.0,substrate_temp=15.0,cd_initial=200.0)

def run(x):
    try:
        r = run_forward_simulation(make_cond(x), mp, verbose=False)
        if not np.isfinite(r.aspect_ratio): return None
        return r
    except Exception:
        return None

# ── 탐색: AR 높고 bowing 낮은 조건 ──────────────────────────────────────
print("탐색 중 (LHS 2500) ...")
N = 2500
samp = LO + qmc.LatinHypercube(d=5, seed=11).random(N)*(HI-LO)
rows = []   # (AR, bowing, x, result)
for s in samp:
    r = run(s)
    if r is None: continue
    rows.append((r.aspect_ratio, r.bowing_index, s.copy(), r))

ARs   = np.array([a for a,_,_,_ in rows])
BOWs  = np.array([b for _,b,_,_ in rows])
print(f"  valid {len(rows)}개 | AR {ARs.min():.2f}~{ARs.max():.2f} | bowing {BOWs.min():.2f}~{BOWs.max():.2f}")

# 고AR 기준 (상위권) 중 bowing 최소
AR_THRESH = 8.0
cand = [(a,b,x,r) for (a,b,x,r) in rows if a >= AR_THRESH]
if not cand:
    AR_THRESH = np.percentile(ARs, 90)
    cand = [(a,b,x,r) for (a,b,x,r) in rows if a >= AR_THRESH]
cand.sort(key=lambda z: z[1])         # bowing 오름차순
a_lo, b_lo, x_lo, r_lo = cand[0]      # 저bowing 고AR

# 비교용: 고bowing (AR>=8 중 bowing 최대)
a_hi, b_hi, x_hi, r_hi = max(cand, key=lambda z: z[1])

def desc(x):
    return f"CF4={x[0]:.1f} Ar={x[1]:.1f} V={x[2]:.0f} t={x[3]:.0f} P={x[4]:.1f}"

print(f"\n[저 bowing] AR={a_lo:.2f} bowing={b_lo:.3f}  {desc(x_lo)}")
print(f"   depth={r_lo.total_depth:.0f} CD_top={r_lo.cd_top:.0f} CD_bot={r_lo.cd_bot:.0f}")
print(f"[고 bowing] AR={a_hi:.2f} bowing={b_hi:.3f}  {desc(x_hi)}")
print(f"   depth={r_hi.total_depth:.0f} CD_top={r_hi.cd_top:.0f} CD_bot={r_hi.cd_bot:.0f}")

# ── 프로파일 그리기 (비교) ──────────────────────────────────────────────
def draw(ax, r, x, title, color):
    z = r.z_grid; half = r.cd_profile/2.0
    ax.fill_betweenx(z, -half, half, color=color, alpha=0.25)
    ax.plot(half, z, color=color, lw=1.8); ax.plot(-half, z, color=color, lw=1.8)
    ax.invert_yaxis()
    ax.set_xlim(-130,130)
    ax.set_xlabel('radius (nm)'); ax.set_ylabel('depth (nm)')
    ax.set_title(f'{title}\nAR={r.aspect_ratio:.2f}  bowing={r.bowing_index:.3f}\n{desc(x)}', fontsize=10)

fig, ax = plt.subplots(1, 2, figsize=(11, 6.2))
fig.suptitle('Etch Profile — Low vs High Bowing (AR≥8)', fontsize=14, fontweight='bold')
draw(ax[0], r_lo, x_lo, 'Low bowing (straighter)', '#2563EB')
draw(ax[1], r_hi, x_hi, 'High bowing (bulged)', '#DC2626')
plt.tight_layout(rect=[0,0,1,0.92])
out = os.path.join(_HERE,'figures','profile_low_bowing.png')
plt.savefig(out, facecolor='white'); plt.close()
print(f"\n그림 저장: {out}")
