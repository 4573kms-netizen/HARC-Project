"""
=============================================================================
PINN -> GPR -> BO 파이프라인으로 AR=10 조건 탐색  (세련된 시각화 버전)
=============================================================================
[흐름]
  1. 검증된 시뮬레이터로 5D 공정조건(CF4,Ar,V_bias,time,pressure) LHS 샘플
  2. PINN 학습: 조건 -> (depth,CD_top,CD_bot,AR), 물리손실 포함
  3. GPR: 조건 -> AR + 불확실도(σ)  (별도 대리모델, 신뢰영역 시각화)
  4. BO: PINN을 평가자로 |AR-10| 최소화, GPR 대리 + EI 획득함수
  5. 검증: BO 최적조건을 실제 시뮬레이터로 검증
[출력 그림] figures/
  - fig1_pinn.png   : PINN parity + R² 막대 + 학습곡선
  - fig2_gpr.png    : GPR ±σ 슬라이스 + σ 신뢰영역 히트맵
  - fig3_bo.png     : BO 수렴곡선 + BO vs Random 효율
  - fig4_result.png : PINN vs 시뮬 검증 + 최적 프로파일 단면
=============================================================================
"""
import os, json, warnings, sys
warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
from scipy.stats import qmc, norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from harc_etch_simulator_v2 import ModelParameters, ProcessConditions, run_forward_simulation

_HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(_HERE, 'figures')
os.makedirs(FIGDIR, exist_ok=True)
np.random.seed(0)

# ── 세련된 공통 스타일 ────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 120, 'savefig.dpi': 170,
    'font.size': 11, 'font.family': 'DejaVu Sans',
    'axes.titlesize': 12.5, 'axes.titleweight': 'bold', 'axes.labelsize': 11,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#444444', 'axes.linewidth': 1.0,
    'figure.facecolor': 'white', 'axes.facecolor': '#FAFAFA',
    'legend.frameon': False, 'legend.fontsize': 9.5,
})
C_PINN, C_GPR, C_BO, C_SIM, C_RED = '#2563EB', '#7C3AED', '#059669', '#16A34A', '#DC2626'

# ── 캘리브 파라미터 로드 (고정) ───────────────────────────────────────────
mp = ModelParameters()
with open(os.path.join(_HERE, 'harc_v2_calibrated_params_physics.json')) as f:
    for k, v in json.load(f).items():
        if hasattr(mp, k):
            setattr(mp, k, v)

# ── 입력 공간 (5D) ────────────────────────────────────────────────────────
VARS  = ['CF4', 'Ar', 'V_bias', 'time', 'pressure']
LO = np.array([  2.0,   5.0, -1500.0, 120.0,  4.0])
HI = np.array([ 28.0,  40.0,  -200.0, 600.0, 40.0])
TARGET_AR = 10.0

def make_cond(x):
    cf4, ar, vb, t, p = x
    return ProcessConditions(
        cf4_flow=float(cf4), ar_flow=float(ar), total_flow=float(cf4+ar),
        v_bias=float(vb), etch_time=float(t),
        pressure=float(p), source_power=250.0, substrate_temp=15.0, cd_initial=200.0,
    )

def sim_full(x):
    try:
        return run_forward_simulation(make_cond(x), mp, verbose=False)
    except Exception:
        return None

def sim(x):
    r = sim_full(x)
    if r is None or not np.isfinite(r.aspect_ratio):
        return None
    return [r.total_depth, r.cd_top, r.cd_bot, r.aspect_ratio]

# =============================================================================
# STEP 1: 시뮬레이터로 학습 데이터 생성
# =============================================================================
print("="*60); print("  STEP 1: 시뮬레이터 학습 데이터 (5D LHS)"); print("="*60)
N = 1200
u = qmc.LatinHypercube(d=5, seed=42).random(N)
samples = LO + u * (HI - LO)
X_list, Y_list = [], []
for s in samples:
    y = sim(s)
    if y is not None:
        X_list.append(s); Y_list.append(y)
X_raw = np.array(X_list); Y_raw = np.array(Y_list)
print(f"  valid 샘플: {len(X_raw)} / {N}   AR범위 {Y_raw[:,3].min():.2f}~{Y_raw[:,3].max():.2f}")

def to_unit(X):   return (X - LO)/(HI - LO)
def from_unit(U): return LO + U*(HI - LO)

# =============================================================================
# STEP 2: PINN 학습  (5 -> 64 -> 64 -> 4, tanh, 물리손실)  + 손실 히스토리
# =============================================================================
print("\n" + "="*60); print("  STEP 2: PINN 학습"); print("="*60)
Xm, Xs = X_raw.mean(0), X_raw.std(0) + 1e-8
Ym, Ys = Y_raw.mean(0), Y_raw.std(0) + 1e-8
Xn = (X_raw - Xm) / Xs
Yn = (Y_raw - Ym) / Ys

n_in, n_h, n_out = 5, 64, 4
def unpack(th):
    i = 0
    W1 = th[i:i+n_in*n_h].reshape(n_in, n_h); i += n_in*n_h
    b1 = th[i:i+n_h]; i += n_h
    W2 = th[i:i+n_h*n_h].reshape(n_h, n_h); i += n_h*n_h
    b2 = th[i:i+n_h]; i += n_h
    W3 = th[i:i+n_h*n_out].reshape(n_h, n_out); i += n_h*n_out
    b3 = th[i:i+n_out]
    return W1, b1, W2, b2, W3, b3
def fwd(th, x):
    W1, b1, W2, b2, W3, b3 = unpack(th)
    h1 = np.tanh(x @ W1 + b1); h2 = np.tanh(h1 @ W2 + b2)
    return h2 @ W3 + b3
def data_loss(th):  return np.mean((fwd(th, Xn) - Yn)**2)
def physics_loss(th):
    y = fwd(th, Xn) * Ys + Ym
    depth, cdt, cdb, ar = y[:,0], y[:,1], y[:,2], y[:,3]
    r1 = (ar - depth/np.maximum(cdt,1.))/(Ys[3]+1e-8)
    r2 = np.maximum(cdb-cdt,0.)/(Ys[2]+1e-8)
    r3 = np.maximum(-depth,0.)/(Ys[0]+1e-8)
    r4 = np.maximum(1.0-ar,0.)/(Ys[3]+1e-8)
    return np.mean(r1**2+r2**2+r3**2+r4**2)
LAM = 0.5
def loss(th): return data_loss(th) + LAM*physics_loss(th)

hist = {'total': [], 'data': [], 'phys': []}
def cb(th):
    hist['data'].append(data_loss(th)); hist['phys'].append(physics_loss(th))
    hist['total'].append(hist['data'][-1] + LAM*hist['phys'][-1])

n_theta = n_in*n_h + n_h + n_h*n_h + n_h + n_h*n_out + n_out
np.random.seed(1)
th0 = np.random.randn(n_theta)*0.05
res = minimize(loss, th0, method='L-BFGS-B', callback=cb,
               options={'maxiter':3000, 'ftol':1e-14})
theta = res.x
yp = fwd(theta, Xn)*Ys + Ym
r2 = [1 - np.sum((Y_raw[:,j]-yp[:,j])**2)/(np.sum((Y_raw[:,j]-Y_raw[:,j].mean())**2)+1e-12) for j in range(4)]
print(f"  PINN R²: depth={r2[0]:.3f} CD_top={r2[1]:.3f} CD_bot={r2[2]:.3f} AR={r2[3]:.3f}")

def pinn_predict(x):
    return (fwd(theta, (np.atleast_2d(x)-Xm)/Xs)*Ys + Ym)[0]
def pinn_AR(x): return pinn_predict(x)[3]

# =============================================================================
# STEP 3: GPR (조건 -> AR + σ)  별도 대리모델
# =============================================================================
print("\n" + "="*60); print("  STEP 3: GPR 대리모델 (불확실도 σ)"); print("="*60)
gpr_kernel = ConstantKernel(1.0)*Matern(length_scale=[0.2]*5, nu=2.5) + WhiteKernel(1e-3)
gpr = GaussianProcessRegressor(kernel=gpr_kernel, normalize_y=True,
                               n_restarts_optimizer=3, random_state=0)
gpr.fit(to_unit(X_raw), Y_raw[:,3])     # target = AR
ar_gpr = gpr.predict(to_unit(X_raw))
r2_gpr = 1 - np.sum((Y_raw[:,3]-ar_gpr)**2)/np.sum((Y_raw[:,3]-Y_raw[:,3].mean())**2)
print(f"  GPR R²(AR)={r2_gpr:.3f}")

# =============================================================================
# STEP 4: BO (GPR 대리 + EI) — PINN 평가자로 |AR-10| 최소화  + Random 비교
# =============================================================================
print("\n" + "="*60); print("  STEP 4: Bayesian Optimization"); print("="*60)
def objective(x): return abs(pinn_AR(x) - TARGET_AR)

n_init, n_iter = 12, 40
U_bo = qmc.LatinHypercube(d=5, seed=7).random(n_init)
X_bo = from_unit(U_bo); y_bo = np.array([objective(x) for x in X_bo])
bo_kernel = ConstantKernel(1.0)*Matern(length_scale=[0.2]*5, nu=2.5) + WhiteKernel(1e-4)
bo_hist = [y_bo.min()]
for it in range(n_iter):
    g = GaussianProcessRegressor(kernel=bo_kernel, normalize_y=True,
                                 n_restarts_optimizer=2, random_state=0)
    g.fit(to_unit(X_bo), y_bo)
    Uc = qmc.LatinHypercube(d=5, seed=100+it).random(3000)
    mu, sd = g.predict(Uc, return_std=True); sd = np.maximum(sd, 1e-9)
    fb = y_bo.min(); z = (fb - mu - 0.01)/sd
    ei = np.maximum((fb-mu-0.01)*norm.cdf(z) + sd*norm.pdf(z), 0.0)
    xn_ = from_unit(Uc[np.argmax(ei)]); yn_ = objective(xn_)
    X_bo = np.vstack([X_bo, xn_]); y_bo = np.append(y_bo, yn_)
    bo_hist.append(y_bo.min())
x_best = X_bo[np.argmin(y_bo)]; ar_pinn = pinn_AR(x_best)
print(f"  BO best |AR-10|={y_bo.min():.4f}")

# Random search baseline (동일 예산)
budget = n_init + n_iter
Ur = qmc.LatinHypercube(d=5, seed=999).random(budget)
yr = np.array([objective(x) for x in from_unit(Ur)])
rand_hist = np.minimum.accumulate(yr)

# =============================================================================
# STEP 5: 실제 시뮬레이터 검증
# =============================================================================
print("\n" + "="*60); print("  STEP 5: 시뮬레이터 검증"); print("="*60)
r_best = sim_full(x_best); y_sim = [r_best.total_depth, r_best.cd_top, r_best.cd_bot, r_best.aspect_ratio]
print(f"  최적: CF4={x_best[0]:.2f} Ar={x_best[1]:.2f} V_bias={x_best[2]:.0f} time={x_best[3]:.0f} P={x_best[4]:.1f}")
print(f"  PINN AR={ar_pinn:.3f}  | 시뮬 AR={y_sim[3]:.3f}  (depth={y_sim[0]:.0f}, CD_top={y_sim[1]:.0f}, CD_bot={y_sim[2]:.0f})")

# =============================================================================
# 그림 1 — PINN
# =============================================================================
fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))
fig.suptitle('① PINN — Physics-Informed Surrogate', fontsize=14, fontweight='bold', x=0.5)
# parity
sc = ax[0].scatter(Y_raw[:,3], yp[:,3], c=Y_raw[:,3], cmap='viridis', s=16, alpha=0.6, edgecolor='none')
lims=[Y_raw[:,3].min()-0.3, Y_raw[:,3].max()+0.3]
ax[0].plot(lims, lims, '--', color=C_RED, lw=1.5, label='ideal')
ax[0].set_xlim(lims); ax[0].set_ylim(lims)
ax[0].set_xlabel('Simulator AR'); ax[0].set_ylabel('PINN predicted AR')
ax[0].set_title(f'Parity (AR)   R²={r2[3]:.3f}'); ax[0].legend(loc='upper left')
# R² bar
names=['depth','CD_top','CD_bot','AR']
bars=ax[1].bar(names, r2, color=[C_PINN if v>0.5 else '#9CA3AF' for v in r2],
               edgecolor='k', lw=0.6, alpha=0.9)
ax[1].axhline(0.8, color=C_RED, ls='--', lw=1.2, label='R²=0.8')
ax[1].set_ylim(0,1); ax[1].set_ylabel('R²'); ax[1].set_title('Accuracy per output'); ax[1].legend()
for b,v in zip(bars,r2): ax[1].text(b.get_x()+b.get_width()/2, v+0.02, f'{v:.2f}', ha='center', fontsize=9, fontweight='bold')
# loss curve
ax[2].plot(hist['total'], color=C_PINN, lw=2, label='total')
ax[2].plot(hist['data'],  color=C_SIM,  lw=1.4, ls='--', label='data')
ax[2].plot(np.array(hist['phys'])*LAM, color=C_RED, lw=1.4, ls=':', label='physics×λ')
ax[2].set_yscale('log'); ax[2].set_xlabel('L-BFGS iteration'); ax[2].set_ylabel('loss (log)')
ax[2].set_title('Training loss'); ax[2].legend()
plt.tight_layout(rect=[0,0,1,0.94]); plt.savefig(os.path.join(FIGDIR,'fig1_pinn.png'), facecolor='white'); plt.close()

# =============================================================================
# 그림 2 — GPR (불확실도)
# =============================================================================
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
fig.suptitle('② GPR — Uncertainty Quantification', fontsize=14, fontweight='bold')
# (a) 1D 슬라이스: V_bias 변화, 나머지 최적점 고정 — GPR mean±2σ vs 시뮬 truth
vb_grid = np.linspace(LO[2], HI[2], 60)
Xslice = np.tile(x_best, (60,1)); Xslice[:,2] = vb_grid
mu_s, sd_s = gpr.predict(to_unit(Xslice), return_std=True)
truth_ar = []
for vb in vb_grid:
    t = sim([x_best[0], x_best[1], vb, x_best[3], x_best[4]])
    truth_ar.append(t[3] if t is not None else np.nan)
truth_ar = np.array(truth_ar)
ax[0].fill_between(vb_grid, mu_s-2*sd_s, mu_s+2*sd_s, color=C_GPR, alpha=0.2, label='GPR ±2σ')
ax[0].plot(vb_grid, mu_s, color=C_GPR, lw=2, label='GPR mean')
ax[0].plot(vb_grid, truth_ar, color=C_SIM, lw=1.6, ls='--', label='Simulator truth')
ax[0].axhline(10, color=C_RED, ls=':', lw=1.2, label='target AR=10')
ax[0].axvline(x_best[2], color='k', lw=1, alpha=0.4)
ax[0].set_xlabel('V_bias (V)'); ax[0].set_ylabel('AR')
ax[0].set_title('AR vs V_bias  (others fixed at optimum)'); ax[0].legend(loc='upper right', fontsize=8.5)
# (b) 2D σ 히트맵: V_bias × pressure, 신뢰영역
nb=50
vb_ax = np.linspace(LO[2], HI[2], nb); p_ax = np.linspace(LO[4], HI[4], nb)
VB, PP = np.meshgrid(vb_ax, p_ax)
grid = np.tile(x_best, (nb*nb,1)); grid[:,2]=VB.ravel(); grid[:,4]=PP.ravel()
_, sd_g = gpr.predict(to_unit(grid), return_std=True)
im = ax[1].contourf(VB, PP, sd_g.reshape(nb,nb), levels=20, cmap='magma')
ax[1].scatter(X_raw[:,2], X_raw[:,4], s=6, c='cyan', alpha=0.35, label='training data')
ax[1].scatter([x_best[2]],[x_best[4]], s=140, marker='*', c='white', edgecolor='k', lw=1.2, label='BO optimum', zorder=5)
cb_=plt.colorbar(im, ax=ax[1]); cb_.set_label('GPR σ (uncertainty)')
ax[1].set_xlabel('V_bias (V)'); ax[1].set_ylabel('pressure (mTorr)')
ax[1].set_title('Trust region (σ map)'); ax[1].legend(loc='upper right', fontsize=8.5)
plt.tight_layout(rect=[0,0,1,0.93]); plt.savefig(os.path.join(FIGDIR,'fig2_gpr.png'), facecolor='white'); plt.close()

# =============================================================================
# 그림 3 — BO
# =============================================================================
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
fig.suptitle('③ Bayesian Optimization', fontsize=14, fontweight='bold')
# convergence
ax[0].plot(bo_hist, 'o-', color=C_BO, ms=4, lw=1.8)
ax[0].axhline(0, color=C_RED, ls='--', lw=1)
ax[0].set_xlabel('BO iteration'); ax[0].set_ylabel('best |AR − 10|')
ax[0].set_title('Convergence'); ax[0].set_ylim(bottom=0)
# BO vs Random
ax[1].plot(range(n_init, budget+1), bo_hist, '-', color=C_BO, lw=2, label='Bayesian Opt')
ax[1].plot(range(1, budget+1), rand_hist, '--', color='#9CA3AF', lw=1.8, label='Random search')
ax[1].axhline(0, color=C_RED, ls=':', lw=1)
ax[1].set_xlabel('# simulator/PINN evaluations'); ax[1].set_ylabel('best |AR − 10|')
ax[1].set_title('Efficiency: BO vs Random'); ax[1].legend()
plt.tight_layout(rect=[0,0,1,0.93]); plt.savefig(os.path.join(FIGDIR,'fig3_bo.png'), facecolor='white'); plt.close()

# =============================================================================
# 그림 4 — 검증 + 최적 프로파일
# =============================================================================
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
fig.suptitle('④ Verification & Optimal Profile', fontsize=14, fontweight='bold')
# PINN vs Sim
vals=[ar_pinn, y_sim[3]]
b=ax[0].bar(['PINN\nprediction','Simulator\ntruth'], vals, color=[C_PINN,C_SIM], edgecolor='k', alpha=0.9, width=0.55)
ax[0].axhline(10, color=C_RED, ls='--', lw=1.4, label='target AR=10')
ax[0].set_ylabel('AR at BO-optimal condition'); ax[0].set_title('PINN vs Simulator'); ax[0].legend()
for bb,v in zip(b,vals): ax[0].text(bb.get_x()+bb.get_width()/2, v+0.06, f'{v:.2f}', ha='center', fontweight='bold')
# 최적 프로파일 단면 (좌우 대칭으로)
z = r_best.z_grid; cd = r_best.cd_profile
half = cd/2.0
ax[1].fill_betweenx(z, -half, half, color='#BFDBFE', alpha=0.7)
ax[1].plot(half, z, color=C_PINN, lw=1.6); ax[1].plot(-half, z, color=C_PINN, lw=1.6)
ax[1].invert_yaxis()
ax[1].set_xlabel('radius (nm)'); ax[1].set_ylabel('depth (nm)')
ax[1].set_title(f'Etch profile  AR={y_sim[3]:.2f}\nCF4={x_best[0]:.1f} Ar={x_best[1]:.1f} V={x_best[2]:.0f} t={x_best[3]:.0f} P={x_best[4]:.1f}')
plt.tight_layout(rect=[0,0,1,0.92]); plt.savefig(os.path.join(FIGDIR,'fig4_result.png'), facecolor='white'); plt.close()

print("\n  그림 4장 저장 완료: fig1_pinn / fig2_gpr / fig3_bo / fig4_result (.png)")
print("="*60)
