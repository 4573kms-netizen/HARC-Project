"""
=============================================================================
3-변수 PINN(analytic backprop) -> GPR 3D 곡면 -> BO  (AR 탐색)
=============================================================================
변경 변수 (3개):  CF4, V_bias, time
고정:  total_flow=30 (Ar=30-CF4), pressure=10, power=250, temp=15  (캘리브 기준)

[개선점]
  - PINN에 해석적 기울기(backprop) 적용 → 수치미분 대비 R²↑ & 속도↑
  - 입력 3D → GPR 곡면을 3차원 surface로 시각화
[출력 그림] figures/
  - fig3d_pinn.png  : parity(R²) + 학습곡선
  - fig3d_gpr.png   : 3D AR 곡면 (V_bias × time, CF4 고정) + σ
=============================================================================
"""
import os, json, warnings, sys
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
import numpy as np
from scipy.stats import qmc, norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from harc_etch_simulator_v2 import ModelParameters, ProcessConditions, run_forward_simulation

_HERE = os.path.dirname(os.path.abspath(__file__)); FIGDIR = os.path.join(_HERE,'figures')
os.makedirs(FIGDIR, exist_ok=True)
plt.rcParams.update({'figure.dpi':120,'savefig.dpi':170,'font.size':11,
    'axes.titlesize':12.5,'axes.titleweight':'bold','axes.grid':True,'grid.alpha':0.25,
    'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'#FAFAFA'})
C_PINN,C_GPR,C_SIM,C_RED = '#2563EB','#7C3AED','#16A34A','#DC2626'

mp = ModelParameters()
with open(os.path.join(_HERE,'harc_v2_calibrated_params_physics.json')) as f:
    for k,v in json.load(f).items():
        if hasattr(mp,k): setattr(mp,k,v)

# ── 3 입력 (CF4, V_bias, time), total flow=30 고정 ────────────────────────
TOTAL_FLOW = 30.0
VARS = ['CF4','V_bias','time']
LO = np.array([  2.0, -1500.0, 120.0])
HI = np.array([ 28.0,  -200.0, 600.0])
TARGET_AR = 10.0

def make_cond(x):
    cf4, vb, t = x
    return ProcessConditions(cf4_flow=float(cf4), ar_flow=float(TOTAL_FLOW-cf4), total_flow=TOTAL_FLOW,
        v_bias=float(vb), etch_time=float(t), pressure=10.0,
        source_power=250.0, substrate_temp=15.0, cd_initial=200.0)

def sim(x):
    try:
        r = run_forward_simulation(make_cond(x), mp, verbose=False)
        if not np.isfinite(r.aspect_ratio): return None
        return r
    except Exception: return None

# =============================================================================
# STEP 1: 데이터
# =============================================================================
print("="*60); print("  STEP 1: 시뮬 데이터 (3D LHS)"); print("="*60)
N = 1500
samp = LO + qmc.LatinHypercube(d=3, seed=42).random(N)*(HI-LO)
Xl, Yl = [], []
for s in samp:
    r = sim(s)
    if r: Xl.append(s); Yl.append([r.total_depth, r.cd_top, r.cd_bot, r.aspect_ratio])
X_raw = np.array(Xl); Y_raw = np.array(Yl)
print(f"  valid {len(X_raw)}/{N}  AR {Y_raw[:,3].min():.2f}~{Y_raw[:,3].max():.2f}")

def to_unit(X): return (X-LO)/(HI-LO)
def from_unit(U): return LO+U*(HI-LO)

# =============================================================================
# STEP 2: PINN (analytic backprop)
# =============================================================================
print("\n"+"="*60); print("  STEP 2: PINN 학습 (해석적 backprop)"); print("="*60)
Xm,Xs = X_raw.mean(0), X_raw.std(0)+1e-8
Ym,Ys = Y_raw.mean(0), Y_raw.std(0)+1e-8
Xn = (X_raw-Xm)/Xs; Yn = (Y_raw-Ym)/Ys
N_, n_in, n_h, n_out = len(Xn), 3, 64, 4
LAM = 0.3

def unpack(th):
    i=0
    W1=th[i:i+n_in*n_h].reshape(n_in,n_h); i+=n_in*n_h
    b1=th[i:i+n_h]; i+=n_h
    W2=th[i:i+n_h*n_h].reshape(n_h,n_h); i+=n_h*n_h
    b2=th[i:i+n_h]; i+=n_h
    W3=th[i:i+n_h*n_out].reshape(n_h,n_out); i+=n_h*n_out
    b3=th[i:i+n_out]
    return W1,b1,W2,b2,W3,b3
def pack(*a): return np.concatenate([x.ravel() for x in a])

def fwd_full(th,x):
    W1,b1,W2,b2,W3,b3 = unpack(th)
    z1=x@W1+b1; h1=np.tanh(z1)
    z2=h1@W2+b2; h2=np.tanh(z2)
    yn=h2@W3+b3
    return yn,h1,h2

def phys_grad_on_yn(yn):
    """물리손실의 d/d(yn) (N,4) 와 손실값 반환"""
    y = yn*Ys+Ym
    depth,cdt,cdb,ar = y[:,0],y[:,1],y[:,2],y[:,3]
    mcd = np.maximum(cdt,1.0)
    r1=(ar-depth/mcd)/Ys[3]; r2=np.maximum(cdb-cdt,0)/Ys[2]
    r3=np.maximum(-depth,0)/Ys[0]; r4=np.maximum(1-ar,0)/Ys[3]
    Lp = np.mean(r1**2+r2**2+r3**2+r4**2)
    gun = np.zeros_like(y)
    gun[:,3]+=2*r1*(1.0/Ys[3])
    gun[:,0]+=2*r1*(-1.0/mcd)/Ys[3]
    gun[:,1]+=np.where(cdt>1.0, 2*r1*(depth/cdt**2)/Ys[3], 0.0)
    m2=(cdb-cdt)>0
    gun[:,2]+=np.where(m2, 2*r2*(1.0/Ys[2]),0.0); gun[:,1]+=np.where(m2,2*r2*(-1.0/Ys[2]),0.0)
    gun[:,0]+=np.where((-depth)>0, 2*r3*(-1.0/Ys[0]),0.0)
    gun[:,3]+=np.where((1-ar)>0, 2*r4*(-1.0/Ys[3]),0.0)
    g_yn = gun*Ys/N_
    return g_yn, Lp

def loss_and_grad(th):
    yn,h1,h2 = fwd_full(th,Xn)
    Ld = np.mean((yn-Yn)**2)
    g_yn_p, Lp = phys_grad_on_yn(yn)
    g_y = 2*(yn-Yn)/(N_*n_out) + LAM*g_yn_p
    W1,b1,W2,b2,W3,b3 = unpack(th)
    gW3=h2.T@g_y; gb3=g_y.sum(0)
    gh2=g_y@W3.T; gz2=gh2*(1-h2**2)
    gW2=h1.T@gz2; gb2=gz2.sum(0)
    gh1=gz2@W2.T; gz1=gh1*(1-h1**2)
    gW1=Xn.T@gz1; gb1=gz1.sum(0)
    return Ld+LAM*Lp, pack(gW1,gb1,gW2,gb2,gW3,gb3)

n_theta = n_in*n_h+n_h+n_h*n_h+n_h+n_h*n_out+n_out
np.random.seed(1); th0 = np.random.randn(n_theta)*0.1

# --- 빠른 gradient check ---
L0,g0 = loss_and_grad(th0)
idxs = np.random.RandomState(0).randint(0,n_theta,5); eps=1e-5; errs=[]
for j in idxs:
    tp=th0.copy(); tp[j]+=eps; tm=th0.copy(); tm[j]-=eps
    num=(loss_and_grad(tp)[0]-loss_and_grad(tm)[0])/(2*eps)
    errs.append(abs(num-g0[j])/(abs(num)+1e-9))
print(f"  gradient check (rel err): max={max(errs):.2e}  (작으면 OK)")

hist=[]
def cb(th): hist.append(loss_and_grad(th)[0])
res = minimize(loss_and_grad, th0, jac=True, method='L-BFGS-B', callback=cb,
               options={'maxiter':5000,'ftol':1e-15,'gtol':1e-12})
theta=res.x
yp = (fwd_full(theta,Xn)[0])*Ys+Ym
r2 = [1-np.sum((Y_raw[:,j]-yp[:,j])**2)/(np.sum((Y_raw[:,j]-Y_raw[:,j].mean())**2)+1e-12) for j in range(4)]
print(f"  PINN R²: depth={r2[0]:.3f} CD_top={r2[1]:.3f} CD_bot={r2[2]:.3f} AR={r2[3]:.3f}")
def pinn_pred(x): return (fwd_full(theta,(np.atleast_2d(x)-Xm)/Xs)[0])*Ys+Ym
def pinn_AR(x): return pinn_pred(x)[0,3]

# =============================================================================
# STEP 3: GPR
# =============================================================================
print("\n"+"="*60); print("  STEP 3: GPR"); print("="*60)
gk = ConstantKernel(1.0)*Matern(length_scale=[0.2]*3,nu=2.5)+WhiteKernel(1e-3)
gpr = GaussianProcessRegressor(kernel=gk, normalize_y=True, n_restarts_optimizer=3, random_state=0)
gpr.fit(to_unit(X_raw), Y_raw[:,3])
print(f"  GPR R²(AR)={gpr.score(to_unit(X_raw),Y_raw[:,3]):.3f}")

# =============================================================================
# STEP 4: BO (PINN 평가자, EI)  -> AR 최대(또는 10 근접) 조건
# =============================================================================
print("\n"+"="*60); print("  STEP 4: BO"); print("="*60)
def objective(x): return abs(pinn_AR(x)-TARGET_AR)
n_init,n_iter=10,35
Xb=from_unit(qmc.LatinHypercube(d=3,seed=7).random(n_init)); yb=np.array([objective(x) for x in Xb])
bk=ConstantKernel(1.0)*Matern(length_scale=[0.2]*3,nu=2.5)+WhiteKernel(1e-4)
for it in range(n_iter):
    g=GaussianProcessRegressor(kernel=bk,normalize_y=True,n_restarts_optimizer=2,random_state=0); g.fit(to_unit(Xb),yb)
    Uc=qmc.LatinHypercube(d=3,seed=100+it).random(4000); mu,sd=g.predict(Uc,return_std=True); sd=np.maximum(sd,1e-9)
    fb=yb.min(); z=(fb-mu-0.01)/sd; ei=np.maximum((fb-mu-0.01)*norm.cdf(z)+sd*norm.pdf(z),0)
    xn_=from_unit(Uc[np.argmax(ei)]); Xb=np.vstack([Xb,xn_]); yb=np.append(yb,objective(xn_))
x_best=Xb[np.argmin(yb)]; r_best=sim(x_best)
print(f"  최적: CF4={x_best[0]:.2f} (Ar={TOTAL_FLOW-x_best[0]:.2f}) V_bias={x_best[1]:.0f} time={x_best[2]:.0f}")
print(f"  PINN AR={pinn_AR(x_best):.3f} | 시뮬 AR={r_best.aspect_ratio:.3f} (depth={r_best.total_depth:.0f}, CD_top={r_best.cd_top:.0f})")

# =============================================================================
# 그림 A — PINN
# =============================================================================
fig,ax=plt.subplots(1,2,figsize=(11,4.6))
fig.suptitle('PINN (3-var, analytic backprop)', fontsize=14, fontweight='bold')
ax[0].scatter(Y_raw[:,3],yp[:,3],c=Y_raw[:,3],cmap='viridis',s=14,alpha=0.6)
lim=[Y_raw[:,3].min()-0.3,Y_raw[:,3].max()+0.3]; ax[0].plot(lim,lim,'--',color=C_RED,lw=1.5)
ax[0].set_xlim(lim);ax[0].set_ylim(lim);ax[0].set_xlabel('Simulator AR');ax[0].set_ylabel('PINN AR')
ax[0].set_title(f'Parity (AR)  R²={r2[3]:.3f}')
ax[1].plot(hist,color=C_PINN,lw=2); ax[1].set_yscale('log'); ax[1].set_xlabel('L-BFGS iter'); ax[1].set_ylabel('loss'); ax[1].set_title('Training loss')
plt.tight_layout(rect=[0,0,1,0.93]); plt.savefig(os.path.join(FIGDIR,'fig3d_pinn.png'),facecolor='white'); plt.close()

# =============================================================================
# 그림 B — GPR 3D 곡면 (V_bias × time, CF4=최적 고정)
# =============================================================================
cf4_fix = x_best[0]
nb=45
vb_ax=np.linspace(LO[1],HI[1],nb); t_ax=np.linspace(LO[2],HI[2],nb)
VB,TT=np.meshgrid(vb_ax,t_ax)
grid=np.column_stack([np.full(VB.size,cf4_fix), VB.ravel(), TT.ravel()])
mu_g,sd_g=gpr.predict(to_unit(grid),return_std=True)
AR_surf=mu_g.reshape(nb,nb); SD_surf=sd_g.reshape(nb,nb)

fig=plt.figure(figsize=(14,5.5)); fig.suptitle(f'GPR surrogate — 3D AR surface  (CF4={cf4_fix:.1f} fixed)', fontsize=14, fontweight='bold')
ax1=fig.add_subplot(1,2,1,projection='3d')
surf=ax1.plot_surface(VB,TT,AR_surf,cmap='viridis',alpha=0.9,linewidth=0,antialiased=True)
ax1.contour(VB,TT,AR_surf,zdir='z',offset=AR_surf.min(),cmap='viridis',linewidths=0.8)
ax1.scatter([x_best[1]],[x_best[2]],[r_best.aspect_ratio],c='red',s=80,marker='*',label='BO optimum')
ax1.set_xlabel('V_bias (V)'); ax1.set_ylabel('time (s)'); ax1.set_zlabel('AR')
ax1.set_title('AR = f(V_bias, time)'); fig.colorbar(surf,ax=ax1,shrink=0.5,pad=0.1)
# σ 곡면
ax2=fig.add_subplot(1,2,2,projection='3d')
s2=ax2.plot_surface(VB,TT,SD_surf,cmap='magma',alpha=0.9,linewidth=0)
ax2.set_xlabel('V_bias (V)'); ax2.set_ylabel('time (s)'); ax2.set_zlabel('σ (uncertainty)')
ax2.set_title('Uncertainty σ surface'); fig.colorbar(s2,ax=ax2,shrink=0.5,pad=0.1)
plt.tight_layout(rect=[0,0,1,0.93]); plt.savefig(os.path.join(FIGDIR,'fig3d_gpr.png'),facecolor='white'); plt.close()

print("\n  그림 저장: fig3d_pinn.png, fig3d_gpr.png")
print("="*60)
