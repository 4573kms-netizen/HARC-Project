"""
=============================================================================
물리 보정 PINN — 시뮬값의 비물리적 부분을 강한 물리손실로 교정
=============================================================================
[아이디어]
  시뮬레이터는 극단 조건에서 CD_bot를 비물리적으로 좁게(<~30nm) 예측.
  실제로는 그 정도면 식각 전면이 막힘(pinch-off / ARDE).
  → CD_bot에 pinch-off 하한(CD_MIN) 물리 패널티를 강하게 부여.
  → PINN이 시뮬을 단순 복사하지 않고, 비물리 영역만 교정.
  → R²(vs 시뮬) < 1 이 되는데, 그 '차이'가 물리 보정의 증거.

변경 변수: CF4, V_bias, time   (Ar=30-CF4, pressure=10 고정)
=============================================================================
"""
import os, json, warnings, sys
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
import numpy as np
from scipy.stats import qmc
from scipy.optimize import minimize
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from harc_etch_simulator_v2 import ModelParameters, ProcessConditions, run_forward_simulation

_HERE = os.path.dirname(os.path.abspath(__file__)); FIGDIR = os.path.join(_HERE,'figures')
os.makedirs(FIGDIR, exist_ok=True)
plt.rcParams.update({'figure.dpi':120,'savefig.dpi':170,'font.size':11,
    'axes.titlesize':12,'axes.titleweight':'bold','axes.grid':True,'grid.alpha':0.25,
    'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'#FAFAFA'})
C_PINN,C_SIM,C_RED,C_OK = '#2563EB','#16A34A','#DC2626','#9CA3AF'

mp = ModelParameters()
with open(os.path.join(_HERE,'harc_v2_calibrated_params_physics.json')) as f:
    for k,v in json.load(f).items():
        if hasattr(mp,k): setattr(mp,k,v)

TOTAL_FLOW = 30.0
LO = np.array([  2.0, -1500.0, 120.0]); HI = np.array([ 28.0, -200.0, 600.0])
CD_MIN = 30.0      # [물리] pinch-off 하한 (이보다 좁으면 비물리)
W_PINCH = 6.0      # pinch-off 패널티 가중 (크게)

def make_cond(x):
    cf4,vb,t = x
    return ProcessConditions(cf4_flow=float(cf4),ar_flow=float(TOTAL_FLOW-cf4),total_flow=TOTAL_FLOW,
        v_bias=float(vb),etch_time=float(t),pressure=10.0,source_power=250.0,substrate_temp=15.0,cd_initial=200.0)
def sim(x):
    try:
        r=run_forward_simulation(make_cond(x),mp,verbose=False)
        return None if not np.isfinite(r.aspect_ratio) else [r.total_depth,r.cd_top,r.cd_bot,r.aspect_ratio]
    except Exception: return None

# ── 데이터 ────────────────────────────────────────────────────────────────
print("STEP 1: 데이터 생성 ...")
N=1500
samp = LO + qmc.LatinHypercube(d=3,seed=42).random(N)*(HI-LO)
Xl,Yl=[],[]
for s in samp:
    y=sim(s)
    if y: Xl.append(s); Yl.append(y)
X_raw=np.array(Xl); Y_raw=np.array(Yl)
n_pinch = int(np.sum(Y_raw[:,2] < CD_MIN))
print(f"  valid {len(X_raw)} | CD_bot<{CD_MIN}nm(pinch) 샘플 {n_pinch}개 ({100*n_pinch/len(X_raw):.0f}%)")

# ── PINN (backprop) + pinch-off 물리 보정 ────────────────────────────────
print("STEP 2: 물리보정 PINN 학습 ...")
Xm,Xs=X_raw.mean(0),X_raw.std(0)+1e-8; Ym,Ys=Y_raw.mean(0),Y_raw.std(0)+1e-8
Xn=(X_raw-Xm)/Xs; Yn=(Y_raw-Ym)/Ys
N_,n_in,n_h,n_out = len(Xn),3,64,4
LAM=0.5

def unpack(th):
    i=0
    W1=th[i:i+n_in*n_h].reshape(n_in,n_h);i+=n_in*n_h
    b1=th[i:i+n_h];i+=n_h
    W2=th[i:i+n_h*n_h].reshape(n_h,n_h);i+=n_h*n_h
    b2=th[i:i+n_h];i+=n_h
    W3=th[i:i+n_h*n_out].reshape(n_h,n_out);i+=n_h*n_out
    b3=th[i:i+n_out]
    return W1,b1,W2,b2,W3,b3
def pack(*a): return np.concatenate([x.ravel() for x in a])
def fwd_full(th,x):
    W1,b1,W2,b2,W3,b3=unpack(th)
    z1=x@W1+b1;h1=np.tanh(z1); z2=h1@W2+b2;h2=np.tanh(z2); yn=h2@W3+b3
    return yn,h1,h2

def phys(yn):
    y=yn*Ys+Ym
    depth,cdt,cdb,ar=y[:,0],y[:,1],y[:,2],y[:,3]; mcd=np.maximum(cdt,1.0)
    r1=(ar-depth/mcd)/Ys[3]; r2=np.maximum(cdb-cdt,0)/Ys[2]
    r3=np.maximum(-depth,0)/Ys[0]; r4=np.maximum(1-ar,0)/Ys[3]
    r5=np.maximum(CD_MIN-cdb,0)/Ys[2]                     # ← pinch-off 하한 (핵심)
    Lp=np.mean(r1**2+r2**2+r3**2+r4**2 + W_PINCH*r5**2)
    gun=np.zeros_like(y)
    gun[:,3]+=2*r1/Ys[3]; gun[:,0]+=2*r1*(-1.0/mcd)/Ys[3]
    gun[:,1]+=np.where(cdt>1.0,2*r1*(depth/cdt**2)/Ys[3],0.0)
    m2=(cdb-cdt)>0; gun[:,2]+=np.where(m2,2*r2/Ys[2],0.0); gun[:,1]+=np.where(m2,-2*r2/Ys[2],0.0)
    gun[:,0]+=np.where((-depth)>0,-2*r3/Ys[0],0.0)
    gun[:,3]+=np.where((1-ar)>0,-2*r4/Ys[3],0.0)
    gun[:,2]+=np.where((CD_MIN-cdb)>0, W_PINCH*2*r5*(-1.0/Ys[2]),0.0)   # pinch grad → cdb 올림
    return gun*Ys/N_, Lp

def loss_and_grad(th):
    yn,h1,h2=fwd_full(th,Xn); Ld=np.mean((yn-Yn)**2)
    gp,Lp=phys(yn); g_y=2*(yn-Yn)/(N_*n_out)+LAM*gp
    W1,b1,W2,b2,W3,b3=unpack(th)
    gW3=h2.T@g_y;gb3=g_y.sum(0); gh2=g_y@W3.T;gz2=gh2*(1-h2**2)
    gW2=h1.T@gz2;gb2=gz2.sum(0); gh1=gz2@W2.T;gz1=gh1*(1-h1**2)
    gW1=Xn.T@gz1;gb1=gz1.sum(0)
    return Ld+LAM*Lp, pack(gW1,gb1,gW2,gb2,gW3,gb3)

n_theta=n_in*n_h+n_h+n_h*n_h+n_h+n_h*n_out+n_out
np.random.seed(1); th0=np.random.randn(n_theta)*0.1
res=minimize(loss_and_grad,th0,jac=True,method='L-BFGS-B',options={'maxiter':5000,'ftol':1e-15,'gtol':1e-12})
theta=res.x
yp=(fwd_full(theta,Xn)[0])*Ys+Ym
def R2(j): return 1-np.sum((Y_raw[:,j]-yp[:,j])**2)/(np.sum((Y_raw[:,j]-Y_raw[:,j].mean())**2)+1e-12)
r2=[R2(j) for j in range(4)]
print(f"  PINN R² vs 시뮬: depth={r2[0]:.3f} CD_top={r2[1]:.3f} CD_bot={r2[2]:.3f} AR={r2[3]:.3f}")
print(f"  (CD_bot R²가 1보다 작음 = pinch 영역에서 물리 교정한 증거)")

# pinch 영역에서 얼마나 교정했나
pinch_mask = Y_raw[:,2] < CD_MIN
if pinch_mask.sum()>0:
    sim_cdb = Y_raw[pinch_mask,2]; pinn_cdb = yp[pinch_mask,2]
    print(f"  pinch 영역: 시뮬 CD_bot 평균={sim_cdb.mean():.1f}nm → PINN={pinn_cdb.mean():.1f}nm (끌어올림)")

# ── 그림 1: 4출력 parity (CD_bot만 어긋남) ───────────────────────────────
fig,ax=plt.subplots(1,4,figsize=(18,4.4))
fig.suptitle('Physics-corrected PINN — parity vs Simulator  (CD_bot deviates = correction)',fontsize=13,fontweight='bold')
names=['depth','CD_top','CD_bot','AR']
for j in range(4):
    a=ax[j]
    if j==2:  # CD_bot: pinch 강조
        norm=~pinch_mask
        a.scatter(Y_raw[norm,2],yp[norm,2],s=12,alpha=0.5,color=C_OK,label='normal')
        a.scatter(Y_raw[pinch_mask,2],yp[pinch_mask,2],s=22,alpha=0.8,color=C_RED,label='pinch (corrected)')
        a.axhline(CD_MIN,color=C_PINN,ls=':',lw=1.4,label=f'CD_min={CD_MIN:.0f}')
        a.axvline(CD_MIN,color=C_PINN,ls=':',lw=1.0)
        a.legend(fontsize=8)
    else:
        a.scatter(Y_raw[:,j],yp[:,j],s=12,alpha=0.5,color=C_PINN)
    lim=[Y_raw[:,j].min(),Y_raw[:,j].max()]; a.plot(lim,lim,'--',color=C_RED,lw=1.3)
    a.set_xlabel(f'Simulator {names[j]}'); a.set_ylabel(f'PINN {names[j]}')
    a.set_title(f'{names[j]}  R²={r2[j]:.3f}')
plt.tight_layout(rect=[0,0,1,0.92]); plt.savefig(os.path.join(FIGDIR,'fig_pc_parity.png'),facecolor='white'); plt.close()

# ── 그림 2: CD_bot 교정 집중 ─────────────────────────────────────────────
fig,ax=plt.subplots(1,1,figsize=(6.5,5.5))
ax.scatter(Y_raw[~pinch_mask,2],yp[~pinch_mask,2],s=14,alpha=0.5,color=C_OK,label='normal (PINN=Sim)')
ax.scatter(Y_raw[pinch_mask,2],yp[pinch_mask,2],s=30,alpha=0.85,color=C_RED,label='pinch-off region')
lim=[Y_raw[:,2].min()-2,Y_raw[:,2].max()]; ax.plot(lim,lim,'--',color='k',lw=1.2,label='PINN=Simulator')
ax.axhspan(lim[0],CD_MIN,color=C_PINN,alpha=0.07)
ax.axhline(CD_MIN,color=C_PINN,ls=':',lw=1.6,label=f'physics floor CD_min={CD_MIN:.0f}nm')
ax.set_xlabel('Simulator CD_bot (nm)'); ax.set_ylabel('Physics-corrected PINN CD_bot (nm)')
ax.set_title('PINN raises unphysically narrow CD_bot\n(pinch-off correction)')
ax.legend(fontsize=9);
plt.tight_layout(); plt.savefig(os.path.join(FIGDIR,'fig_pc_cdbot.png'),facecolor='white'); plt.close()

print("\n  그림 저장: fig_pc_parity.png, fig_pc_cdbot.png")
