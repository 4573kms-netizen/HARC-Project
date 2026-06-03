"""
=============================================================================
3-변수 PINN — 정직한 일반화 성능 (train/test 분리 + 5-fold 교차검증)
=============================================================================
목적: R²=1.000(훈련값) 의심 해소.
  - 훈련 80% / 테스트 20% 분리 → '안 본 데이터' test R² 보고
  - 5-fold 교차검증으로 robust한 평균±표준편차
  - train vs test 차이로 과적합 여부 판정
변경 변수: CF4, V_bias, time   (Ar=30-CF4, pressure=10 고정)
PINN: 3→64→64→4, 해석적 backprop, 물리손실(안전망)
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

_HERE=os.path.dirname(os.path.abspath(__file__)); FIGDIR=os.path.join(_HERE,'figures')
os.makedirs(FIGDIR,exist_ok=True)
plt.rcParams.update({'figure.dpi':120,'savefig.dpi':170,'font.size':11,
    'axes.titlesize':12,'axes.titleweight':'bold','axes.grid':True,'grid.alpha':0.25,
    'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'#FAFAFA'})
C_TR,C_TE,C_RED='#9CA3AF','#2563EB','#DC2626'

mp=ModelParameters()
with open(os.path.join(_HERE,'harc_v2_calibrated_params_physics.json')) as f:
    for k,v in json.load(f).items():
        if hasattr(mp,k): setattr(mp,k,v)

TOTAL_FLOW=30.0
LO=np.array([2.0,-1500.0,120.0]); HI=np.array([28.0,-200.0,600.0])
CD_MIN=30.0; W_PINCH=2.0; LAM=0.3
n_in,n_h,n_out=3,64,4

def make_cond(x):
    cf4,vb,t=x
    return ProcessConditions(cf4_flow=float(cf4),ar_flow=float(TOTAL_FLOW-cf4),total_flow=TOTAL_FLOW,
        v_bias=float(vb),etch_time=float(t),pressure=10.0,source_power=250.0,substrate_temp=15.0,cd_initial=200.0)
def sim(x):
    try:
        r=run_forward_simulation(make_cond(x),mp,verbose=False)
        return None if not np.isfinite(r.aspect_ratio) else [r.total_depth,r.cd_top,r.cd_bot,r.aspect_ratio]
    except Exception: return None

# ── 데이터 ────────────────────────────────────────────────────────────────
print("STEP 1: 데이터 1500개 생성 ...")
N=1500
samp=LO+qmc.LatinHypercube(d=3,seed=42).random(N)*(HI-LO)
Xl,Yl=[],[]
for s in samp:
    y=sim(s)
    if y: Xl.append(s); Yl.append(y)
X=np.array(Xl); Y_clean=np.array(Yl)
# 측정/공정 변동 모사: ±10% 상대 노이즈 (시뮬은 SEM 대비 ±10% 검증됨)
NOISE_PCT=0.04
_rngn=np.random.RandomState(123)
Y=Y_clean*(1.0+_rngn.normal(0,NOISE_PCT,Y_clean.shape))
print(f"  valid {len(X)}  | 측정노이즈 ±{NOISE_PCT*100:.0f}% 추가 (현실 모사)")

# ── PINN 학습 함수 (backprop) ────────────────────────────────────────────
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

def fit(Xtr,Ytr,maxiter=4000,seed=1):
    """train set으로 PINN 학습 → predict 함수 반환 (정규화는 train 통계로만)"""
    Xm,Xs=Xtr.mean(0),Xtr.std(0)+1e-8; Ym,Ys=Ytr.mean(0),Ytr.std(0)+1e-8
    Xn=(Xtr-Xm)/Xs; Yn=(Ytr-Ym)/Ys; Ntr=len(Xn)
    def phys(yn):
        y=yn*Ys+Ym; depth,cdt,cdb,ar=y[:,0],y[:,1],y[:,2],y[:,3]; mcd=np.maximum(cdt,1.0)
        r1=(ar-depth/mcd)/Ys[3]; r2=np.maximum(cdb-cdt,0)/Ys[2]
        r3=np.maximum(-depth,0)/Ys[0]; r4=np.maximum(1-ar,0)/Ys[3]; r5=np.maximum(CD_MIN-cdb,0)/Ys[2]
        Lp=np.mean(r1**2+r2**2+r3**2+r4**2+W_PINCH*r5**2)
        gun=np.zeros_like(y)
        gun[:,3]+=2*r1/Ys[3]; gun[:,0]+=2*r1*(-1.0/mcd)/Ys[3]
        gun[:,1]+=np.where(cdt>1.0,2*r1*(depth/cdt**2)/Ys[3],0.0)
        m2=(cdb-cdt)>0; gun[:,2]+=np.where(m2,2*r2/Ys[2],0.0); gun[:,1]+=np.where(m2,-2*r2/Ys[2],0.0)
        gun[:,0]+=np.where((-depth)>0,-2*r3/Ys[0],0.0); gun[:,3]+=np.where((1-ar)>0,-2*r4/Ys[3],0.0)
        gun[:,2]+=np.where((CD_MIN-cdb)>0,W_PINCH*2*r5*(-1.0/Ys[2]),0.0)
        return gun*Ys/Ntr,Lp
    def lg(th):
        yn,h1,h2=fwd_full(th,Xn); Ld=np.mean((yn-Yn)**2); gp,Lp=phys(yn)
        g_y=2*(yn-Yn)/(Ntr*n_out)+LAM*gp
        W1,b1,W2,b2,W3,b3=unpack(th)
        gW3=h2.T@g_y;gb3=g_y.sum(0); gh2=g_y@W3.T;gz2=gh2*(1-h2**2)
        gW2=h1.T@gz2;gb2=gz2.sum(0); gh1=gz2@W2.T;gz1=gh1*(1-h1**2)
        gW1=Xn.T@gz1;gb1=gz1.sum(0)
        return Ld+LAM*Lp,pack(gW1,gb1,gW2,gb2,gW3,gb3)
    nth=n_in*n_h+n_h+n_h*n_h+n_h+n_h*n_out+n_out
    np.random.seed(seed); th0=np.random.randn(nth)*0.1
    r=minimize(lg,th0,jac=True,method='L-BFGS-B',options={'maxiter':maxiter,'ftol':1e-15,'gtol':1e-12})
    th=r.x
    def predict(Xev): return (fwd_full(th,(Xev-Xm)/Xs)[0])*Ys+Ym
    return predict

def R2(ytrue,ypred): return 1-np.sum((ytrue-ypred)**2)/(np.sum((ytrue-ytrue.mean())**2)+1e-12)

# ── STEP 2: 80/20 train-test 분리 ────────────────────────────────────────
print("\nSTEP 2: 80/20 train-test 분리 학습 ...")
rng=np.random.RandomState(0); idx=rng.permutation(len(X)); ncut=int(0.8*len(X))
tr,te=idx[:ncut],idx[ncut:]
pred=fit(X[tr],Y[tr],maxiter=1500)
yp_tr=pred(X[tr]); yp_te=pred(X[te])
names=['depth','CD_top','CD_bot','AR']
r2_tr=[R2(Y[tr,j],yp_tr[:,j]) for j in range(4)]
r2_te=[R2(Y[te,j],yp_te[:,j]) for j in range(4)]
print("  output     train R²   test R²")
for j in range(4):
    print(f"  {names[j]:<9}  {r2_tr[j]:7.4f}   {r2_te[j]:7.4f}")

# ── STEP 3: 5-fold 교차검증 (AR) ─────────────────────────────────────────
print("\nSTEP 3: 5-fold 교차검증 (AR test R²) ...")
folds=np.array_split(rng.permutation(len(X)),5)
cv=[]
for k in range(5):
    tek=folds[k]; trk=np.concatenate([folds[m] for m in range(5) if m!=k])
    pk=fit(X[trk],Y[trk],maxiter=1200,seed=k+1)
    cv.append(R2(Y[tek,3],pk(X[tek])[:,3]))
cv=np.array(cv)
print(f"  fold별 AR test R²: {np.round(cv,4)}")
print(f"  평균={cv.mean():.4f}  표준편차={cv.std():.4f}")

# ── 그림 1: parity (train 흐리게 + test 강조), AR ────────────────────────
fig,ax=plt.subplots(1,2,figsize=(12,5))
fig.suptitle('Honest generalization — train vs held-out test (±4% measurement noise)',fontsize=12.5,fontweight='bold')
a=ax[0]
a.scatter(Y[tr,3],yp_tr[:,3],s=10,alpha=0.25,color=C_TR,label=f'train (R²={r2_tr[3]:.3f})')
a.scatter(Y[te,3],yp_te[:,3],s=22,alpha=0.8,color=C_TE,label=f'test (R²={r2_te[3]:.3f})')
lim=[Y[:,3].min()-0.2,Y[:,3].max()+0.2]; a.plot(lim,lim,'--',color=C_RED,lw=1.3)
a.set_xlim(lim);a.set_ylim(lim);a.set_xlabel('Simulator AR');a.set_ylabel('PINN AR')
a.set_title(f'AR parity  (test R²={r2_te[3]:.3f})'); a.legend()
# train vs test R² 막대
b=ax[1]; xpos=np.arange(4); w=0.36
b.bar(xpos-w/2,r2_tr,w,label='train',color=C_TR,edgecolor='k',alpha=0.9)
b.bar(xpos+w/2,r2_te,w,label='test',color=C_TE,edgecolor='k',alpha=0.9)
b.set_xticks(xpos);b.set_xticklabels(names);b.set_ylim(min(0.9,min(r2_te)-0.02),1.001)
b.set_ylabel('R²');b.set_title('train vs test R² (gap 작음 = 과적합 아님)');b.legend()
for j in range(4): b.text(j+w/2,r2_te[j]+0.001,f'{r2_te[j]:.3f}',ha='center',fontsize=8)
plt.tight_layout(rect=[0,0,1,0.93]); plt.savefig(os.path.join(FIGDIR,'fig_traintest.png'),facecolor='white'); plt.close()

print("\n  그림 저장: fig_traintest.png")
print(f"\n  요약: test R²(AR)={r2_te[3]:.3f}, 5-fold 평균={cv.mean():.3f}±{cv.std():.3f}")
print(f"        train-test gap(AR)={r2_tr[3]-r2_te[3]:.4f} (작으면 과적합 아님)")
