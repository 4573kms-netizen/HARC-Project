"""
=============================================================================
정확한 파이프라인: 시뮬 → PINN → (PINN이 GPR 데이터 생성) → GPR → BO → AR=10
=============================================================================
전제: 시뮬레이터는 시뮬레이터일 뿐(역산 기능 없음). PINN→GPR→BO로 AR=10 예측.
흐름:
  1. 시뮬 1500개(±4% 측정노이즈) → PINN 학습 (backprop, 물리손실)
       → R²는 노이즈 때문에 ~0.9X (1.0 아님)
  2. 학습된 PINN(검증된 surrogate)으로 GPR 학습용 데이터 생성
  3. GPR 학습 (PINN 데이터 기반, 불확실도 σ)
  4. BO(GPR + EI) → PINN을 평가자로 |AR-10| 최소화 → AR=10 조건 예측
입력 5D: CF4, Ar, V_bias, time, pressure  (total=CF4+Ar)
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
from harc_etch_simulator_v2 import ModelParameters, ProcessConditions, run_forward_simulation

_HERE=os.path.dirname(os.path.abspath(__file__)); FIGDIR=os.path.join(_HERE,'figures')
os.makedirs(FIGDIR,exist_ok=True)
plt.rcParams.update({'figure.dpi':120,'savefig.dpi':175,'font.size':11,'font.family':'DejaVu Sans',
    'axes.titlesize':12,'axes.titleweight':'bold','axes.labelsize':10.5,'axes.grid':True,'grid.alpha':0.25,
    'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'#FBFBFC',
    'legend.frameon':False,'legend.fontsize':9})
C_PINN,C_GPR,C_BO,C_SIM,C_RED='#2563EB','#7C3AED','#059669','#F59E0B','#DC2626'

mp=ModelParameters()
with open(os.path.join(_HERE,'harc_v2_calibrated_params_physics.json')) as f:
    for k,v in json.load(f).items():
        if hasattr(mp,k): setattr(mp,k,v)

LO=np.array([2.0,5.0,-1500.0,120.0,4.0]); HI=np.array([28.0,40.0,-200.0,600.0,40.0])
TARGET_AR=10.0; NOISE=0.04
def make_cond(x):
    cf4,ar,vb,t,p=x
    return ProcessConditions(cf4_flow=float(cf4),ar_flow=float(ar),total_flow=float(cf4+ar),
        v_bias=float(vb),etch_time=float(t),pressure=float(p),source_power=250.0,substrate_temp=15.0,cd_initial=200.0)
def sim_out(x):
    try:
        r=run_forward_simulation(make_cond(x),mp,verbose=False)
        return None if not np.isfinite(r.aspect_ratio) else [r.total_depth,r.cd_top,r.cd_bot,r.aspect_ratio]
    except Exception: return None
def to_unit(X): return (X-LO)/(HI-LO)
def from_unit(U): return LO+U*(HI-LO)

# =============================================================================
# STEP 1: 시뮬 데이터(+노이즈) → PINN
# =============================================================================
print("="*60); print("  STEP 1: 시뮬 → PINN (±4% 노이즈)"); print("="*60)
N=1500
samp=LO+qmc.LatinHypercube(d=5,seed=42).random(N)*(HI-LO)
Xl,Yl=[],[]
for s in samp:
    y=sim_out(s)
    if y: Xl.append(s); Yl.append(y)
X=np.array(Xl); Yc=np.array(Yl)
Yn_noise=Yc*(1.0+np.random.RandomState(123).normal(0,NOISE,Yc.shape))   # 측정노이즈
print(f"  시뮬 {len(X)}개 (+노이즈)")

n_in,n_h,n_out=5,64,4; LAM=0.3
def unpack(th):
    i=0
    W1=th[i:i+n_in*n_h].reshape(n_in,n_h);i+=n_in*n_h
    b1=th[i:i+n_h];i+=n_h
    W2=th[i:i+n_h*n_h].reshape(n_h,n_h);i+=n_h*n_h
    b2=th[i:i+n_h];i+=n_h
    W3=th[i:i+n_h*n_out].reshape(n_h,n_out);i+=n_h*n_out
    b3=th[i:i+n_out]; return W1,b1,W2,b2,W3,b3
def pack(*a): return np.concatenate([x.ravel() for x in a])
def fwd(th,x):
    W1,b1,W2,b2,W3,b3=unpack(th)
    z1=x@W1+b1;h1=np.tanh(z1); z2=h1@W2+b2;h2=np.tanh(z2); yn=h2@W3+b3
    return yn,h1,h2

# train/test 분리 (믿을 만한 R² 보고용)
rng=np.random.RandomState(0); idx=rng.permutation(len(X)); nc=int(0.8*len(X))
tr,te=idx[:nc],idx[nc:]
def fit_pinn(Xtr,Ytr,maxiter=2500):
    Xm,Xs=Xtr.mean(0),Xtr.std(0)+1e-8; Ym,Ys=Ytr.mean(0),Ytr.std(0)+1e-8
    Xn=(Xtr-Xm)/Xs; Yn=(Ytr-Ym)/Ys; Ntr=len(Xn)
    def phys(yn):
        y=yn*Ys+Ym; d,ct,cb,ar=y[:,0],y[:,1],y[:,2],y[:,3]; mcd=np.maximum(ct,1.0)
        r1=(ar-d/mcd)/Ys[3]; r3=np.maximum(-d,0)/Ys[0]; r4=np.maximum(1-ar,0)/Ys[3]
        Lp=np.mean(r1**2+r3**2+r4**2); gun=np.zeros_like(y)
        gun[:,3]+=2*r1/Ys[3]; gun[:,0]+=2*r1*(-1.0/mcd)/Ys[3]
        gun[:,1]+=np.where(ct>1.0,2*r1*(d/ct**2)/Ys[3],0.0)
        gun[:,0]+=np.where((-d)>0,-2*r3/Ys[0],0.0); gun[:,3]+=np.where((1-ar)>0,-2*r4/Ys[3],0.0)
        return gun*Ys/Ntr,Lp
    def lg(th):
        yn,h1,h2=fwd(th,Xn); Ld=np.mean((yn-Yn)**2); gp,Lp=phys(yn)
        g=2*(yn-Yn)/(Ntr*n_out)+LAM*gp
        W1,b1,W2,b2,W3,b3=unpack(th)
        gW3=h2.T@g;gb3=g.sum(0); gh2=g@W3.T;gz2=gh2*(1-h2**2)
        gW2=h1.T@gz2;gb2=gz2.sum(0); gh1=gz2@W2.T;gz1=gh1*(1-h1**2)
        gW1=Xn.T@gz1;gb1=gz1.sum(0)
        return Ld+LAM*Lp,pack(gW1,gb1,gW2,gb2,gW3,gb3)
    nth=n_in*n_h+n_h+n_h*n_h+n_h+n_h*n_out+n_out
    np.random.seed(1); th0=np.random.randn(nth)*0.1
    r=minimize(lg,th0,jac=True,method='L-BFGS-B',options={'maxiter':maxiter,'ftol':1e-15})
    th=r.x
    return lambda Xe:(fwd(th,(Xe-Xm)/Xs)[0])*Ys+Ym
pinn=fit_pinn(X[tr],Yn_noise[tr])
yp_te=pinn(X[te])
def R2(yt,yp): return 1-np.sum((yt-yp)**2)/(np.sum((yt-yt.mean())**2)+1e-12)
r2_ar=R2(Yn_noise[te,3],yp_te[:,3])
print(f"  PINN test R²(AR)={r2_ar:.3f}  (노이즈 때문에 1.0 아님 = 정상)")
def pinn_AR(x): return pinn(np.atleast_2d(x))[0,3]

# =============================================================================
# STEP 2: PINN이 GPR 학습용 데이터 생성
# =============================================================================
print("\n"+"="*60); print("  STEP 2: PINN → GPR 학습 데이터 생성"); print("="*60)
M=250
Xg=from_unit(qmc.LatinHypercube(d=5,seed=77).random(M))
yg=pinn(Xg)[:,3]   # PINN이 예측한 AR
print(f"  PINN으로 {M}개 (조건→AR) 생성")

# =============================================================================
# STEP 3 & 4: GPR 학습 + BO (GPR + EI, PINN 평가자) → AR=10
# =============================================================================
print("\n"+"="*60); print("  STEP 3-4: GPR 학습 + BO 탐색"); print("="*60)
kern=ConstantKernel(1.0)*Matern(length_scale=[0.3]*5,nu=2.5)+WhiteKernel(1e-3)
def obj(x): return abs(pinn_AR(x)-TARGET_AR)
# 초기 GPR 데이터 = PINN 생성 데이터의 목적함수값
Xb=Xg.copy(); yb=np.abs(yg-TARGET_AR)
bo_hist=[yb.min()]
for it in range(40):
    g=GaussianProcessRegressor(kernel=kern,normalize_y=True,n_restarts_optimizer=1,random_state=0)
    g.fit(to_unit(Xb),yb)
    Uc=qmc.LatinHypercube(d=5,seed=200+it).random(4000); mu,sd=g.predict(Uc,return_std=True); sd=np.maximum(sd,1e-9)
    fb=yb.min(); z=(fb-mu-0.01)/sd; ei=np.maximum((fb-mu-0.01)*norm.cdf(z)+sd*norm.pdf(z),0)
    xn_=from_unit(Uc[np.argmax(ei)]); Xb=np.vstack([Xb,xn_]); yb=np.append(yb,obj(xn_)); bo_hist.append(yb.min())
x_best=Xb[np.argmin(yb)]
# 최종 GPR (불확실도용)
gpr=GaussianProcessRegressor(kernel=kern,normalize_y=True,n_restarts_optimizer=2,random_state=0)
gpr.fit(to_unit(Xb), np.array([pinn_AR(x) for x in Xb]))   # 조건→AR GPR
ar_pinn=pinn_AR(x_best); ar_gpr,sd_gpr=gpr.predict(to_unit(x_best).reshape(1,-1),return_std=True)
r_sim=sim_out(x_best)
print(f"  예측 조건: CF4={x_best[0]:.1f} Ar={x_best[1]:.1f} V_bias={x_best[2]:.0f} time={x_best[3]:.0f} P={x_best[4]:.1f}")
print(f"  PINN AR={ar_pinn:.2f} | GPR AR={ar_gpr[0]:.2f}±{sd_gpr[0]:.2f} | (참고)시뮬 AR={r_sim[3]:.2f}")

# =============================================================================
# 플롯 데이터 저장 (재플롯용) + 세련된 그림
# =============================================================================
vb=np.linspace(LO[2],HI[2],80); Xsl=np.tile(x_best,(80,1)); Xsl[:,2]=vb
mu_s,sd_s=gpr.predict(to_unit(Xsl),return_std=True)
np.savez(os.path.join(_HERE,'_pipeline_plotdata.npz'),
         par_x=Yn_noise[te,3], par_y=yp_te[:,3], r2_ar=r2_ar,
         vb=vb, mu_s=mu_s, sd_s=sd_s, bo_hist=np.array(bo_hist),
         ar_pinn=ar_pinn, ar_gpr=ar_gpr[0], sd_gpr=sd_gpr[0], sim_ar=r_sim[3], x_best=x_best)

# =============================================================================
# 개별 figure 저장 (각 그래프를 하나의 파일로 — 한 장에 여러 개 안 넣음)
# =============================================================================
def save_one(fname):
    plt.tight_layout(); plt.savefig(os.path.join(FIGDIR,fname),facecolor='white',bbox_inches='tight',dpi=180); plt.close()

# (1) PINN parity
plt.figure(figsize=(6.2,5.8))
lim=[min(Yn_noise[te,3].min(),yp_te[:,3].min())-0.3, max(Yn_noise[te,3].max(),yp_te[:,3].max())+0.3]
plt.plot(lim,lim,'--',color='#94A3B8',lw=1.4)
plt.scatter(Yn_noise[te,3],yp_te[:,3],s=24,alpha=0.55,color=C_PINN,edgecolor='white',lw=0.4)
plt.xlim(lim);plt.ylim(lim);plt.gca().set_aspect('equal')
plt.xlabel('Simulator AR (with ±4% noise)');plt.ylabel('PINN predicted AR')
plt.text(0.05,0.92,f'R² = {r2_ar:.3f}',transform=plt.gca().transAxes,fontsize=13,fontweight='bold',color=C_PINN,
         bbox=dict(boxstyle='round,pad=0.4',fc='white',ec=C_PINN))
plt.title('PINN surrogate — held-out test')
save_one('fig_pinn_parity.png')

# (2) BO convergence
plt.figure(figsize=(6.6,5.2))
bo=np.array(bo_hist)
plt.plot(range(len(bo)),bo,'-',color=C_BO,lw=2.2); plt.scatter(range(len(bo)),bo,s=20,color=C_BO)
conv=int(np.argmax(bo<=bo.min()+1e-6))
plt.scatter([conv],[bo[conv]],s=130,marker='o',facecolor='white',edgecolor=C_BO,lw=2)
plt.annotate(f'converged @ iter {conv}',xy=(conv,bo[conv]),xytext=(conv+6,bo.max()*0.55),
             fontsize=10,color=C_BO,arrowprops=dict(arrowstyle='->',color=C_BO))
plt.axhline(0,color=C_RED,ls='--',lw=1.2)
plt.xlabel('BO iteration');plt.ylabel('best  |AR − 10|');plt.ylim(bottom=-0.02)
plt.title('Bayesian optimization convergence')
save_one('fig_bo_convergence.png')

# (3) prediction
plt.figure(figsize=(6.0,5.4))
vals=[ar_pinn,ar_gpr[0],r_sim[3]]; labs=['PINN','GPR','Simulator\n(check)']
bars=plt.bar(labs,vals,color=[C_PINN,C_GPR,C_SIM],edgecolor='white',lw=1.5,alpha=0.92,width=0.62)
plt.errorbar([1],[ar_gpr[0]],yerr=[2*sd_gpr[0]],fmt='none',ecolor='#1E293B',capsize=6,lw=1.5)
plt.axhline(10,color=C_RED,ls='--',lw=1.6,label='target AR = 10')
plt.ylabel('predicted AR');plt.ylim(0,max(vals)*1.18);plt.legend(loc='lower right')
plt.title('Predicted AR at optimal condition')
for bb,v in zip(bars,vals): plt.text(bb.get_x()+bb.get_width()/2,v+max(vals)*0.02,f'{v:.2f}',ha='center',fontweight='bold',fontsize=11)
save_one('fig_prediction.png')

# (4) 5개 변수 단면 — 각각 개별 파일
VARNAMES=['CF4 (sccm)','Ar (sccm)','V_bias (V)','time (s)','pressure (mTorr)']
VTAG=['CF4','Ar','Vbias','time','pressure']
for i in range(5):
    plt.figure(figsize=(6.2,5.0))
    grid=np.linspace(LO[i],HI[i],80); Xsl=np.tile(x_best,(80,1)); Xsl[:,i]=grid
    mu,sd=gpr.predict(to_unit(Xsl),return_std=True)
    plt.fill_between(grid,mu-2*sd,mu+2*sd,color=C_GPR,alpha=0.18,label='GPR ±2σ')
    plt.plot(grid,mu,color=C_GPR,lw=2.4,label='GPR mean')
    plt.axhline(10,color=C_RED,ls=':',lw=1.5,label='target AR=10')
    plt.scatter([x_best[i]],[ar_gpr[0]],s=160,marker='*',color=C_RED,edgecolor='white',lw=1,zorder=5,label='optimum')
    plt.xlabel(VARNAMES[i]);plt.ylabel('AR');plt.title(f'GPR: AR vs {VTAG[i]}  (others fixed at optimum)')
    plt.legend(fontsize=8.5)
    save_one(f'fig_slice_{VTAG[i]}.png')

print("\n  개별 그림 8개 저장: fig_pinn_parity, fig_bo_convergence, fig_prediction, fig_slice_{CF4,Ar,Vbias,time,pressure}")
