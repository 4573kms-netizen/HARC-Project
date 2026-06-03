"""
=============================================================================
진짜 PINN: 시뮬 + 실험4점 + 물리식 융합 → 실험 예측 정확도 향상 검증
=============================================================================
[목적] "예측 정확도 향상" = 시뮬레이터보다 '실제 실험값'을 더 정확히 예측
[방법]
  - 정답 = 실제 SEM 실험 4점 (시뮬 아님)
  - PINN = 시뮬데이터(약한 가중) + 실험점(강한 가중) + 물리식 융합
  - 물리식: AR=depth/CD_top 항등(coupling) + 양수성 + taper (loss에 잔차)
[검증] leave-one-experiment-out
  - 실험 1점 빼고 (시뮬+나머지3점+물리)로 학습 → 뺀 점 예측
  - 시뮬레이터 오차 vs PINN 오차 비교 (실험값 기준 %)
  - 지표 = 실험 대비 오차(%) / RMSE.  CD_top 포함 (안 뺌)
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
C_SIM,C_PINN='#F59E0B','#2563EB'

mp=ModelParameters()
with open(os.path.join(_HERE,'harc_v2_calibrated_params_physics.json')) as f:
    for k,v in json.load(f).items():
        if hasattr(mp,k): setattr(mp,k,v)

# ── 실제 실험 4점 (reliable) : V=-1000, t=240, flow=30, P=10 ─────────────
EXP = [  # cf4, depth, cd_top, cd_bot  (실측)
    (6.0,  1369.0, 210.0, 74.4),
    (10.0, 1390.0, 205.0, 44.2),
    (14.0, 1298.0, 204.0, 53.4),
    (18.0, 1159.0, 213.0, 73.6),
]
def exp_cond(cf4):
    return ProcessConditions(cf4_flow=cf4,ar_flow=30.0-cf4,total_flow=30.0,
        v_bias=-1000.0,etch_time=240.0,pressure=10.0,source_power=250.0,substrate_temp=15.0,cd_initial=200.0)
# 실험 입력(3D: CF4,V_bias,time) & 타깃(depth,cd_top,cd_bot,AR)
Xexp_all=np.array([[c,-1000.0,240.0] for c,_,_,_ in EXP])
Yexp_all=np.array([[d,ct,cb,d/ct] for _,d,ct,cb in EXP])
names=['depth','CD_top','CD_bot','AR']

# ── 시뮬레이터 학습 데이터 (broad space) ─────────────────────────────────
LO=np.array([2.0,-1500.0,120.0]); HI=np.array([28.0,-200.0,600.0])
def make_cond(x):
    cf4,vb,t=x
    return ProcessConditions(cf4_flow=float(cf4),ar_flow=float(30.0-cf4),total_flow=30.0,
        v_bias=float(vb),etch_time=float(t),pressure=10.0,source_power=250.0,substrate_temp=15.0,cd_initial=200.0)
def sim_out(x):
    try:
        r=run_forward_simulation(make_cond(x),mp,verbose=False)
        return None if not np.isfinite(r.aspect_ratio) else [r.total_depth,r.cd_top,r.cd_bot,r.aspect_ratio]
    except Exception: return None

print("STEP 1: 시뮬 학습데이터 생성 ...")
Nsim=400
samp=LO+qmc.LatinHypercube(d=3,seed=42).random(Nsim)*(HI-LO)
Xs_l,Ys_l=[],[]
for s in samp:
    y=sim_out(s)
    if y: Xs_l.append(s); Ys_l.append(y)
Xsim=np.array(Xs_l); Ysim=np.array(Ys_l)
print(f"  시뮬 {len(Xsim)}개")

# ── PINN (backprop, data-fusion + physics) ───────────────────────────────
n_in,n_h,n_out=3,64,4
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

def fit(Xsim,Ysim,Xexp,Yexp,w_exp=40.0,lam=0.5,maxiter=2500,seed=1):
    Xm,Xs_=Xsim.mean(0),Xsim.std(0)+1e-8; Ym,Ys_=Ysim.mean(0),Ysim.std(0)+1e-8
    Xn_s=(Xsim-Xm)/Xs_; Yn_s=(Ysim-Ym)/Ys_
    Xn_e=(Xexp-Xm)/Xs_; Yn_e=(Yexp-Ym)/Ys_
    Ns,Ne=len(Xn_s),len(Xn_e)
    # 데이터 가중: 행별 weight (sim=1/Ns, exp=w_exp/Ne)
    Xall=np.vstack([Xn_s,Xn_e]); Yall=np.vstack([Yn_s,Yn_e])
    wrow=np.concatenate([np.full(Ns,1.0/Ns), np.full(Ne,w_exp/Ne)])[:,None]
    Nall=len(Xall)
    def phys(yn):
        y=yn*Ys_+Ym; depth,cdt,cdb,ar=y[:,0],y[:,1],y[:,2],y[:,3]; mcd=np.maximum(cdt,1.0)
        r1=(ar-depth/mcd)/Ys_[3]; r2=np.maximum(cdb-cdt,0)/Ys_[2]
        r3=np.maximum(-depth,0)/Ys_[0]; r4=np.maximum(1-ar,0)/Ys_[3]
        Lp=np.mean(r1**2+r2**2+r3**2+r4**2)
        gun=np.zeros_like(y)
        gun[:,3]+=2*r1/Ys_[3]; gun[:,0]+=2*r1*(-1.0/mcd)/Ys_[3]
        gun[:,1]+=np.where(cdt>1.0,2*r1*(depth/cdt**2)/Ys_[3],0.0)
        m2=(cdb-cdt)>0; gun[:,2]+=np.where(m2,2*r2/Ys_[2],0.0); gun[:,1]+=np.where(m2,-2*r2/Ys_[2],0.0)
        gun[:,0]+=np.where((-depth)>0,-2*r3/Ys_[0],0.0); gun[:,3]+=np.where((1-ar)>0,-2*r4/Ys_[3],0.0)
        return gun*Ys_/Nall, Lp
    def lg(th):
        yn,h1,h2=fwd(th,Xall)
        diff=yn-Yall
        Ld=np.sum(wrow*diff**2)/n_out
        gp,Lp=phys(yn)
        g_y=2*wrow*diff/n_out + lam*gp
        W1,b1,W2,b2,W3,b3=unpack(th)
        gW3=h2.T@g_y;gb3=g_y.sum(0); gh2=g_y@W3.T;gz2=gh2*(1-h2**2)
        gW2=h1.T@gz2;gb2=gz2.sum(0); gh1=gz2@W2.T;gz1=gh1*(1-h1**2)
        gW1=Xall.T@gz1;gb1=gz1.sum(0)
        return Ld+lam*Lp, pack(gW1,gb1,gW2,gb2,gW3,gb3)
    nth=n_in*n_h+n_h+n_h*n_h+n_h+n_h*n_out+n_out
    np.random.seed(seed); th0=np.random.randn(nth)*0.1
    r=minimize(lg,th0,jac=True,method='L-BFGS-B',options={'maxiter':maxiter,'ftol':1e-15,'gtol':1e-12})
    th=r.x
    def predict(Xev): return (fwd(th,(Xev-Xm)/Xs_)[0])*Ys_+Ym
    return predict

# ── 시뮬레이터 자체 오차 (각 실험점) ─────────────────────────────────────
print("STEP 2: 시뮬레이터의 실험 대비 오차 ...")
sim_pred=np.array([sim_out([c,-1000.0,240.0]) for c,_,_,_ in EXP])

# ── leave-one-experiment-out ─────────────────────────────────────────────
print("STEP 3: leave-one-experiment-out (시뮬 vs PINN) ...\n")
pinn_pred=np.zeros_like(Yexp_all)
for i in range(4):
    keep=[k for k in range(4) if k!=i]
    pred=fit(Xsim,Ysim,Xexp_all[keep],Yexp_all[keep],w_exp=40.0,maxiter=2500,seed=i+1)
    pinn_pred[i]=pred(Xexp_all[i:i+1])[0]

# ── 오차(%) 계산 & 출력 ──────────────────────────────────────────────────
def pct(pred): return 100.0*(pred-Yexp_all)/Yexp_all
sim_e=pct(sim_pred); pinn_e=pct(pinn_pred)
print(f"  held-out 실험점별 |오차%| (시뮬 → PINN):")
print(f"  {'CF4':>5} {'output':>7} {'exp':>8} {'sim':>8} {'PINN':>8} {'|sim%|':>7} {'|PINN%|':>8}")
for i in range(4):
    for j in range(4):
        print(f"  {EXP[i][0]:>5.0f} {names[j]:>7} {Yexp_all[i,j]:>8.1f} {sim_pred[i,j]:>8.1f} {pinn_pred[i,j]:>8.1f}"
              f" {abs(sim_e[i,j]):>6.1f}% {abs(pinn_e[i,j]):>7.1f}%")
mae_sim=np.abs(sim_e).mean(); mae_pinn=np.abs(pinn_e).mean()
print(f"\n  전체 평균 |오차%|:  시뮬={mae_sim:.2f}%   PINN={mae_pinn:.2f}%")
print(f"  → {'PINN이 더 정확 ✓' if mae_pinn<mae_sim else 'PINN이 개선 못함'} (차이 {mae_sim-mae_pinn:+.2f}%p)")

# ── 그림: 출력별 평균 |오차%| 시뮬 vs PINN ────────────────────────────────
fig,ax=plt.subplots(1,2,figsize=(13,5))
fig.suptitle('True PINN — accuracy vs REAL experiments (leave-one-out)',fontsize=13,fontweight='bold')
# (a) 출력별 평균 |오차%|
sim_by=np.abs(sim_e).mean(0); pinn_by=np.abs(pinn_e).mean(0)
xp=np.arange(4); w=0.36
ax[0].bar(xp-w/2,sim_by,w,label='Simulator',color=C_SIM,edgecolor='k',alpha=0.9)
ax[0].bar(xp+w/2,pinn_by,w,label='PINN (sim+exp+physics)',color=C_PINN,edgecolor='k',alpha=0.9)
ax[0].set_xticks(xp);ax[0].set_xticklabels(names);ax[0].set_ylabel('mean |error %| vs experiment')
ax[0].set_title('Per-output error vs real experiments');ax[0].legend()
for j in range(4):
    ax[0].text(j-w/2,sim_by[j]+0.05,f'{sim_by[j]:.1f}',ha='center',fontsize=8)
    ax[0].text(j+w/2,pinn_by[j]+0.05,f'{pinn_by[j]:.1f}',ha='center',fontsize=8)
# (b) held-out 점별 전체 평균 |오차%|
sim_pt=np.abs(sim_e).mean(1); pinn_pt=np.abs(pinn_e).mean(1)
xp2=np.arange(4)
ax[1].bar(xp2-w/2,sim_pt,w,label='Simulator',color=C_SIM,edgecolor='k',alpha=0.9)
ax[1].bar(xp2+w/2,pinn_pt,w,label='PINN',color=C_PINN,edgecolor='k',alpha=0.9)
ax[1].set_xticks(xp2);ax[1].set_xticklabels([f'CF4={int(EXP[i][0])}' for i in range(4)])
ax[1].set_ylabel('mean |error %|');ax[1].set_title('Per held-out experiment');ax[1].legend()
plt.tight_layout(rect=[0,0,1,0.93]); plt.savefig(os.path.join(FIGDIR,'fig_pinn_accuracy.png'),facecolor='white'); plt.close()
print(f"\n  그림 저장: fig_pinn_accuracy.png")
