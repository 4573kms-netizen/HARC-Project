"""
세련된 그림: leave-one-out 결과 — 실험 측정값을 시뮬 vs PINN이 얼마나 잘 따라가나
(값은 pinn_data_fusion_loo.py 결과를 그대로 사용, 재시뮬 불필요)
"""
import os, sys
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE=os.path.dirname(os.path.abspath(__file__)); FIGDIR=os.path.join(_HERE,'figures')
plt.rcParams.update({'figure.dpi':120,'savefig.dpi':180,'font.size':11,'font.family':'DejaVu Sans',
    'axes.titlesize':12,'axes.titleweight':'bold','axes.labelsize':10.5,
    'axes.grid':True,'grid.alpha':0.25,'grid.linewidth':0.8,
    'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#444',
    'figure.facecolor':'white','axes.facecolor':'#FBFBFC','legend.frameon':False,'legend.fontsize':9.5})
C_EXP,C_SIM,C_PINN='#111111','#F59E0B','#2563EB'

cf4=np.array([6,10,14,18])
data={
 'depth (nm)':   ([1369,1390,1298,1159],[1400.5,1374.4,1290.9,1158.4],[1404.2,1366.4,1306.5,1151.2]),
 'CD_top (nm)':  ([210,205,204,213],    [211.3,209.8,208.5,207.5],    [210.8,209.6,208.5,207.4]),
 'CD_bot (nm)':  ([74.4,44.2,53.4,73.6],[74.9,44.8,52.6,73.7],        [73.9,45.5,52.8,75.3]),
 'AR':           ([6.519,6.780,6.363,5.441],[6.628,6.551,6.191,5.583],[6.661,6.519,6.266,5.551]),
}

fig,axes=plt.subplots(2,2,figsize=(12.5,9))
fig.suptitle('Leave-one-experiment-out — Simulator vs PINN tracking REAL measurements\n'
             '(both within ±5% of SEM; PINN ≈ Simulator → no accuracy gain, simulator already excellent)',
             fontsize=13,fontweight='bold')
for ax,(label,(exp,sim,pinn)) in zip(axes.ravel(),data.items()):
    exp=np.array(exp); sim=np.array(sim); pinn=np.array(pinn)
    # ±5% 허용 밴드 (실험 기준)
    ax.fill_between(cf4, exp*0.95, exp*1.05, color='gray', alpha=0.12, label='±5% band')
    # 실험 = 검은 굵은 마커 (진실)
    ax.plot(cf4,exp,'-o',color=C_EXP,lw=1.4,ms=9,mfc='white',mew=2,label='Experiment (SEM)',zorder=5)
    # 시뮬
    ax.plot(cf4,sim,'--s',color=C_SIM,lw=1.6,ms=7,alpha=0.9,label='Simulator',zorder=4)
    # PINN
    ax.plot(cf4,pinn,'-^',color=C_PINN,lw=1.6,ms=7,alpha=0.9,label='PINN (sim+exp+physics)',zorder=4)
    # 평균 오차% 주석
    e_sim=100*np.mean(np.abs(sim-exp)/exp); e_pinn=100*np.mean(np.abs(pinn-exp)/exp)
    ax.set_title(f'{label}    |err|: sim {e_sim:.1f}%  ·  PINN {e_pinn:.1f}%')
    ax.set_xlabel('CF4 flow (sccm)   [Ar = 30 − CF4]'); ax.set_ylabel(label)
    ax.set_xticks(cf4)
    ax.legend(loc='best',fontsize=8.5)

plt.tight_layout(rect=[0,0,1,0.94])
out=os.path.join(FIGDIR,'fig_pinn_accuracy_v2.png')
plt.savefig(out,facecolor='white'); plt.close()
print(f"저장: {out}")
