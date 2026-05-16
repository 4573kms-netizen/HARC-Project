# -*- coding: utf-8 -*-
"""HARC Simulator PPT Generator"""
import matplotlib
matplotlib.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
import numpy as np, os, sys

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

OUTDIR = r'C:\Users\4573k\Desktop\HARC_simulation_Claude'

def rgb(h):
    h = h.lstrip('#')
    return RGBColor(int(h[0:2],16), int(h[2:4],16), int(h[4:6],16))

# ── FLOWCHART ─────────────────────────────────────────────────────────────────
def make_flowchart(path):
    fig, ax = plt.subplots(figsize=(7.2, 12.0))
    ax.set_xlim(-0.6, 7.8); ax.set_ylim(-0.3, 12.5)
    ax.axis('off'); ax.set_facecolor('white'); fig.patch.set_facecolor('white')
    CX = 3.6; BW = 6.8

    # ── 회색-슬레이트 톤 색상 팔레트 ────────────────────────────────────────
    C_IN  = '#1E293B'   # 최어두운 슬레이트 (입력/출력)
    C_P1  = '#1E3A5F'   # 네이비 블루-그레이 (① 플라즈마)
    C_P2  = '#243448'   # 슬레이트 (② 쉬스)
    C_P3  = '#2D3F52'   # 중간 슬레이트 (③ 전달)
    C_P4  = '#374151'   # 차콜 (④ 표면)
    C_P5  = '#1A3040'   # 딥 슬레이트 (⑤ 시간)
    C_DIA = '#5A6E82'   # 밝은 슬레이트 (결정)
    C_ARR = '#64748B'   # 화살표/루프

    def rbox(x, y, w, h, t1, fill='#1E3A5F'):
        ax.add_patch(FancyBboxPatch((x-w/2+0.06, y-h/2-0.06), w, h,
            boxstyle='round,pad=0.10', facecolor='#94A3B8', edgecolor='none',
            linewidth=0, zorder=2, alpha=0.28))
        ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
            boxstyle='round,pad=0.10', facecolor=fill, edgecolor='#94A3B8',
            linewidth=1.2, zorder=3))
        ax.text(x, y, t1, ha='center', va='center', fontsize=12.0,
                fontweight='bold', color='white', zorder=4, multialignment='center')

    def pbox(x, y, w, h, t1, fill='#1E293B', sk=0.30):
        pts = np.array([[x-w/2+sk, y+h/2], [x+w/2+sk, y+h/2],
                        [x+w/2-sk, y-h/2], [x-w/2-sk, y-h/2]])
        shadow = pts + np.array([0.06, -0.06])
        ax.add_patch(Polygon(shadow, facecolor='#94A3B8', edgecolor='none',
                              linewidth=0, zorder=2, alpha=0.28))
        ax.add_patch(Polygon(pts, facecolor=fill, edgecolor='#94A3B8',
                              linewidth=1.2, zorder=3))
        ax.text(x, y, t1, ha='center', va='center', fontsize=12.0,
                fontweight='bold', color='white', zorder=4)

    def dia(x, y, hw, hh, text, fill='#5A6E82'):
        pts = np.array([[x, y+hh], [x+hw, y], [x, y-hh], [x-hw, y]])
        shadow = pts + np.array([0.06, -0.06])
        ax.add_patch(Polygon(shadow, facecolor='#94A3B8', edgecolor='none',
                              linewidth=0, zorder=2, alpha=0.28))
        ax.add_patch(Polygon(pts, facecolor=fill, edgecolor='#94A3B8',
                              linewidth=1.8, zorder=3))
        ax.text(x, y, text, ha='center', va='center', fontsize=12,
                fontweight='bold', color='white', zorder=4)

    def arr(x1, y1, x2, y2, col=None):
        col = col or C_ARR
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle='->', color=col, lw=2.2,
                            mutation_scale=20), zorder=2)

    def label(x, y, text, ha='left'):
        ax.text(x, y, text, ha=ha, va='center', fontsize=10,
                fontweight='bold', color='#475569',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor='#CBD5E1', linewidth=1.0))

    # ── y 위치 계산 (박스 높이 줄임: 제목만, 세부 설명 없음) ─────────────
    HI=0.68; HP=0.72; HDIA=0.90; G=0.42
    yi   = 11.8
    yp1  = yi  - HI/2 - G - HP/2
    yp2  = yp1 - HP/2 - G - HP/2
    yp3  = yp2 - HP/2 - G - HP/2
    yp4  = yp3 - HP/2 - G - HP/2
    yp5  = yp4 - HP/2 - G - HP/2
    yd   = yp5 - HP/2 - G - HDIA
    yo   = yd  - HDIA - G - HI/2

    # ── 박스 (제목만, 세부 설명은 PPT 슬라이드 우측에 별도 표시) ────────
    pbox(CX, yi,  BW, HI, '공정 조건 입력  (Process Conditions)', fill=C_IN)
    rbox(CX, yp1, BW, HP, '① 0-D 전체 플라즈마 모델',            fill=C_P1)
    rbox(CX, yp2, BW, HP, '② 쉬스 / 이온 에너지 모델',           fill=C_P2)
    rbox(CX, yp3, BW, HP, '③ 피처 내부 입자 전달  [for each z]', fill=C_P3)
    rbox(CX, yp4, BW, HP, '④ 표면 반응 모델',                    fill=C_P4)
    rbox(CX, yp5, BW, HP, '⑤ 시간 적분  (Δt = 0.5 s)',          fill=C_P5)
    dia(CX, yd, 2.2, HDIA, '식각 완료?',                         fill=C_DIA)
    pbox(CX, yo, BW, HI, '시뮬레이션 결과  (Simulation Result)',  fill=C_IN)

    # ── 수직 화살표 ──────────────────────────────────────────────────────────
    arr(CX, yi-HI/2,  CX, yp1+HP/2)
    arr(CX, yp1-HP/2, CX, yp2+HP/2)
    arr(CX, yp2-HP/2, CX, yp3+HP/2)
    arr(CX, yp3-HP/2, CX, yp4+HP/2)
    arr(CX, yp4-HP/2, CX, yp5+HP/2)
    arr(CX, yp5-HP/2, CX, yd+HDIA)
    arr(CX, yd-HDIA,  CX, yo+HI/2)
    label(CX+0.15, (yd-HDIA+yo+HI/2)/2, '예')

    # 아니오 → 루프 (왼쪽 → 위 → ③ transport 박스)
    lx_dia  = CX - 2.2
    lx_loop = -0.28
    ax.annotate('', xy=(lx_loop, yd), xytext=(lx_dia, yd),
        arrowprops=dict(arrowstyle='-', color=C_ARR, lw=2.2), zorder=2)
    label(lx_dia - 0.05, yd + 0.24, '아니오', ha='right')
    ax.plot([lx_loop, lx_loop], [yd, yp3], color=C_ARR, lw=2.2, zorder=2)
    arr(lx_loop, yp3, CX-BW/2, yp3)

    plt.tight_layout(pad=0.1)
    plt.savefig(path, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(); print(f'  flowchart → {path}')


# ── CALIBRATION ACCURACY FIGURE ───────────────────────────────────────────────
def make_cal_figure(path):
    fig,axes=plt.subplots(1,3,figsize=(11,3.6)); fig.patch.set_facecolor('white')
    labels=['6/24','10/20','14/16','18/12']
    exp_d=[1369,1390,1298,1159]; sim_d=[1379.1,1359.7,1284.4,1163.9]
    exp_t=[210.0,205.0,204.0,213.0]; sim_t=[209.9,207.8,206.2,204.9]
    exp_b=[74.4,44.2,53.4,73.6];   sim_b=[54.6,54.4,57.0,63.4]
    x=np.arange(4); w=0.36
    for ax,(title,exp,sim) in zip(axes,[('식각 깊이 [nm]',exp_d,sim_d),
                                         ('상단 CD [nm]',exp_t,sim_t),
                                         ('하단 CD [nm]',exp_b,sim_b)]):
        ax.bar(x-w/2,exp,width=w,label='실험',color='#94A3B8',edgecolor='white',linewidth=0.5)
        ax.bar(x+w/2,sim,width=w,label='시뮬',color='#2563EB',alpha=0.9,edgecolor='white',linewidth=0.5)
        for xi,(e,s) in enumerate(zip(exp,sim)):
            err=100*(s-e)/e
            c='#16A34A' if abs(err)<5 else '#D97706' if abs(err)<15 else '#DC2626'
            ax.text(xi+w/2,s+max(exp)*0.01,f'{err:+.1f}%',
                    ha='center',va='bottom',fontsize=6.5,color=c,fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=8)
        ax.set_title(title,fontsize=9.5,fontweight='bold',color='#1E3A5F',pad=4)
        ax.legend(fontsize=7.5); ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False); ax.grid(axis='y',alpha=0.3,lw=0.5)
        ax.set_facecolor('#FAFAFA'); ax.tick_params(labelsize=8)
    fig.suptitle('캘리브레이션 정확도  (4점:  CF4/Ar = 6/24 · 10/20 · 14/16 · 18/12)',
                 fontsize=10,fontweight='bold',color='#1E3A5F',y=1.02)
    plt.tight_layout(pad=0.5)
    plt.savefig(path,dpi=200,bbox_inches='tight',facecolor='white')
    plt.close(); print(f'  cal figure → {path}')


# ── AR SWEEP FIGURE ───────────────────────────────────────────────────────────
def make_ar_sweep(path):
    cf4=[2,4,6,8,10,12,14,16,18,20,22,24,26,28]
    ar =[5.961,6.409,6.571,6.600,6.543,6.416,6.230,
         5.986,5.679,5.304,4.842,4.267,3.527,2.497]
    fig,ax=plt.subplots(figsize=(6.0,3.8)); fig.patch.set_facecolor('white')
    ax.plot(cf4,ar,'o-',color='#2563EB',lw=2,ms=6,
            markerfacecolor='white',markeredgewidth=2,label='시뮬레이션')
    ax.axvline(7.5,color='#DC2626',ls='--',lw=1.8,alpha=0.8)
    ax.scatter([7.5],[6.603],s=110,color='#DC2626',zorder=5)
    ax.text(7.7,6.63,'최적\nCF4=7.50 sccm\nAR=6.603',
            fontsize=8.5,color='#DC2626',fontweight='bold',va='bottom')
    ax.scatter([6,10,14,18],[6.519,6.780,6.363,5.441],
               s=80,color='#374151',zorder=5,marker='D',label='실험값')
    ax.set_xlabel('CF4 유량 [sccm]  (Ar = 30 − CF4)',fontsize=9)
    ax.set_ylabel('Aspect Ratio (AR)',fontsize=9)
    ax.set_title('AR vs CF4/Ar 비율  (V_bias=−1000V, 250W, 10mTorr, 240s)',
                 fontsize=9.5,fontweight='bold',color='#1E3A5F')
    ax.legend(fontsize=8.5); ax.set_xlim(0,30); ax.set_ylim(1.5,8.0)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3); ax.set_facecolor('#FAFAFA')
    plt.tight_layout()
    plt.savefig(path,dpi=200,bbox_inches='tight',facecolor='white')
    plt.close(); print(f'  AR sweep → {path}')


# ── PPT 조립 ─────────────────────────────────────────────────────────────────
def build_ppt(fc,cal,ars,opt_fig,out):
    prs=Presentation()
    prs.slide_width=Inches(13.33); prs.slide_height=Inches(7.5)
    blank=prs.slide_layouts[6]

    def txb(slide,text,l,t,w,h,fs=10,bold=False,col='#111827',
            align=PP_ALIGN.LEFT,fname='Malgun Gothic'):
        tb=slide.shapes.add_textbox(Inches(l),Inches(t),Inches(w),Inches(h))
        tf=tb.text_frame; tf.word_wrap=True
        lines=text.split('\n')
        for i,ln in enumerate(lines):
            p=tf.paragraphs[0] if i==0 else tf.add_paragraph()
            p.alignment=align
            run=p.add_run(); run.text=ln
            run.font.size=Pt(fs); run.font.bold=bold
            run.font.color.rgb=rgb(col); run.font.name=fname
        return tb

    def rect(slide,l,t,w,h,fc,lc=None):
        sh=slide.shapes.add_shape(1,Inches(l),Inches(t),Inches(w),Inches(h))
        sh.fill.solid(); sh.fill.fore_color.rgb=rgb(fc)
        if lc: sh.line.color.rgb=rgb(lc)
        else:  sh.line.fill.background()
        return sh

    def img(slide,path,l,t,w,h=None):
        if h: return slide.shapes.add_picture(path,Inches(l),Inches(t),Inches(w),Inches(h))
        return slide.shapes.add_picture(path,Inches(l),Inches(t),Inches(w))

    def header(slide,title,sub='',bg='#1E3A5F'):
        rect(slide,0,0,13.33,1.05,bg)
        txb(slide,title,0.28,0.07,12.5,0.55,fs=22,bold=True,col='#FFFFFF')
        if sub: txb(slide,sub,0.28,0.58,12.5,0.42,fs=11,col='#BFDBFE')

    # ── SLIDE 1: 표지 ─────────────────────────────────────────────────────────
    s1=prs.slides.add_slide(blank)
    rect(s1,0,0,13.33,7.5,'#1E3A5F')
    rect(s1,0,2.85,13.33,1.9,'#2563EB')
    txb(s1,'HARC 식각 공정',0.5,1.1,12.3,0.9,fs=34,bold=True,col='#BFDBFE',align=PP_ALIGN.CENTER)
    txb(s1,'물리 기반 시뮬레이터 및 역최적화',0.5,2.0,12.3,0.75,fs=20,col='#FFFFFF',align=PP_ALIGN.CENTER)
    txb(s1,'Physics-Based Forward Simulation & Inverse Optimization',
        0.5,3.05,12.3,0.55,fs=14,col='#E0F2FE',align=PP_ALIGN.CENTER,fname='Calibri')
    txb(s1,'CF4/Ar 혼합 플라즈마  |  Si HARC 식각  |  250W · 10mTorr · −1000V · 240s',
        0.5,3.65,12.3,0.5,fs=11.5,col='#BFDBFE',align=PP_ALIGN.CENTER)
    txb(s1,'아주대학교 화학공학과  |  2026. 05',3.0,6.4,7.33,0.5,
        fs=11,col='#94A3B8',align=PP_ALIGN.CENTER)

    # ── SLIDE 2: 1-1 시뮬레이터 개요 ────────────────────────────────────────
    s2=prs.slides.add_slide(blank)
    header(s2,'1-1.  시뮬레이터 개요','순방향 시뮬레이션 알고리즘  —  물리 기반 다중 스텝 계산 흐름')
    img(s2,fc,0.2,1.1,5.0)

    rx=5.45; rw=7.65
    items=[
        ('#1E3A5F','① 0-D 전체 플라즈마 모델',
         'CF4/Ar 가스에 전기를 켜면 불소(F)·이온·폴리머 입자가 얼마나 많이 생기는지 계산합니다.'),
        ('#243448','② 쉬스 / 이온 에너지 모델',
         '이온이 바이어스 전압에 의해 얼마나 빠른 총알처럼 가속되어 날아오는지 계산합니다.'),
        ('#2D3F52','③ 피처 내부 입자 전달',
         '좁고 깊은 구멍 속으로 입자가 깊이별로 얼마나 파고 들어갈 수 있는지 계산합니다.'),
        ('#374151','④ 표면 반응 모델',
         '날아온 입자가 실리콘(Si) 벽에 부딪혀 얼마나 깎이고, 폴리머가 얼마나 쌓이는지 계산합니다.'),
        ('#1A3040','⑤ 마스크 개구부 진화',
         '식각이 진행될수록 구멍 입구(마스크)의 크기가 어떻게 변하는지 0.5초마다 추적합니다.'),
    ]
    y=1.12
    for color,title,body in items:
        rect(s2,rx,y,rw,0.30,color)
        txb(s2,title,rx+0.10,y+0.04,rw-0.16,0.26,fs=9.5,bold=True,col='#FFFFFF')
        txb(s2,body, rx+0.10,y+0.34,rw-0.16,0.44,fs=9.0,col='#1F2937')
        y+=0.86

    txb(s2,'캘리브레이션 대상: 20 파라미터  |  시간 적분: Explicit Euler Δt=0.5s  |  공간 격자: Δz=20nm',
        0.2,7.12,13.0,0.33,fs=7.5,col='#6B7280')

    # ── SLIDE 3: 1-2 캘리브레이션 ───────────────────────────────────────────
    s3=prs.slides.add_slide(blank)
    header(s3,'1-2.  캘리브레이션',
           '실험 4점 기반 파라미터 최적화  (log10-space TRF 2단계)')

    steps=[
        ('1','초기값 설정',
         '물리적 추정치로 20개 파라미터 초기화  (K_chem, K_ie, K_sput, sigma_iad, ...)'),
        ('2','순방향 시뮬레이션 × 4조건',
         'CF4/Ar = 6/24 · 10/20 · 14/16 · 18/12 각각에 대해 Depth · CD_top · CD_bot · AR 계산'),
        ('3','Residual (잔차) 계산',
         '시뮬값−실험값 차이를 가중치 적용  →  W_AR=2.5 (최우선) · W_Depth=1.2 · W_top=1.0 · W_bot=0.5'),
        ('4','log10-space TRF 최적화',
         'log 스케일 변환 후 Bounded TRF 최소화  →  Stage 1 (ftol=1e-3)  →  Stage 2 (ftol=1e-6)'),
        ('5','수렴 확인 & 파라미터 저장',
         '20 파라미터 · 16 방정식 (DOF=−4) 수렴 후  →  harc_v2_calibrated_params_physics.json 저장'),
        ('6','CF4=22/Ar=8 조건 제외',
         'CD_top=140nm 이상치 (측정 불확실성) → 캘리브레이션에서 제외, Validation에서 별도 표시'),
    ]
    step_colors=['#1E3A5F','#1E3A5F','#2D3F52','#2D3F52','#374151','#92400E']
    y=1.18
    for (num,title,body),color in zip(steps,step_colors):
        rect(s3,0.22,y,0.38,0.38,color)
        txb(s3,num,0.22,y+0.04,0.38,0.32,fs=12,bold=True,col='#FFFFFF',align=PP_ALIGN.CENTER)
        txb(s3,title,0.70,y,12.3,0.26,fs=9.5,bold=True,col=color)
        txb(s3,body, 0.70,y+0.25,12.3,0.26,fs=8.5,col='#374151')
        if num!='6':
            rect(s3,0.39,y+0.38,0.02,0.20,'#D1D5DB')
        y+=0.64

    rect(s3,0.2,6.82,13.0,0.46,'#F0F4F8')
    txb(s3,'최적화 파라미터: 20개  |  방정식: 4조건 × 4출력 = 16개  |  DOF = −4  |  Bounded TRF로 과다 파라미터 처리',
        0.3,6.86,12.8,0.36,fs=9,bold=False,col='#1E3A5F',align=PP_ALIGN.CENTER)

    # ── SLIDE 4: 1-3 Validation ──────────────────────────────────────────────
    s4=prs.slides.add_slide(blank)
    header(s4,'1-3.  Validation  —  실험 vs 시뮬레이션 비교',
           '캘리브레이션된 파라미터로 4개 실험 조건 예측 → 정확도 검증')

    img(s4,cal,0.2,1.1,12.9,3.8)

    # 정확도 테이블
    v_hdr=['CF4/Ar','깊이 오차','CD_top 오차','CD_bot 오차','AR  실험','AR  시뮬']
    v_dat=[
        ['6/24',  '+0.7 %', '-0.1 %',  '-26.6 %', '6.519', '6.571'],
        ['10/20', '-2.2 %', '+1.4 %',  '+23.2 %', '6.780', '6.543'],
        ['14/16', '-1.1 %', '+1.1 %',  '+6.7 %',  '6.363', '6.230'],
        ['18/12', '+0.4 %', '-3.8 %',  '-13.8 %', '5.441', '5.679'],
    ]
    cws=[1.5,2.05,2.05,2.05,1.85,1.85]; cx=0.35
    cxs=[]
    for w in cws:
        cxs.append(cx); cx+=w
    rh=0.38; yt=5.1
    # 헤더 행
    for j,(h,x,w) in enumerate(zip(v_hdr,cxs,cws)):
        rect(s4,x,yt,w-0.04,rh,'#1E293B','#475569')
        txb(s4,h,x+0.06,yt+0.07,w-0.10,rh-0.1,fs=8.5,bold=True,col='#FFFFFF',align=PP_ALIGN.CENTER)
    yt+=rh+0.02
    # 데이터 행
    err_thres=[(5,'#15803D'),(15,'#D97706'),(999,'#DC2626')]
    for i,row in enumerate(v_dat):
        bg='#F0F4F8' if i%2==0 else '#FFFFFF'
        for j,(val,x,w) in enumerate(zip(row,cxs,cws)):
            rect(s4,x,yt,w-0.04,rh,bg,'#D1D5DB')
            fc='#111827'
            if j in [1,2,3]:  # 오차 열 색깔 코딩
                num_s=val.replace('%','').replace('+','').replace('-','').replace(' ','')
                try:
                    err=abs(float(num_s))
                    for thr,c in err_thres:
                        if err<=thr: fc=c; break
                except: pass
            bold=(j==4 and val in ['6.780']) or (j==5 and val in ['6.603'])
            txb(s4,val,x+0.06,yt+0.07,w-0.10,rh-0.1,fs=9,bold=False,col=fc,align=PP_ALIGN.CENTER)
        yt+=rh+0.02

    rect(s4,0.2,7.05,13.0,0.38,'#FEF3C7')
    txb(s4,'CD_bot 오차 최대 ±27% — 폴리머 축적 비선형성을 단순 선형 모델로 처리한 한계  |  AR·깊이·CD_top 오차 ±4% 이내로 양호',
        0.3,7.09,12.8,0.30,fs=8.5,bold=False,col='#92400E',align=PP_ALIGN.CENTER)

    # ── SLIDE 5: 1-4 역최적화 결과 ───────────────────────────────────────────
    s5=prs.slides.add_slide(blank)
    header(s5,'1-4.  역최적화 결과  —  AR 최대화 공정 조건 추천',
           'V_bias=−1000V 고정  |  CF4+Ar=30sccm  |  t=240s  |  CD_init=200nm')

    img(s5,ars,0.2,1.12,6.3)

    txb(s5,'최적 공정 조건 및 예측 결과',6.65,1.12,6.5,0.38,
        fs=12,bold=True,col='#1E3A5F')

    rows=[
        ('공정 변수','최적값','#EFF6FF','#1E3A5F',True),
        ('CF4 유량','7.50 sccm','#F9FAFB','#374151',False),
        ('Ar 유량','22.50 sccm','#F9FAFB','#374151',False),
        ('CF4 분율','25.0 %','#F9FAFB','#374151',False),
        ('V_bias (고정)','−1000 V','#F9FAFB','#374151',False),
        ('예측 결과','','#DCFCE7','#15803D',True),
        ('식각 깊이','1380 nm','#F0FDF4','#15803D',False),
        ('CD_top','209 nm','#F0FDF4','#15803D',False),
        ('CD_bot','54 nm','#F0FDF4','#15803D',False),
        ('종횡비 AR','6.603  ★ 최대','#FEF2F2','#DC2626',False),
        ('Taper 지수','0.741','#F0FDF4','#15803D',False),
        ('Bowing 지수','0.625','#F0FDF4','#15803D',False),
    ]
    yt=1.58; rh=0.34
    for label,val,bg,col,hdr in rows:
        rect(s5,6.65,yt,3.2,rh,bg,'#E2E8F0')
        rect(s5,9.85,yt,3.35,rh,bg,'#E2E8F0')
        txb(s5,label,6.72,yt+0.05,3.1,rh-0.08,fs=8.5,bold=hdr,col=col)
        if val:
            txb(s5,val,9.92,yt+0.05,3.2,rh-0.08,fs=8.5,bold=(col=='#DC2626'),col=col)
        yt+=rh+0.02

    if os.path.exists(opt_fig):
        img(s5,opt_fig,0.2,5.08,13.0,2.32)

    prs.save(out); print(f'  PPT → {out}')


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__=='__main__':
    fc  = os.path.join(OUTDIR,'ppt_flowchart.png')
    cal = os.path.join(OUTDIR,'ppt_calibration.png')
    ars = os.path.join(OUTDIR,'ppt_ar_sweep.png')
    opt = os.path.join(OUTDIR,'harc_optimization_result_new.png')
    ppt = os.path.join(OUTDIR,'HARC_Simulator_v2.pptx')
    print('그림 생성 중...')
    make_flowchart(fc)
    make_cal_figure(cal)
    make_ar_sweep(ars)
    print('PPT 조립 중...')
    build_ppt(fc,cal,ars,opt,ppt)
    print(f'\n완료! -> {ppt}')
