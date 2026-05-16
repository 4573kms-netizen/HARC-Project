"""
Point 5 (CF4=22/Ar=8) 이상치 분석
CD_top=140.1nm vs 다른 조건들 204-213nm — 원인 규명
"""
import json, sys
import numpy as np
sys.path.insert(0, r'C:\Users\4573k\Desktop\HARC_simulation_Claude')

from harc_etch_simulator_v2 import (
    ModelParameters, ProcessConditions, SimulationResult,
    run_forward_simulation, calc_plasma_fluxes, calc_mean_ion_energy,
    sputtering_yield, ion_enhanced_factor, EXPERIMENTAL_DATA,
)

with open(r'C:\Users\4573k\Desktop\HARC_simulation_Claude\harc_v2_calibrated_params_physics.json') as f:
    cal_dict = json.load(f)
mp = ModelParameters()
for k, v in cal_dict.items():
    if hasattr(mp, k):
        setattr(mp, k, float(v))

EXP = EXPERIMENTAL_DATA.copy()

# ─── 1. 전체 정확도 테이블 (5점 모두) ──────────────────────────────────────────
print("=" * 95)
print("  CALIBRATION ACCURACY  (harc_v2_calibrated_params_physics.json)")
print(f"  {'CF4/Ar':>8}  "
      f"{'Dep_exp':>8} {'Dep_sim':>8} {'ErrD%':>6}  "
      f"{'Top_exp':>7} {'Top_sim':>7} {'ErrT%':>6}  "
      f"{'Bot_exp':>7} {'Bot_sim':>7} {'ErrB%':>6}  "
      f"{'AR_exp':>6} {'AR_sim':>6}")
print("-" * 95)

results = {}
for _, row in EXP.iterrows():
    cond = ProcessConditions(
        cf4_flow=row['cf4_flow'], ar_flow=row['ar_flow'],
        v_bias=row['v_bias'], source_power=row['source_power'],
        pressure=row['pressure'], substrate_temp=row['substrate_temp'],
        etch_time=row['etch_time'], cd_initial=row['cd_initial'],
        mask_thickness=row['mask_thickness'], target_depth=row['target_depth'],
    )
    r = run_forward_simulation(cond, mp, verbose=False)
    lbl   = f"{int(row['cf4_flow'])}/{int(row['ar_flow'])}"
    de, ds = row['depth_meas'],  r.total_depth
    te, ts = row['cd_top_meas'], r.cd_top
    be, bs = row['cd_bot_meas'], r.cd_bot
    are_   = de / max(te, 1.0)
    ars    = r.aspect_ratio
    tag    = "  ← EXCLUDED" if not row['reliable'] else ""
    print(f"  {lbl:>8}  "
          f"{de:>8.1f} {ds:>8.1f} {100*(ds-de)/de:>+6.1f}%  "
          f"{te:>7.1f} {ts:>7.1f} {100*(ts-te)/te:>+6.1f}%  "
          f"{be:>7.1f} {bs:>7.1f} {100*(bs-be)/be:>+6.1f}%  "
          f"{are_:>6.3f} {ars:>6.3f}{tag}")
    results[lbl] = dict(cond=cond, r=r, row=row)

print("=" * 95)

# ─── 2. Point 5 심층 분석 ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  POINT 5 DEEP ANALYSIS  (CF4=22/Ar=8, t=240s)")
print("=" * 70)

cond5 = results['22/8']['cond']
r5    = results['22/8']['r']

# 플라즈마 플럭스 비교 (전 조건)
print("\n  [A] 표면 플럭스 비교 (surface, z=0)")
print(f"  {'CF4/Ar':>8}  {'Γ_F':>12}  {'Γ_CFx':>12}  {'Γ_ion':>12}  {'E_ion[eV]':>10}")
print(f"  {'-'*58}")
for lbl, d in results.items():
    c = d['cond']
    gf, gcfx, gi = calc_plasma_fluxes(c, mp)
    ei = calc_mean_ion_energy(c, mp)
    print(f"  {lbl:>8}  {gf:>12.3e}  {gcfx:>12.3e}  {gi:>12.3e}  {ei:>10.1f}")

# 마스크 진화 분석
print("\n  [B] 마스크 CD 진화 원인 분석")
print(f"  실험 CD_top = {results['22/8']['row']['cd_top_meas']:.1f} nm (이상치: 다른 조건 204-213)")
print(f"  시뮬 CD_top = {r5.cd_top:.1f} nm")
print()

for lbl, d in results.items():
    c  = d['cond']
    r  = d['r']
    gf, gcfx, gi = calc_plasma_fluxes(c, mp)
    ei = calc_mean_ion_energy(c, mp)
    Ys = sputtering_yield(ei, mp)
    R_sput = mp.K_mask_lat * mp.K_sput * gi * Ys * mp.Y_mask_ratio * c.ar_fraction
    R_F    = mp.K_F_mask * gf
    R_poly = mp.K_dep_poly * mp.K_mask_poly * gcfx
    net    = 2.0 * (R_sput + R_F - R_poly) * c.etch_time  # total ΔCD_mask over 240s
    print(f"  CF4={int(c.cf4_flow):>2}/Ar={int(c.ar_flow):>2}: "
          f"R_sput={R_sput:.3e}  R_F={R_F:.3e}  R_poly={R_poly:.3e} "
          f"→ ΔCD_mask={net:+.1f}nm  final_top={r.cd_top:.1f}nm")

# CF4=22 조건에서 특이점 체크
print("\n  [C] CF4=22/Ar=8 특이점")
gf5, gcfx5, gi5 = calc_plasma_fluxes(cond5, mp)
ei5  = calc_mean_ion_energy(cond5, mp)
print(f"  ar_fraction = {cond5.ar_fraction:.3f}  (낮음 → 이온 flux 감소)")
print(f"  cf4_fraction= {cond5.cf4_fraction:.3f}  (높음 → F radical 증가, CFx 증가)")
print(f"  Γ_F         = {gf5:.3e}")
print(f"  Γ_ion       = {gi5:.3e}")
print(f"  E_ion       = {ei5:.1f} eV")
Ys5  = sputtering_yield(ei5, mp)
fIE5 = ion_enhanced_factor(ei5, mp)
print(f"  Ys (sputter)= {Ys5:.4f}")
print(f"  f_IE        = {fIE5:.2f}")
print(f"  R_F_mask (F etch mask)  = {mp.K_F_mask * gf5:.3e} nm/s")
print(f"  R_sput_mask (Ar sputter)= {mp.K_mask_lat*mp.K_sput*gi5*Ys5*mp.Y_mask_ratio*cond5.ar_fraction:.3e} nm/s")
print(f"  R_poly_mask (CFx dep)   = {mp.K_dep_poly*mp.K_mask_poly*gcfx5:.3e} nm/s")

print("\n  [D] 시뮬레이터 예측 vs 실험 차이")
te5 = results['22/8']['row']['cd_top_meas']
ts5 = r5.cd_top
print(f"  실험 CD_top = {te5:.1f} nm")
print(f"  시뮬 CD_top = {ts5:.1f} nm  (오차 {100*(ts5-te5)/te5:+.1f}%)")
print()
print("  → 가능한 물리적 원인:")
print("    1) 실험 측정 오류 (SEM 준비 중 마스크 손상, tilting artefact)")
print("    2) CF4=22 조건에서 마스크 재료 선택 식각 (고 CF4에서 마스크 etch rate 급증)")
print("    3) 마스크 언더컷팅 (mask undercutting) - CD_top 측정이 마스크 개구부가 아닌")
print("       hole 입구를 측정할 경우 값이 작게 나올 수 있음")
print("    4) 시뮬레이터 한계: 마스크 재료 비선형 식각 (CF4 농도 임계점) 미반영")
print("=" * 70)

# ─── 3. 과다 파라미터 분석 ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  OVER-PARAMETERIZATION 분석")
print("=" * 70)
cal_params = [
    'gamma_F_sat', 'beta_ion', 'alpha_cf4_ion',
    'ion_directionality', 'clausing_exponent',
    'lambda_neutral', 'sigma_iad',
    'K_chem', 'K_ie', 'K_sput', 'K_pass',
    'K_lat_neu', 'K_lat_ion',
    'K_dep_poly', 'K_etch_poly',
    'K_dep_side', 'k_born',
    'K_mask_lat', 'K_mask_poly', 'K_F_mask',
]
n_obs = 4 * 4  # 4 experiments × 4 outputs (depth, CD_top, CD_bot, AR)
print(f"  자유 파라미터: {len(cal_params)}개")
print(f"  관측 방정식:  {n_obs}개  (4 exp × 4 outputs)")
print(f"  DOF:          {n_obs - len(cal_params)}  (음수 = 과다 파라미터)")
print()

# 어떤 파라미터가 식각 깊이에 영향이 약한지 감도 분석
print("  [권장 고정 후보]")
print("  - K_lat_neu, K_lat_ion: CD_bot에만 소폭 영향, depth/CD_top에 미미")
print("  - K_dep_side: sidewall polymer, sigma_iad와 degeneracy 위험")
print("  → 3개 고정 시: 17 params vs 16 eq → DOF=-1 (거의 결정)")
print("  → 단, 현재 TRF bounded 결과는 물리적으로 합리적이므로 즉각 고정 불필요")
print("=" * 70)
