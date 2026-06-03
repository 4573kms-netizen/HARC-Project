"""
=============================================================================
PINN 기반 물리 모델 파라미터 적절성 검증
HARC Etch Simulator v2 — 10개 Calibrated Parameters
=============================================================================

[분석 목적]
  calibration으로 결정된 10개 물리 파라미터가 물리적으로 타당한지 검증

[분석 항목]
  1. Local Sensitivity   : 파라미터 ±5/20% 변화 → AR 변화량 측정
  2. Boundary Risk       : calibrated 값이 물리 허용 범위 어디에 위치하는지
  3. Identifiability     : Jacobian SVD — 데이터로 파라미터를 구분할 수 있는지
  4. Prediction Accuracy : calibrated 파라미터로 얻은 예측 정확도

[실행 방법]
  1. 이 파일을 harc_etch_simulator_v2.py 와 같은 폴더에 놓기
  2. harc_v2_calibrated_params_physics.json 이 같은 폴더에 있어야 함
     (없으면 calibration을 직접 실행해서 생성)
  3. python pinn_parameter_analysis.py

[출력]
  - 콘솔: 각 분석 결과 수치 출력
  - figures/pinn_param_analysis.png : 종합 시각화
=============================================================================
"""

import sys
import os
import json
import copy
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

# harc_etch_simulator_v2.py 와 같은 폴더에 있다고 가정
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from harc_etch_simulator_v2 import (
    ModelParameters,
    ProcessConditions,
    EXPERIMENTAL_DATA,
    run_forward_simulation,
    _build_experiments,
    calibrate_model_parameters,
)


# =============================================================================
# 0. 설정 상수
# =============================================================================

# 10개 calibrated 파라미터의 물리적 허용 범위 (harc_etch_simulator_v2.py BOUNDS 참조)
ABS_BOUNDS = {
    'k_born':              (0.05,  0.6),
    'k_born_spread_left':  (1.0,   60.0),
    'k_born_spread_right': (0.1,   15.0),
    'K_dep_side':          (1e-19, 1e-14),
    'k_clausing_neutral':  (0.01,  1.0),
    'lambda_neutral':      (0.3,   50.0),
    'K_dep_poly':          (1e-18, 1e-12),
    'alpha_cf4_ion':       (1e-3,  0.8),
    'K_mask_poly':         (0.005, 5.0),
    'K_F_mask':            (1e-20, 1e-14),
}

COND_NAMES = ['CF4=6/Ar=24', 'CF4=10/Ar=20', 'CF4=14/Ar=16', 'CF4=18/Ar=12']


# =============================================================================
# 1. Calibrated 파라미터 로드 (없으면 자동 calibration 실행)
# =============================================================================

def load_or_calibrate(json_path: str) -> tuple:
    """
    JSON 파일에서 calibrated 파라미터를 로드합니다.
    파일이 없으면 calibration을 직접 실행해서 생성합니다.

    Returns:
        mp_cal      : 파라미터가 세팅된 ModelParameters 객체
        CAL_PARAMS  : 파라미터 이름 리스트 (10개)
        cal_vals    : calibrated 값 배열 (10개)
    """
    mp_cal = ModelParameters()

    if os.path.exists(json_path):
        print(f"[로드] {json_path}")
        with open(json_path) as f:
            d = json.load(f)
        for k, v in d.items():
            if hasattr(mp_cal, k):
                setattr(mp_cal, k, v)
    else:
        print(f"[경고] {json_path} 없음 → calibration 자동 실행")
        exp_data = EXPERIMENTAL_DATA.copy()
        cal_data = exp_data[exp_data['reliable']].copy().reset_index(drop=True)
        mp_init  = ModelParameters()

        mp_cal1, _ = calibrate_model_parameters(cal_data, mp_init,  verbose=False)
        mp_cal,  info = calibrate_model_parameters(cal_data, mp_cal1, verbose=False)

        d = {p: getattr(mp_cal, p) for p in info['calibrate_params']}
        with open(json_path, 'w') as f:
            json.dump(d, f, indent=2)
        print(f"  → {json_path} 저장 완료")

    CAL_PARAMS = list(ABS_BOUNDS.keys())          # 순서 고정
    cal_vals   = np.array([getattr(mp_cal, p) for p in CAL_PARAMS])
    return mp_cal, CAL_PARAMS, cal_vals


# =============================================================================
# 2. 헬퍼 함수
# =============================================================================

def run_sim_all_conditions(mp_try, experiments) -> dict:
    """
    주어진 ModelParameters로 4개 실험 조건을 모두 실행하고
    {조건명: {ar, depth, cd_top, cd_bot, meas_*}} 형태로 반환합니다.
    """
    results = {}
    for cond, meas in experiments:
        try:
            r = run_forward_simulation(cond, mp_try, verbose=False)
            label = f"CF4={int(cond.cf4_flow)}/Ar={int(cond.ar_flow)}"
            results[label] = {
                'ar':         r.aspect_ratio,
                'depth':      r.total_depth,
                'cd_top':     r.cd_top,
                'cd_bot':     r.cd_bot,
                'meas_ar':    meas['ar'],
                'meas_depth': meas['depth'],
                'meas_top':   meas['cd_top'],
                'meas_bot':   meas['cd_bot'],
            }
        except Exception:
            pass
    return results


# =============================================================================
# 3. PINN 훈련 (파라미터 공간 → 시뮬레이션 결과 매핑)
# =============================================================================

def build_pinn_training_data(mp_cal, CAL_PARAMS, cal_vals, experiments,
                              n_samples: int = 200, seed: int = 42):
    """
    Latin Hypercube Sampling으로 파라미터 공간을 샘플링하고
    각 샘플에 대해 시뮬레이터를 실행해 훈련 데이터를 생성합니다.

    샘플링 범위: calibrated 값 기준 ×0.3 ~ ×3.0 (로그 스케일)
                 단, 물리 허용 범위(ABS_BOUNDS)를 초과하지 않음

    Returns:
        X_raw : (N_valid, 10)  파라미터 값
        Y_raw : (N_valid, 16)  시뮬레이션 출력 [depth,cdtop,cdbot,ar] × 4조건
    """
    n_p = len(CAL_PARAMS)

    # --- 샘플링 범위 (log-space) ---
    SAMPLE_RATIO = (0.3, 3.0)   # calibrated 값 대비 하한/상한 배율
    lo_log = np.array([
        np.log10(max(cal_vals[i] * SAMPLE_RATIO[0], ABS_BOUNDS[CAL_PARAMS[i]][0]))
        for i in range(n_p)
    ])
    hi_log = np.array([
        np.log10(min(cal_vals[i] * SAMPLE_RATIO[1], ABS_BOUNDS[CAL_PARAMS[i]][1]))
        for i in range(n_p)
    ])

    # --- Latin Hypercube Sampling ---
    sampler   = qmc.LatinHypercube(d=n_p, seed=seed)
    sample_u  = sampler.random(n_samples)                       # (N, 10) in [0,1]
    sample_log = lo_log + sample_u * (hi_log - lo_log)
    samples    = 10.0 ** sample_log                             # 실제 파라미터 값

    # --- 시뮬레이터 실행 ---
    print(f"\n[STEP 1] 파라미터 공간 샘플링 & 시뮬레이터 실행")
    print(f"  {n_samples}개 샘플 × 4개 실험 조건 = {n_samples * 4}회 시뮬레이션")

    X_list, Y_list = [], []
    fail = 0
    for i, samp in enumerate(samples):
        mp_try = copy.deepcopy(mp_cal)
        for j, p in enumerate(CAL_PARAMS):
            setattr(mp_try, p, float(samp[j]))

        row_y, ok = [], True
        for cond, meas in experiments:
            try:
                r = run_forward_simulation(cond, mp_try, verbose=False)
                row_y.extend([r.total_depth, r.cd_top, r.cd_bot, r.aspect_ratio])
            except Exception:
                ok = False
                break

        if ok and len(row_y) == 16:
            X_list.append(samp)
            Y_list.append(row_y)
        else:
            fail += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1:3d}/{n_samples}] valid={len(X_list)}, fail={fail}")

    X_raw = np.array(X_list)
    Y_raw = np.array(Y_list)
    print(f"  최종 valid 샘플: {len(X_raw)} / {n_samples}")
    return X_raw, Y_raw, lo_log, hi_log


def train_pinn(X_raw, Y_raw, n_hidden: int = 64, max_iter: int = 2000,
               lam_physics: float = 0.5):
    """
    2-hidden-layer tanh 신경망을 훈련합니다.

    구조: 10 → 64 → 64 → 16  (tanh 활성화)

    Loss = L_data + lam_physics × L_physics
      L_data   : (PINN 예측 - 시뮬레이터 결과)²  평균
      L_physics : P1(AR 정의) + P2(taper) + P3(depth>0) + P4(cdtop>0) + P5(AR>1)

    최적화: L-BFGS-B (scipy.optimize.minimize)

    Returns:
        theta_opt : 학습된 파라미터 벡터
        Xm, Xs   : 입력 정규화 평균/표준편차
        Ym, Ys   : 출력 정규화 평균/표준편차
    """
    n_in   = X_raw.shape[1]   # 10
    n_out  = Y_raw.shape[1]   # 16

    # --- 정규화 (log-space for params) ---
    X_log  = np.log10(X_raw)
    Xm, Xs = X_log.mean(0), X_log.std(0) + 1e-8
    Ym, Ys = Y_raw.mean(0),  Y_raw.std(0)  + 1e-8
    X = (X_log - Xm) / Xs
    Y = (Y_raw - Ym) / Ys

    # --- 네트워크 파라미터 pack/unpack ---
    def pack(W1, b1, W2, b2, W3, b3):
        return np.concatenate([W1.ravel(), b1, W2.ravel(), b2, W3.ravel(), b3])

    def unpack(th):
        i = 0
        W1 = th[i:i+n_in*n_hidden].reshape(n_in, n_hidden);       i += n_in*n_hidden
        b1 = th[i:i+n_hidden];                                      i += n_hidden
        W2 = th[i:i+n_hidden*n_hidden].reshape(n_hidden, n_hidden); i += n_hidden*n_hidden
        b2 = th[i:i+n_hidden];                                      i += n_hidden
        W3 = th[i:i+n_hidden*n_out].reshape(n_hidden, n_out);       i += n_hidden*n_out
        b3 = th[i:i+n_out]
        return W1, b1, W2, b2, W3, b3

    def forward(th, x):
        W1, b1, W2, b2, W3, b3 = unpack(th)
        h1 = np.tanh(x @ W1 + b1)
        h2 = np.tanh(h1 @ W2 + b2)
        return h2 @ W3 + b3          # (N, 16)

    # --- Physics Loss ---
    # Y 레이아웃: [depth_c1, cdtop_c1, cdbot_c1, ar_c1,
    #             depth_c2, cdtop_c2, cdbot_c2, ar_c2, ...] (4개 조건)
    def physics_loss(th, x_norm):
        y_n = forward(th, x_norm)
        y   = y_n * Ys + Ym              # un-normalise
        total = 0.0
        for c in range(4):
            depth = y[:, c*4+0]
            cdt   = y[:, c*4+1]
            cdb   = y[:, c*4+2]
            ar    = y[:, c*4+3]

            # P1: AR = depth / cd_top
            r1 = (ar - depth / np.maximum(cdt, 1.)) / (Ys[c*4+3] + 1e-8)
            # P2: cd_bot <= cd_top  (taper 방향)
            r2 = np.maximum(cdb - cdt, 0.) / (Ys[c*4+2] + 1e-8)
            # P3: depth > 0
            r3 = np.maximum(-depth, 0.) / (Ys[c*4+0] + 1e-8)
            # P4: cd_top > 0
            r4 = np.maximum(-cdt, 0.) / (Ys[c*4+1] + 1e-8)
            # P5: AR > 1  (HARC 최소 조건)
            r5 = np.maximum(1.0 - ar, 0.) / (Ys[c*4+3] + 1e-8)

            total += np.mean(r1**2 + r2**2 + r3**2 + r4**2 + r5**2)
        return total / 4.0

    def loss_fn(th):
        Ld = np.mean((forward(th, X) - Y)**2)
        Lp = physics_loss(th, X)
        return Ld + lam_physics * Lp

    # --- 학습 ---
    print(f"\n[STEP 2] PINN 훈련 (L-BFGS-B, max_iter={max_iter})")
    print(f"  구조: {n_in} → {n_hidden} → {n_hidden} → {n_out}  (tanh)")
    print(f"  Loss = L_data + {lam_physics} × L_physics")

    n_theta = n_in*n_hidden + n_hidden + n_hidden**2 + n_hidden + n_hidden*n_out + n_out
    np.random.seed(42)
    th0 = np.random.randn(n_theta) * 0.05

    res = minimize(loss_fn, th0, method='L-BFGS-B',
                   options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10})
    theta_opt = res.x

    # R² 계산
    y_pred_n = forward(theta_opt, X)
    y_pred   = y_pred_n * Ys + Ym
    r2_vals  = []
    for j in range(n_out):
        ss_res = np.sum((Y_raw[:, j] - y_pred[:, j])**2)
        ss_tot = np.sum((Y_raw[:, j] - Y_raw[:, j].mean())**2) + 1e-12
        r2_vals.append(1 - ss_res / ss_tot)
    print(f"  Final loss={res.fun:.5f}  Avg R²={np.mean(r2_vals):.4f}")

    # forward 함수를 클로저로 반환
    def predict(th, x_raw):
        """파라미터 배열(raw) → 시뮬레이션 결과 예측 (un-normalised)"""
        x_log  = np.log10(x_raw)
        x_norm = (x_log - Xm) / Xs
        y_norm = forward(th, x_norm.reshape(1, -1))
        return (y_norm * Ys + Ym)[0]   # (16,)

    return theta_opt, predict, Xm, Xs, Ym, Ys


# =============================================================================
# 4. Analysis 1 — Local Sensitivity (시뮬레이터 직접 사용)
# =============================================================================

def sensitivity_analysis(mp_cal, CAL_PARAMS, cal_vals, experiments,
                          pct_list=(0.05, 0.20)):
    """
    각 파라미터를 ±pct% 변화시켜 AR에 미치는 영향을 측정합니다.

    ΔAR(%)/Δparam(%) = (AR_plus - AR_minus) / (2 × pct × base_AR) × 100

    반환: {파라미터명: {dAR_5pct, dAR_20pct, grade}}
    """
    print(f"\n[STEP 3] Sensitivity Analysis")
    print(f"{'파라미터':<28} {'±5% ΔAR':>10} {'±20% ΔAR':>11} {'등급':>10} {'AR방향':>8}")
    print("-" * 70)

    # baseline AR
    base_res = run_sim_all_conditions(mp_cal, experiments)
    base_ar  = np.mean([base_res[c]['ar'] for c in COND_NAMES if c in base_res])

    sens = {}
    for p in CAL_PARAMS:
        idx    = CAL_PARAMS.index(p)
        base_v = cal_vals[idx]
        dAR_by_pct = {}

        for pct in pct_list:
            # +pct%
            mp_p = copy.deepcopy(mp_cal)
            setattr(mp_p, p, base_v * (1 + pct))
            # -pct%
            mp_m = copy.deepcopy(mp_cal)
            setattr(mp_m, p, base_v * (1 - pct))

            try:
                ar_p = np.mean([
                    run_forward_simulation(cond, mp_p, verbose=False).aspect_ratio
                    for cond, _ in experiments
                ])
                ar_m = np.mean([
                    run_forward_simulation(cond, mp_m, verbose=False).aspect_ratio
                    for cond, _ in experiments
                ])
                # AR 변화율 (base_AR 대비 %)
                dAR = (ar_p - ar_m) / (2 * pct * base_ar) * 100
            except Exception:
                dAR = 0.0

            dAR_by_pct[pct] = dAR

        s5  = dAR_by_pct[pct_list[0]]
        s20 = dAR_by_pct[pct_list[1]]
        grade = "HIGH 🔴" if abs(s5) > 5 else "MED 🟡" if abs(s5) > 0.5 else "LOW 🟢"
        direction = "AR↑" if s5 > 0 else "AR↓"
        sens[p] = {'dAR_5pct': s5, 'dAR_20pct': s20, 'grade': grade}

        print(f"  {p:<28} {s5:>+10.3f}% {s20:>+10.3f}%  {grade:>10} {direction:>8}")

    return sens, base_ar


# =============================================================================
# 5. Analysis 2 — Boundary Risk
# =============================================================================

def boundary_risk_analysis(CAL_PARAMS, cal_vals):
    """
    calibrated 값이 물리 허용 범위(ABS_BOUNDS)의 어디에 위치하는지 계산합니다.

    위치(%) = (log(cal) - log(lo)) / (log(hi) - log(lo)) × 100

    위험 기준:
      < 10% 또는 > 90%  →  HIGH ⚠⚠
      < 20% 또는 > 80%  →  MED  ⚠
      그 외              →  LOW  ✓
    """
    print(f"\n[STEP 4] Boundary Risk Analysis")
    print(f"{'파라미터':<28} {'Cal.값':>12} {'하한':>10} {'상한':>10} {'위치%':>8} {'위험도'}")
    print("-" * 78)

    positions, risks = [], []
    for p in CAL_PARAMS:
        lo, hi = ABS_BOUNDS[p]
        cv  = cal_vals[CAL_PARAMS.index(p)]
        pos = (np.log10(cv) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
        risk = ("HIGH ⚠⚠" if (pos < 0.10 or pos > 0.90) else
                "MED ⚠"   if (pos < 0.20 or pos > 0.80) else
                "LOW ✓")
        positions.append(pos)
        risks.append(risk)
        print(f"  {p:<28} {cv:>12.3e} {lo:>10.2e} {hi:>10.2e} {pos*100:>7.1f}%  {risk}")

    return positions, risks


# =============================================================================
# 6. Analysis 3 — Identifiability (Jacobian SVD)
# =============================================================================

def identifiability_analysis(mp_cal, CAL_PARAMS, cal_vals, experiments,
                              base_ar, pct: float = 0.10):
    """
    Jacobian J[i,j] = ∂AR_i / ∂(log param_j) 를 수치 미분으로 계산하고
    SVD로 분해해 파라미터 구분 가능성을 분석합니다.

    - 정규화된 Jacobian 사용: 단위계 혼재 문제 제거
    - Condition Number κ = σ_max / σ_min
        κ < 100   → 양호 (파라미터들이 서로 구분 가능)
        κ > 1000  → 불량 (일부 파라미터 degenerate)
    """
    print(f"\n[STEP 5] Identifiability — Jacobian SVD")

    ar_std = np.std([
        run_forward_simulation(cond, mp_cal, verbose=False).aspect_ratio
        for cond, _ in experiments
    ]) + 0.01

    n_p  = len(CAL_PARAMS)
    n_c  = 4   # 4개 실험 조건
    J    = np.zeros((n_c, n_p))

    for j, p in enumerate(CAL_PARAMS):
        bv   = cal_vals[j]
        mp_p = copy.deepcopy(mp_cal); setattr(mp_p, p, bv * (1 + pct))
        mp_m = copy.deepcopy(mp_cal); setattr(mp_m, p, bv * (1 - pct))
        try:
            ars_p = [run_forward_simulation(c, mp_p, verbose=False).aspect_ratio
                     for c, _ in experiments]
            ars_m = [run_forward_simulation(c, mp_m, verbose=False).aspect_ratio
                     for c, _ in experiments]
            # d(AR) / d(log p)  — log-space derivative
            dlog_p = np.log(1 + pct) - np.log(1 - pct)
            J[:, j] = [(p_ - m_) / (dlog_p * ar_std)
                       for p_, m_ in zip(ars_p, ars_m)]
        except Exception:
            J[:, j] = 0.0

    U, s_svd, Vt = np.linalg.svd(J, full_matrices=False)
    cond_num = s_svd[0] / (s_svd[-1] + 1e-12)

    print(f"  Jacobian shape: {J.shape}  (4 conditions × {n_p} params)")
    print(f"\n  SVD 특이값:")
    for i, sv in enumerate(s_svd):
        flag = "✓ 정보 있음" if sv > 0.1 else "⚠ 정보 부족"
        print(f"    σ_{i+1} = {sv:.4f}  {flag}")
    print(f"\n  Condition Number κ = {cond_num:.1f}  "
          f"({'양호' if cond_num < 100 else '불량 — 일부 파라미터 degenerate'})")

    return J, s_svd, cond_num


# =============================================================================
# 7. Analysis 4 — Prediction Accuracy & Physics Consistency
# =============================================================================

def prediction_accuracy(mp_cal, experiments):
    """
    calibrated 파라미터로 4개 조건을 예측하고 실험값과 비교합니다.
    물리 일관성 (cd_bot ≤ cd_top) 도 확인합니다.
    """
    print(f"\n[STEP 6] Prediction Accuracy & Physics Consistency")
    print(f"  {'조건':<16} {'Depth_exp':>10} {'Depth_sim':>10} {'Err%':>7}  "
          f"{'AR_exp':>7} {'AR_sim':>7} {'Err%':>7}  {'물리'}") 
    print("  " + "-" * 80)

    err_ar, err_depth = [], []
    base_res = run_sim_all_conditions(mp_cal, experiments)
    for c in COND_NAMES:
        if c not in base_res:
            continue
        r = base_res[c]
        d_err  = 100 * (r['depth'] - r['meas_depth']) / r['meas_depth']
        ar_err = 100 * (r['ar']    - r['meas_ar'])    / r['meas_ar']
        phys   = "OK ✓" if r['cd_bot'] <= r['cd_top'] else "VIOLATION ⚠"
        err_ar.append(ar_err)
        err_depth.append(d_err)
        print(f"  {c:<16} {r['meas_depth']:>10.1f} {r['depth']:>10.1f} {d_err:>+7.1f}%  "
              f"{r['meas_ar']:>7.3f} {r['ar']:>7.3f} {ar_err:>+7.1f}%  {phys}")

    return err_ar, err_depth, base_res


# =============================================================================
# 8. 시각화
# =============================================================================

def plot_results(CAL_PARAMS, cal_vals,
                 sens, positions, risks,
                 J_norm, s_svd, cond_num,
                 err_ar, err_depth,
                 save_path: str):
    """
    4가지 분석 결과를 하나의 그림으로 종합합니다.

    패널 구성:
      A (좌상, 넓게): Sensitivity 막대 그래프
      B (우상):       Boundary Risk 수평 막대
      C (좌중):       Jacobian 히트맵
      D (중중):       SVD 특이값 막대
      E (우중):       예측 정확도
      F (하단 전체):  종합 판정 표
    """
    short = ['k_born', 'k_born\n_spr_L', 'k_born\n_spr_R', 'K_dep\n_side',
             'k_claus\n_neut',  'λ_neut', 'K_dep\n_poly',   'α_cf4\n_ion',
             'K_mask\n_poly',   'K_F\n_mask']

    fig = plt.figure(figsize=(18, 15))
    fig.suptitle(
        'PINN-Based Physical Model Parameter Validity Analysis\n'
        'HARC Etch Simulator v2 — 10 Calibrated Parameters',
        fontsize=14, fontweight='bold', y=0.99
    )
    gs = gridspec.GridSpec(3, 3, hspace=0.52, wspace=0.38, figure=fig)

    # ── A: Sensitivity ─────────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, :2])
    vals_s = [sens[p]['dAR_5pct'] for p in CAL_PARAMS]
    colors_s = []
    for v in vals_s:
        if   abs(v) > 5:   colors_s.append('#DC2626' if v < 0 else '#2563EB')
        elif abs(v) > 0.5: colors_s.append('#F97316' if v < 0 else '#3B82F6')
        else:              colors_s.append('#9CA3AF')

    bars_a = ax_a.barh(range(len(CAL_PARAMS)), vals_s,
                       color=colors_s, edgecolor='k', lw=0.7, alpha=0.88)
    ax_a.axvline(0,  color='black', lw=1.2)
    ax_a.axvline(+5, color='red',   lw=1, linestyle='--', alpha=0.5, label='±5% threshold')
    ax_a.axvline(-5, color='red',   lw=1, linestyle='--', alpha=0.5)
    ax_a.set_yticks(range(len(CAL_PARAMS)))
    ax_a.set_yticklabels(CAL_PARAMS, fontsize=9)
    ax_a.set_xlabel('%ΔAR per 5% parameter change')
    ax_a.set_title(
        'Sensitivity Analysis: 각 파라미터 5% 변화 → AR에 미치는 영향\n'
        '(빨강=AR 감소, 파랑=AR 증가, 크기=영향력)',
        fontweight='bold', fontsize=10
    )
    for i, (bar, v) in enumerate(zip(bars_a, vals_s)):
        ax_a.text(v + (0.3 if v >= 0 else -0.3), i, f'{v:+.2f}%',
                  ha='left' if v >= 0 else 'right', va='center', fontsize=8)
    ax_a.grid(axis='x', alpha=0.3)
    ax_a.legend(fontsize=8)

    # ── B: Boundary Risk ───────────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 2])
    risk_colors = {'LOW': '#16A34A', 'MED': '#F59E0B', 'HIGH': '#DC2626'}
    bar_colors_b = [risk_colors[r.split()[0]] for r in risks]
    ax_b.barh(range(len(CAL_PARAMS)), positions,
              color=bar_colors_b, edgecolor='k', lw=0.7, alpha=0.88)
    ax_b.axvline(0.10, color='red',    lw=1.5, linestyle='--', alpha=0.7)
    ax_b.axvline(0.90, color='red',    lw=1.5, linestyle='--', alpha=0.7)
    ax_b.axvline(0.20, color='orange', lw=1.0, linestyle=':',  alpha=0.7)
    ax_b.axvline(0.80, color='orange', lw=1.0, linestyle=':',  alpha=0.7)
    ax_b.axvline(0.50, color='green',  lw=1.0, linestyle='-',  alpha=0.3)
    ax_b.set_xlim(0, 1)
    ax_b.set_yticks(range(len(CAL_PARAMS)))
    ax_b.set_yticklabels(CAL_PARAMS, fontsize=8)
    ax_b.set_xlabel('Position in physical bounds (0=lower, 1=upper)')
    ax_b.set_title('Boundary Risk\n(빨강=경계 위험, 초록=안전)', fontweight='bold', fontsize=10)
    ax_b.legend(handles=[
        Patch(color='#16A34A', label='LOW ✓'),
        Patch(color='#F59E0B', label='MED ⚠'),
        Patch(color='#DC2626', label='HIGH ⚠⚠'),
    ], fontsize=8, loc='lower right')
    ax_b.grid(axis='x', alpha=0.3)

    # ── C: Jacobian Heatmap ────────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    im = ax_c.imshow(np.abs(J_norm), cmap='YlOrRd', aspect='auto')
    ax_c.set_xticks(range(len(CAL_PARAMS)))
    ax_c.set_xticklabels(short, fontsize=6.5, rotation=45, ha='right')
    ax_c.set_yticks(range(4))
    ax_c.set_yticklabels(COND_NAMES, fontsize=8)
    ax_c.set_title('Jacobian |∂AR/∂logParam|\n(크면 = 해당 조건에서 파라미터 영향 큼)',
                   fontweight='bold', fontsize=9)
    plt.colorbar(im, ax=ax_c, shrink=0.8)
    for i in range(4):
        for j in range(len(CAL_PARAMS)):
            ax_c.text(j, i, f'{J_norm[i,j]:.2f}', ha='center', va='center',
                      fontsize=6, color='white' if abs(J_norm[i, j]) > 1.5 else 'black')

    # ── D: SVD 특이값 ──────────────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.bar(range(len(s_svd)), s_svd,
             color=['#2563EB' if s > 0.1 else '#DC2626' for s in s_svd],
             edgecolor='k', lw=0.7, alpha=0.88)
    ax_d.axhline(0.1, color='red', linestyle='--', lw=1.5, label='Threshold(0.1)')
    for i, sv in enumerate(s_svd):
        ax_d.text(i, sv * 1.05, f'{sv:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax_d.set_xticks(range(len(s_svd)))
    ax_d.set_xticklabels([f'σ_{i+1}' for i in range(len(s_svd))], fontsize=9)
    ax_d.set_ylabel('Singular value')
    ax_d.set_title(f'SVD of Jacobian\nCondition κ = {cond_num:.1f}',
                   fontweight='bold', fontsize=9)
    ax_d.legend(fontsize=8)
    ax_d.grid(axis='y', alpha=0.3)

    # ── E: Prediction Accuracy ─────────────────────────────────────────────
    ax_e = fig.add_subplot(gs[1, 2])
    xe = np.arange(4)
    w  = 0.35
    ax_e.bar(xe - w/2, err_ar,    width=w, color='#2563EB', alpha=0.85,
             label='AR error%',    edgecolor='k', lw=0.7)
    ax_e.bar(xe + w/2, err_depth, width=w, color='#16A34A', alpha=0.85,
             label='Depth error%', edgecolor='k', lw=0.7)
    ax_e.axhline(+5, color='red', linestyle='--', lw=1.2, label='±5% target')
    ax_e.axhline(-5, color='red', linestyle='--', lw=1.2)
    ax_e.set_xticks(xe)
    ax_e.set_xticklabels(COND_NAMES, fontsize=7.5, rotation=15)
    ax_e.set_ylabel('Prediction Error [%]')
    ax_e.set_title('Calibrated Param 예측 정확도\n(|err| < 5% = 목표)',
                   fontweight='bold', fontsize=9)
    ax_e.legend(fontsize=8)
    ax_e.grid(axis='y', alpha=0.3)
    for i, (a, dep) in enumerate(zip(err_ar, err_depth)):
        ax_e.text(i - w/2, a + (0.3 if a >= 0 else -0.8),
                  f'{a:+.1f}', ha='center', fontsize=7.5, fontweight='bold',
                  color='red' if abs(a) > 5 else 'navy')
        ax_e.text(i + w/2, dep + (0.3 if dep >= 0 else -0.8),
                  f'{dep:+.1f}', ha='center', fontsize=7.5, fontweight='bold',
                  color='red' if abs(dep) > 5 else 'darkgreen')

    # ── F: 종합 판정 표 ────────────────────────────────────────────────────
    ax_f = fig.add_subplot(gs[2, :])
    ax_f.axis('off')

    rows = []
    for i, p in enumerate(CAL_PARAMS):
        s5   = sens[p]['dAR_5pct']
        pos  = positions[i] * 100
        rk   = risks[i]
        cv   = cal_vals[i]
        lo, hi = ABS_BOUNDS[p]
        jnorm_col = np.linalg.norm(J_norm[:, i])
        ident = 'YES' if jnorm_col > 0.5 else 'WEAK'

        grade = ("HIGH 🔴" if abs(s5) > 5 else
                 "MED 🟡"  if abs(s5) > 0.5 else "LOW 🟢")

        if 'HIGH' in rk:
            verdict = 'REVIEW ⚠'
        elif abs(s5) < 0.1:
            verdict = 'INERT ?'
        else:
            verdict = 'VALID ✓'

        rows.append([
            p,
            f'{cv:.3e}',
            f'{lo:.1e}~{hi:.1e}',
            f'{pos:.0f}%',
            rk,
            f'{s5:+.2f}%',
            grade,
            ident,
            verdict,
        ])

    col_labels = ['Parameter', 'Cal.Value', 'Phys.Range', 'Pos%', 'BoundRisk',
                  'ΔAR/5%', 'SensGrade', 'Identifiable', 'Verdict']
    tbl = ax_f.table(cellText=rows, colLabels=col_labels,
                     cellLoc='center', loc='center', bbox=[0, -0.05, 1, 1.05])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor('#1E40AF')
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    for i in range(1, len(rows) + 1):
        v = rows[i - 1][8]
        fc = ('#DCFCE7' if 'VALID'   in v else
              '#FEF9C3' if 'REVIEW'  in v else '#F1F5F9')
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(fc)
        if 'REVIEW' in v or 'HIGH' in rows[i - 1][4]:
            tbl[i, 4].set_facecolor('#FEE2E2')
            tbl[i, 8].set_facecolor('#FEE2E2')
        if 'INERT' in v:
            tbl[i, 6].set_facecolor('#E2E8F0')

    ax_f.set_title('종합 판정: PINN 기반 물리 모델 파라미터 적절성 검증 결과',
                   fontweight='bold', fontsize=12, y=1.02)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n  그림 저장: {save_path}")


# =============================================================================
# 9. Main
# =============================================================================

def main():
    print("=" * 65)
    print("  PINN 기반 물리 모델 파라미터 적절성 검증")
    print("  HARC Etch Simulator v2")
    print("=" * 65)

    # --- 파라미터 로드 ---
    json_path = os.path.join(_HERE, 'harc_v2_calibrated_params_physics.json')
    mp_cal, CAL_PARAMS, cal_vals = load_or_calibrate(json_path)

    # --- 실험 조건 ---
    exp_data    = EXPERIMENTAL_DATA.copy()
    cal_data    = exp_data[exp_data['reliable']].copy().reset_index(drop=True)
    experiments = _build_experiments(cal_data)

    # --- PINN 훈련 데이터 생성 ---
    X_raw, Y_raw, lo_log, hi_log = build_pinn_training_data(
        mp_cal, CAL_PARAMS, cal_vals, experiments, n_samples=200
    )

    # --- PINN 훈련 ---
    theta_opt, predict, Xm, Xs, Ym, Ys = train_pinn(
        X_raw, Y_raw, n_hidden=64, max_iter=2000, lam_physics=0.5
    )

    # --- Analysis 1: Sensitivity ---
    sens, base_ar = sensitivity_analysis(
        mp_cal, CAL_PARAMS, cal_vals, experiments
    )

    # --- Analysis 2: Boundary Risk ---
    positions, risks = boundary_risk_analysis(CAL_PARAMS, cal_vals)

    # --- Analysis 3: Identifiability ---
    J_norm, s_svd, cond_num = identifiability_analysis(
        mp_cal, CAL_PARAMS, cal_vals, experiments, base_ar
    )

    # --- Analysis 4: Prediction Accuracy ---
    err_ar, err_depth, base_res = prediction_accuracy(mp_cal, experiments)

    # --- 시각화 ---
    save_path = os.path.join(_HERE, 'figures', 'pinn_param_analysis.png')
    plot_results(
        CAL_PARAMS, cal_vals,
        sens, positions, risks,
        J_norm, s_svd, cond_num,
        err_ar, err_depth,
        save_path=save_path,
    )

    print("\n" + "=" * 65)
    print("  분석 완료")
    print("=" * 65)


if __name__ == '__main__':
    main()
