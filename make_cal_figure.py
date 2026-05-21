import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('figures', exist_ok=True)

# ── Experimental values ──────────────────────────────────────────────────────
labels    = ['6/24', '10/20', '14/16', '18/12', '22/8']
exp_depth = np.array([1369.0, 1390.0, 1298.0, 1159.0,  616.0])
exp_top   = np.array([ 210.0,  205.0,  204.0,  213.0,  140.1])
exp_bot   = np.array([  74.4,   44.2,   53.4,   73.6,   65.6])
exp_ar    = exp_depth / exp_top

# Post-calibration errors (%): actual simulation results (hinge-loss calibration)
err_depth = np.array([ +7.9,  -6.2,  -3.1,  +3.2, np.nan])
err_top   = np.array([ -0.1,  +1.4,  +1.1,  -3.8, np.nan])
err_bot   = np.array([ -4.4,  +3.7,  -0.2,  -3.0, np.nan])
err_ar    = np.array([ +8.0,  -7.5,  -4.2,  +7.3, np.nan])

post_depth = exp_depth * (1 + err_depth / 100)
post_top   = exp_top   * (1 + err_top   / 100)
post_bot   = exp_bot   * (1 + err_bot   / 100)
post_ar    = exp_ar    * (1 + err_ar    / 100)

# Pre-calibration (loaded from physics.json, before asymmetric-Gaussian k_born fitting)
pre_depth = np.array([1479.7, 1360.7, 1300.2, 1211.7, 1060.4])
pre_top   = np.array([ 209.9,  207.8,  206.2,  204.9,  204.2])
pre_bot   = np.array([  71.6,   54.9,   60.6,   74.7,  101.1])
pre_ar    = pre_depth / pre_top

C_EXP  = '#555555'
C_PRE  = '#5b9bd5'
C_POST = '#ed7d31'
C_OK   = '#27ae60'
C_WARN = '#e74c3c'

metrics = [
    ('Etch Depth [nm]', exp_depth, pre_depth, post_depth, err_depth),
    ('Top CD [nm]',     exp_top,   pre_top,   post_top,   err_top),
    ('Bottom CD [nm]',  exp_bot,   pre_bot,   post_bot,   err_bot),
    ('Aspect Ratio',    exp_ar,    pre_ar,    post_ar,    err_ar),
]

x = np.arange(len(labels))
w = 0.25

fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor='white')
fig.suptitle(
    'HARC v2 — Calibration Result  |  '
    '250 W · 10 mTorr · V$_{bias}$=−1000 V · 15 °C · 240 s',
    fontsize=12, fontweight='bold'
)

for ax, (title, meas, pre, post, errs) in zip(axes.flat, metrics):
    ax.set_facecolor('white')

    # Bar order: Pre | Post | Experiment
    ax.bar(x - w, pre,  width=w, color=C_PRE,  alpha=0.9, label='Pre-calibration',
           edgecolor='white', linewidth=0.5)
    ax.bar(x,     post, width=w, color=C_POST, alpha=0.9, label='Post-calibration',
           edgecolor='white', linewidth=0.5)
    ax.bar(x + w, meas, width=w, color=C_EXP,  alpha=0.9, label='Experiment',
           edgecolor='white', linewidth=0.5)

    # Bracket between Post-cal and Experiment bars
    for xi, (p, m, e) in enumerate(zip(post, meas, errs)):
        if np.isnan(e):
            continue
        clr = C_OK if abs(e) < 10 else C_WARN
        x_post = xi          # post-cal bar center
        x_exp  = xi + w      # experiment bar center
        y_post, y_exp = p, m
        y_top  = max(y_post, y_exp) * 1.03
        tick_h = y_top * 0.012   # small tick height

        # Bracket: left tick | top line | right tick
        ax.plot([x_post, x_post], [y_post, y_top], color=clr, lw=1.2)
        ax.plot([x_post, x_exp],  [y_top,  y_top],  color=clr, lw=1.2)
        ax.plot([x_exp,  x_exp],  [y_exp,  y_top],  color=clr, lw=1.2)

        # % label above bracket center
        ax.text((x_post + x_exp) / 2, y_top * 1.005,
                f'{e:+.1f}%', ha='center', va='bottom',
                fontsize=8, fontweight='bold', color=clr)

    ax.set_xticks(x)
    ax.set_xticklabels([f'CF4/Ar\n{lb}' for lb in labels], fontsize=8)
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.set_ylabel(title, fontsize=9)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')

plt.tight_layout()
plt.savefig('figures/harc_v2_calibration.png', dpi=150,
            bbox_inches='tight', facecolor='white')
print("Saved → figures/harc_v2_calibration.png")
plt.show()
