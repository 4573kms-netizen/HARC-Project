"""
make_summary_table.py
Publication-quality summary table for HARC v2.
  Panel A: Calibration — 4 pts, Depth / CD_top / CD_bot / AR
  Panel B: External Validation — RODEo (Chopra et al. SPIE 2018)
Output: figures/harc_v2_summary_table.png  (300 dpi)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(_HERE, 'figures'), exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica Neue', 'DejaVu Sans'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# ─────────────────────────────────────────────────────────────────────────────
# Results from latest simulation run
# ─────────────────────────────────────────────────────────────────────────────
cf4_ar   = ['6 / 24', '10 / 20', '14 / 16', '18 / 12']
dep_e = np.array([1369., 1390., 1298., 1159.])
dep_s = np.array([1403.8, 1361.8, 1284.7, 1163.7])
top_e = np.array([210.0, 205.0, 204.0, 213.0])
top_s = np.array([211.1, 209.4, 208.0, 207.0])
bot_e = np.array([ 74.4,  44.2,  53.4,  73.6])
bot_s = np.array([ 73.8,  45.0,  52.8,  72.9])
ar_e  = dep_e / top_e
ar_s  = dep_s / top_s

pct = lambda s, e: 100.0 * (s - e) / e
dep_err = pct(dep_s, dep_e)
top_err = pct(top_s, top_e)
bot_err = pct(bot_s, bot_e)
ar_err  = pct(ar_s,  ar_e)

val_exp = np.array([39.1, 69.6])
val_mod = np.array([39.1, 66.3])
val_rod = np.array([32.8, 60.4])
val_me  = pct(val_mod, val_exp)
val_re  = pct(val_rod, val_exp)

# ─────────────────────────────────────────────────────────────────────────────
# Color palette
# ─────────────────────────────────────────────────────────────────────────────
C_NAVY    = '#1e3a5f'   # section title bar
C_BLUE    = '#2c5282'   # group / column header
C_SUBHDR  = '#3d6fa8'   # sub-header (lighter blue)
C_WHITE   = '#ffffff'
C_ROW_A   = '#ffffff'
C_ROW_B   = '#f0f5fa'
C_OK_BG   = '#d4edda'   # within ±5%
C_WN_BG   = '#f8d7da'   # beyond ±5%
C_OK_TX   = '#155724'
C_WN_TX   = '#7b1d1d'
C_BORDER  = '#a8b8cc'
C_DARK    = '#1a2a3a'
C_NOTE    = '#4a6685'
THRESH    = 5.0

# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────
FW, FH = 18.0, 7.2          # figure size in inches
fig = plt.figure(figsize=(FW, FH), facecolor='white')
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_aspect('auto'); ax.axis('off')

# ─────────────────────────────────────────────────────────────────────────────
# Layout constants (all in normalized figure coords)
# ─────────────────────────────────────────────────────────────────────────────
TX0 = 0.028          # left margin
TW  = 0.944          # table width
TY0 = 0.965          # top of calibration panel

SH  = 0.055          # section title bar height
H1  = 0.060          # group header height (row 1)
H2  = 0.055          # sub-header height  (row 2)
DH  = 0.093          # data row height
GAP = 0.050          # gap between panels
FTH = 0.028          # footer text height below last table

# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────
def cell(x, y, w, h, text, bg, fg=C_DARK, fs=8.5, bold=False,
         ha='center', wrap=False, pad=0.012):
    rect = mpatches.Rectangle((x, y), w, h,
                               facecolor=bg, edgecolor=C_BORDER,
                               linewidth=0.45, clip_on=False)
    ax.add_patch(rect)
    tx = x + w / 2.0 if ha == 'center' else x + pad
    kw = dict(ha=ha, va='center', fontsize=fs, color=fg,
              fontweight='bold' if bold else 'normal',
              clip_on=False, multialignment='center')
    if wrap:
        kw['wrap'] = True
    ax.text(tx, y + h / 2.0, text, **kw)

def err_bg(v):  return C_OK_BG if abs(v) <= THRESH else C_WN_BG
def err_tx(v):  return C_OK_TX if abs(v) <= THRESH else C_WN_TX
def fmt_e(v):   return f'{v:+.1f}%'

# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION PANEL
# ─────────────────────────────────────────────────────────────────────────────
# 13 columns: CF4/Ar + 4 metrics × (Exp | Sim | Err%)
# Unnormalized widths
_raw = [1.35] + [1.0, 1.0, 0.78] * 4
_tot = sum(_raw)
col_ws = [r / _tot * TW for r in _raw]
col_xs = [TX0 + sum(col_ws[:i]) for i in range(len(col_ws))]

# Row y-bottoms (top-down)
y_sectA = TY0 - SH
y_hdr1  = y_sectA - H1
y_hdr2  = y_hdr1  - H2
y_rows  = [y_hdr2 - DH * (i + 1) for i in range(4)]
cal_bot = y_rows[-1]

# ── Section title ─────────────────────────────────────────────────────────────
cell(TX0, y_sectA, TW, SH,
     'Calibration  |  ICP 250 W · 10 mTorr · V_bias = -1000 V '
     '· 15 °C · t = 240 s  |  CF4=22/Ar=8 excluded (low reliability)',
     C_NAVY, fg='white', fs=9.5, bold=True)

# ── Group header (row 1): merged cells ───────────────────────────────────────
groups = [
    (0,  1,  'CF4/Ar\n[sccm]', C_BLUE),
    (1,  4,  'Etch Depth [nm]',       C_BLUE),
    (4,  7,  'Top CD [nm]',           C_BLUE),
    (7,  10, 'Bottom CD [nm]',        C_BLUE),
    (10, 13, 'Aspect Ratio',          C_BLUE),
]
for s, e, lbl, col in groups:
    x0 = col_xs[s]
    w  = sum(col_ws[s:e])
    cell(x0, y_hdr1, w, H1, lbl, col, fg='white', fs=9.0, bold=True)

# ── Sub-header (row 2) ────────────────────────────────────────────────────────
sub = ['', 'Exp.', 'Sim.', 'Error', 'Exp.', 'Sim.', 'Error',
       'Exp.', 'Sim.', 'Error', 'Exp.', 'Sim.', 'Error']
for txt, x0, w in zip(sub, col_xs, col_ws):
    cell(x0, y_hdr2, w, H2, txt, C_SUBHDR, fg='white', fs=8.5, bold=True)

# ── Data rows ─────────────────────────────────────────────────────────────────
row_bg = [C_ROW_A, C_ROW_B, C_ROW_A, C_ROW_B]
err_cols_map = {3: dep_err, 6: top_err, 9: bot_err, 12: ar_err}

for ri in range(4):
    rb   = row_bg[ri]
    yb   = y_rows[ri]
    row  = [
        cf4_ar[ri],
        f'{dep_e[ri]:.0f}',  f'{dep_s[ri]:.1f}',  fmt_e(dep_err[ri]),
        f'{top_e[ri]:.1f}',  f'{top_s[ri]:.1f}',  fmt_e(top_err[ri]),
        f'{bot_e[ri]:.1f}',  f'{bot_s[ri]:.1f}',  fmt_e(bot_err[ri]),
        f'{ar_e[ri]:.3f}',   f'{ar_s[ri]:.3f}',   fmt_e(ar_err[ri]),
    ]
    for ci, (txt, x0, w) in enumerate(zip(row, col_xs, col_ws)):
        if ci in err_cols_map:
            v = err_cols_map[ci][ri]
            cell(x0, yb, w, DH, txt, err_bg(v), fg=err_tx(v), fs=8.5, bold=True)
        elif ci == 0:
            cell(x0, yb, w, DH, txt, rb, fg=C_DARK, fs=8.5)
        else:
            cell(x0, yb, w, DH, txt, rb, fg=C_DARK, fs=8.5)

# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION PANEL
# ─────────────────────────────────────────────────────────────────────────────
# Columns: Metric | Exp. | This work | Error | Pass ≤10% | RODEo | RODEo err | Note
_vraw = [1.3, 0.85, 0.85, 0.72, 0.68, 0.85, 0.72, 1.5]
_vtot = sum(_vraw)
vcol_ws = [r / _vtot * TW for r in _vraw]
vcol_xs = [TX0 + sum(vcol_ws[:i]) for i in range(len(vcol_ws))]

y_sectB = cal_bot - GAP
y_vhdr  = y_sectB - SH - H1
y_vrows = [y_vhdr - DH * (i + 1) for i in range(2)]
val_bot = y_vrows[-1]

# ── Section title ─────────────────────────────────────────────────────────────
cell(TX0, y_sectB - SH, TW, SH,
     'External Validation  |  Chopra et al. (SPIE 2018)  ·  '
     'Plasma-Therm 790 CCP-RIE  ·  50 mTorr  ·  200 W  ·  '
     'CF4/Ar = 40/10 sccm  ·  t = 180 s  ·  cd0 = 65 nm',
     C_NAVY, fg='white', fs=9.5, bold=True)

# ── Column header ─────────────────────────────────────────────────────────────
vhdrs = ['Metric', 'Experiment\n[nm]', 'This work\n[nm]', 'Error',
         'Pass\n≤ 10%', 'RODEo\n[nm]', 'RODEo\nerror', 'Note']
for txt, x0, w in zip(vhdrs, vcol_xs, vcol_ws):
    cell(x0, y_vhdr, w, H1, txt, C_BLUE, fg='white', fs=9.0, bold=True)

# ── Data rows ─────────────────────────────────────────────────────────────────
val_labels = ['Height', 'Width']
val_notes  = ['k_rate_global fitted', 'forward prediction†']

for ri in range(2):
    rb   = [C_ROW_A, C_ROW_B][ri]
    yb   = y_vrows[ri]
    me   = val_me[ri]
    re_  = val_re[ri]
    ok10 = abs(me) < 10.0

    cells = [
        (val_labels[ri],        rb,       C_DARK,   8.5,  False, 'left'),
        (f'{val_exp[ri]:.1f}',  rb,       C_DARK,   8.5,  False, 'center'),
        (f'{val_mod[ri]:.1f}',  rb,       C_DARK,   8.5,  False, 'center'),
        (fmt_e(me),             err_bg(me), err_tx(me), 8.5, True, 'center'),
        ('PASS' if ok10 else 'FAIL',
         C_OK_BG if ok10 else C_WN_BG,
         C_OK_TX if ok10 else C_WN_TX, 9.0, True, 'center'),
        (f'{val_rod[ri]:.1f}',  rb,       C_DARK,   8.5,  False, 'center'),
        (fmt_e(re_),            C_WN_BG,  C_WN_TX,  8.5,  True,  'center'),
        (val_notes[ri],         rb,       C_NOTE,   7.8,  False, 'left'),
    ]
    for (txt, bg, fg, fs, bd, ha), x0, w in zip(cells, vcol_xs, vcol_ws):
        cell(x0, yb, w, DH, txt, bg, fg=fg, fs=fs, bold=bd, ha=ha)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
fy = val_bot - 0.022
ax.text(TX0, fy,
        '† Width (CD_top) is a free forward prediction — only Height was used '
        'to calibrate k_rate_global (reactor flux scale).  '
        'Error = (Sim − Exp) / Exp × 100%.  '
        'Green: |Error| ≤ 5%.  '
        'Calibration: 3-pass TRF with hinge-loss penalty (δ = 4%).',
        ha='left', va='top', fontsize=7.5, color='#5a6a7a',
        fontstyle='italic', clip_on=False)

# ─────────────────────────────────────────────────────────────────────────────
# Panel labels  (a) / (b)
# ─────────────────────────────────────────────────────────────────────────────
for label, y_pos in [('(a)', TY0 - SH / 2), ('(b)', y_sectB - SH / 2)]:
    ax.text(TX0 - 0.015, y_pos, label,
            ha='right', va='center', fontsize=10, fontweight='bold',
            color=C_DARK, clip_on=False)

_out = os.path.join(_HERE, 'figures', 'harc_v2_summary_table.png')
plt.savefig(_out, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f'Saved -> {_out}')
plt.show()
