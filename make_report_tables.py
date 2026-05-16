"""
보고서용 테이블 두 장 생성
- 실험 데이터와 AR 비교 테이블
- 역계산 결과 테이블
"""
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BG   = '#1a1a2e'
CELL = '#16213e'
HEAD = '#0f3460'
TXT  = 'white'
BOLD = '#e0e0e0'
ACC  = '#e94560'

def make_ar_comparison_table(save_path='report_ar_table.png'):
    rows = [
        ('6/24',               '6.519', '6.571'),
        ('10/20',              '6.780', '6.543', True),
        ('14/16',              '6.363', '6.230'),
        ('18/12',              '5.441', '5.679'),
        ('22/8 (제외)',         '4.397', '4.842'),
        ('7.50/22.50 (역계산)', '—',    '6.603', False, True),
    ]
    cols = ['CF4/Ar', 'AR (실험)', 'AR (시뮬.)']

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis('off')

    ax.text(0.5, 0.97, '실험 데이터와 비교',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=14, fontweight='bold', color=BOLD)

    col_w  = [0.40, 0.28, 0.32]
    x_starts = [0.0, 0.40, 0.68]
    row_h  = 0.11
    top    = 0.82

    # 헤더
    for j, (c, x, w) in enumerate(zip(cols, x_starts, col_w)):
        rect = patches.FancyBboxPatch((x, top), w - 0.005, row_h,
                                      boxstyle='round,pad=0.005',
                                      facecolor=HEAD, edgecolor='#334466',
                                      linewidth=0.8, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + w/2, top + row_h/2, c,
                transform=ax.transAxes, ha='center', va='center',
                fontsize=10, fontweight='bold', color=BOLD)

    for i, row in enumerate(rows):
        y    = top - (i + 1) * row_h
        bold = len(row) > 3 and row[3]
        new  = len(row) > 4 and row[4]
        bg   = '#1e3050' if bold else ('#1e3050' if new else CELL)
        for j, (val, x, w) in enumerate(zip(row[:3], x_starts, col_w)):
            rect = patches.FancyBboxPatch((x, y), w - 0.005, row_h,
                                          boxstyle='round,pad=0.005',
                                          facecolor=bg, edgecolor='#2a3a5a',
                                          linewidth=0.6, transform=ax.transAxes)
            ax.add_patch(rect)
            fw = 'bold' if (bold or new) else 'normal'
            fc = ACC if (new and j == 2) else (BOLD if bold else TXT)
            fs = 10 if bold else (9.5 if new else 9.5)
            lbl = val
            if bold and j == 2:
                lbl = val + '  ← 실험 최대'
            ax.text(x + w/2, y + row_h/2, lbl,
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=fs, fontweight=fw, color=fc)

    plt.tight_layout(pad=0.3)
    plt.savefig(save_path, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'저장 → {save_path}')


def make_inverse_result_table(save_path='report_inverse_table.png'):
    opt_rows = [
        ('CF4 유량', '7.50 sccm'),
        ('Ar 유량',  '22.50 sccm'),
        ('CF4 분율', '25.0 %'),
    ]
    pred_rows = [
        ('식각 깊이',   '1380 nm'),
        ('CD_top',     '209 nm'),
        ('CD_bot',     '54 nm'),
        ('AR (최대)',   '6.603', True),
        ('Taper index','0.741'),
    ]

    fig, ax = plt.subplots(figsize=(6.5, 7.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis('off')

    # 제목
    ax.text(0.5, 0.975,
            '역계산 결과 (t=240s, V_bias=−1000V, P=250W, 10mTorr)',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=10.5, fontweight='bold', color=BOLD)

    def draw_table(title, rows, y_top, col_w=(0.42, 0.58)):
        ax.text(0.04, y_top + 0.025, title,
                transform=ax.transAxes, fontsize=10, fontweight='bold', color=BOLD)
        rh = 0.085
        xs = [0.04, 0.04 + col_w[0]]
        for i, row in enumerate(rows):
            y    = y_top - i * rh
            spec = len(row) > 2 and row[2]
            for j, (val, x, w) in enumerate(zip(row[:2], xs, col_w)):
                bg = HEAD if i == 0 and False else ('#1e3050' if spec else CELL)
                rect = patches.FancyBboxPatch((x, y - rh), w - 0.01, rh,
                                              boxstyle='round,pad=0.005',
                                              facecolor=bg, edgecolor='#2a3a5a',
                                              linewidth=0.6, transform=ax.transAxes)
                ax.add_patch(rect)
                fw = 'bold' if spec else 'normal'
                fc = ACC  if (spec and j == 1) else (BOLD if j == 0 else TXT)
                ax.text(x + w/2, y - rh/2, val,
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=10, fontweight=fw, color=fc)
        return y_top - len(rows) * rh

    y = 0.90
    ax.text(0.04, y, '최적 공정 조건',
            transform=ax.transAxes, fontsize=10.5, fontweight='bold', color=BOLD)
    y -= 0.03
    rh = 0.085
    xs   = [0.04, 0.04 + 0.42]
    col_w = [0.42, 0.54]
    for row in opt_rows:
        for j, (val, x, w) in enumerate(zip(row, xs, col_w)):
            rect = patches.FancyBboxPatch((x, y - rh), w - 0.01, rh,
                                          boxstyle='round,pad=0.005',
                                          facecolor=CELL, edgecolor='#2a3a5a',
                                          linewidth=0.6, transform=ax.transAxes)
            ax.add_patch(rect)
            ax.text(x + w/2, y - rh/2, val,
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=10, color=TXT if j else BOLD)
        y -= rh

    y -= 0.04
    ax.text(0.04, y, '최적 조건에서 예측값',
            transform=ax.transAxes, fontsize=10.5, fontweight='bold', color=BOLD)
    y -= 0.03
    for row in pred_rows:
        spec = len(row) > 2 and row[2]
        for j, (val, x, w) in enumerate(zip(row[:2], xs, col_w)):
            bg = '#1e3050' if spec else CELL
            rect = patches.FancyBboxPatch((x, y - rh), w - 0.01, rh,
                                          boxstyle='round,pad=0.005',
                                          facecolor=bg, edgecolor='#2a3a5a',
                                          linewidth=0.6, transform=ax.transAxes)
            ax.add_patch(rect)
            fw = 'bold' if spec else 'normal'
            fc = ACC if (spec and j == 1) else (BOLD if j == 0 else TXT)
            ax.text(x + w/2, y - rh/2, val,
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=10, fontweight=fw, color=fc)
        y -= rh

    plt.tight_layout(pad=0.3)
    plt.savefig(save_path, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'저장 → {save_path}')


make_ar_comparison_table('report_ar_table.png')
make_inverse_result_table('report_inverse_table.png')
print('완료.')
