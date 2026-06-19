"""
黑白打印风格预览 — 生成 4 种方案的样例图
  style_A: 纯线型区分
  style_B: 线型 + 标记点
  style_C: 灰度 + 线型
  style_D: 线宽 + 线型 + 标记点（综合）
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'data', 'multichannel_ica', 'ica_compare')
os.makedirs(OUT_DIR, exist_ok=True)

rng = np.random.default_rng(0)
t = np.linspace(0, 0.5, 256)

# 模拟 3 条曲线：noisy / denoised / GT
noisy    = np.sin(2 * np.pi * 8 * t) + 0.6 * rng.standard_normal(len(t))
gt       = np.sin(2 * np.pi * 8 * t)
denoised = gt + 0.15 * rng.standard_normal(len(t))

# 模拟 residual_all：7 条曲线（noisy-GT + 6 methods）
METHODS = ['fastica', 'infomax', 'picard', 'jade', 'sobi', 'amuse']
noisy_res = noisy - gt
method_res = {m: gt * rng.uniform(0.05, 0.25) + rng.standard_normal(len(t)) * 0.1
              for m in METHODS}

# ── 方案定义 ──────────────────────────────────────────────────────────────────
# 3-line style sets  (noisy, denoised, GT)
STYLES_3 = {
    'A_linestyle': [
        dict(lw=1.5, ls='-',  color='k',   alpha=0.5),
        dict(lw=1.5, ls='--', color='k',   alpha=1.0),
        dict(lw=1.5, ls=':',  color='k',   alpha=1.0),
    ],
    'B_marker': [
        dict(lw=1.2, ls='-',  color='k', alpha=0.5, marker='', markevery=20),
        dict(lw=1.2, ls='--', color='k', alpha=1.0, marker='s', markevery=20, ms=4),
        dict(lw=1.2, ls='-',  color='k', alpha=1.0, marker='o', markevery=20, ms=4),
    ],
    'C_gray': [
        dict(lw=1.5, ls='-',  color='0.7'),
        dict(lw=1.5, ls='--', color='0.3'),
        dict(lw=2.0, ls='-',  color='0.0'),
    ],
    'D_combined': [
        dict(lw=1.0, ls='-',   color='0.6', marker='',  markevery=25),
        dict(lw=1.8, ls='--',  color='0.2', marker='^', markevery=25, ms=4),
        dict(lw=2.2, ls='-',   color='0.0', marker='o', markevery=25, ms=4),
    ],
}

# 7-line style sets  (noisy-GT + 6 methods)
STYLES_7 = {
    'A_linestyle': [
        dict(lw=1.2, ls='-',   color='k', alpha=0.4),
        dict(lw=1.2, ls='--',  color='k'),
        dict(lw=1.2, ls=':',   color='k'),
        dict(lw=1.2, ls='-.',  color='k'),
        dict(lw=1.2, ls=(0,(5,1)), color='k'),
        dict(lw=1.2, ls=(0,(3,1,1,1)), color='k'),
        dict(lw=1.2, ls=(0,(1,1)), color='k'),
    ],
    'B_marker': [
        dict(lw=1.0, ls='-',  color='k', alpha=0.4, marker=''),
        dict(lw=1.0, ls='--', color='k', marker='o',  markevery=20, ms=3.5),
        dict(lw=1.0, ls='--', color='k', marker='s',  markevery=20, ms=3.5),
        dict(lw=1.0, ls='--', color='k', marker='^',  markevery=20, ms=3.5),
        dict(lw=1.0, ls='--', color='k', marker='D',  markevery=20, ms=3.5),
        dict(lw=1.0, ls='--', color='k', marker='v',  markevery=20, ms=3.5),
        dict(lw=1.0, ls='--', color='k', marker='P',  markevery=20, ms=3.5),
    ],
    'C_gray': [
        dict(lw=1.2, ls='-',  color='0.75'),
        dict(lw=1.5, ls='--', color='0.0'),
        dict(lw=1.5, ls='--', color='0.2'),
        dict(lw=1.5, ls='--', color='0.35'),
        dict(lw=1.5, ls=':',  color='0.0'),
        dict(lw=1.5, ls=':',  color='0.2'),
        dict(lw=1.5, ls='-.',  color='0.0'),
    ],
    'D_combined': [
        dict(lw=1.0, ls='-',  color='0.7', marker=''),
        dict(lw=1.5, ls='--', color='0.0', marker='o',  markevery=22, ms=3.5),
        dict(lw=1.5, ls='--', color='0.2', marker='s',  markevery=22, ms=3.5),
        dict(lw=1.5, ls=':',  color='0.0', marker='^',  markevery=22, ms=3.5),
        dict(lw=1.5, ls=':',  color='0.2', marker='D',  markevery=22, ms=3.5),
        dict(lw=1.5, ls='-.', color='0.0', marker='v',  markevery=22, ms=3.5),
        dict(lw=1.5, ls='-.', color='0.2', marker='P',  markevery=22, ms=3.5),
    ],
}

LABELS_3 = ['Noisy (raw)', 'Denoised', 'GT (clean)']
LABELS_7 = ['Noisy−GT'] + [f'{m}−GT' for m in METHODS]

for style_name, styles in STYLES_3.items():
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), dpi=150)
    fig.patch.set_facecolor('white')

    # 左：时域波形
    ax = axes[0]
    for s, label, data in zip(styles, LABELS_3, [noisy, denoised, gt]):
        ax.plot(t, data, label=label, **s)
    ax.set_title('Amplitude (time domain)', fontsize=10)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Amplitude')
    ax.legend(fontsize=8)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

    # 右：PSD（semilogy）
    from scipy.signal import welch
    ax2 = axes[1]
    for s, label, data in zip(styles, LABELS_3, [noisy, denoised, gt]):
        f, p = welch(data, fs=512, nperseg=128)
        ax2.semilogy(f, p, label=label, **s)
    ax2.set_title('PSD', fontsize=10)
    ax2.set_xlabel('Frequency (Hz)'); ax2.set_ylabel('PSD')
    ax2.legend(fontsize=8)
    for sp in ['top', 'right']: ax2.spines[sp].set_visible(False)

    fig.suptitle(f'Style {style_name}  —  3-line plots', fontsize=11, y=1.01)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f'preview_3line_{style_name}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved → {out}')

for style_name, styles in STYLES_7.items():
    fig, ax = plt.subplots(figsize=(10, 3.5), dpi=150)
    fig.patch.set_facecolor('white')
    all_data = [noisy_res] + list(method_res.values())
    for s, label, data in zip(styles, LABELS_7, all_data):
        ax.plot(t, data, label=label, **s)
    ax.axhline(0, color='k', lw=0.6, ls='--')
    ax.set_title(f'Style {style_name}  —  Residual (7 lines)', fontsize=11)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Residual')
    ax.legend(fontsize=7, ncol=4, loc='upper right')
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f'preview_7line_{style_name}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved → {out}')

print('\nDone. 8 preview images saved.')
