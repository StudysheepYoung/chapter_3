import matplotlib
matplotlib.use('Agg')
import os
import json
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "ica_compare_results")
JSON_PATH  = os.path.join(OUTPUT_DIR, "ica_compare_metrics.json")
METHODS    = ['fastica', 'infomax', 'picard', 'jade', 'sobi', 'amuse']

with open(JSON_PATH) as f:
    results = json.load(f)

snr_db_before = results['before']['snr_db']
methods_bar   = [m for m in METHODS if m in results]
x             = np.arange(len(methods_bar))
snr_vals      = [results[m]['snr_db']       for m in methods_bar]
delta_vals    = [results[m]['delta_snr_db'] for m in methods_bar]
sf_vals       = [results[m]['sf_db']        for m in methods_bar]

colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#CCB974']
MODALITY = 'Hearing'

plots = [
    ('ica_compare_snr.png',   snr_vals,   'SNR (dB)',  'Denoised SNR'),
    ('ica_compare_dsnr.png',  delta_vals, 'dSNR (dB)', 'SNR Improvement'),
    ('ica_compare_sf.png',    sf_vals,    'SF (dB)',   'Suppression Factor SF'),
]

for fname, vals, ylabel, title in plots:
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='white')
    bars = ax.bar(x, vals, color=colors[:len(methods_bar)])
    ax.axhline(0, color='gray', linewidth=0.8)
    if title == 'Denoised SNR':
        ax.axhline(snr_db_before, color='red', linestyle='--', linewidth=1,
                   label=f'Before ICA {snr_db_before:.1f} dB')
        ax.legend(fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in methods_bar])
    ax.set_ylabel(ylabel)
    ax.set_title(f'{MODALITY} MEG — {title}', fontsize=11)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")
