import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── paths ───────────────────────────────────────────────────────────────────
BASE    = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
OUT_DIR = os.path.join(BASE, 'signal_parts')
os.makedirs(OUT_DIR, exist_ok=True)

t = np.arange(512) / 512.0

def norm(x):
    m = np.max(np.abs(x))
    return x / m if m > 0 else x

def make_ecg(n=512, fs=512):
    """Synthetic ECG via sum of Gaussians (PQRST)."""
    t_ = np.arange(n) / fs
    ecg = np.zeros(n)
    for mu, sig, amp in [
        (0.35, 0.025,  0.15),  # P
        (0.47, 0.010, -0.10),  # Q
        (0.50, 0.008,  1.00),  # R
        (0.53, 0.010, -0.25),  # S
        (0.62, 0.030,  0.35),  # T
    ]:
        ecg += amp * np.exp(-0.5 * ((t_ - mu) / sig) ** 2)
    return ecg

def save_waveform(sig, filename, color='#4C72B0', show_xaxis=False):
    """Save a single waveform as a clean PNG (5×1 inch)."""
    fig, ax = plt.subplots(figsize=(5, 1), dpi=300)
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.18)

    ax.plot(t, norm(sig), color=color, linewidth=0.9)
    ax.set_yticks([])
    ax.set_xlim(t[0], t[-1])

    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)

    if show_xaxis:
        ax.spines['bottom'].set_linewidth(0.6)
        ax.tick_params(axis='x', labelsize=7)
        ax.set_xlabel('Time (s)', fontsize=8)
    else:
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])

    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved → {path}')


# ── load S_sources (EEG×4, EOG, EMG) ────────────────────────────────────────
S_all = np.load(os.path.join(BASE, 'multichannel_ica', 'S_sources.npy'))
S = S_all[0]   # (6, 512): EEG×4, EOG, EMG
gwn = np.random.randn(512)

# group 1: EEG×4, EOG, EMG, GWN
for i in range(4):
    save_waveform(S[i], f'eeg{i+1}.png', color='#4C72B0')
save_waveform(S[4], 'eog.png', color='#DD8452')
save_waveform(S[5], 'emg.png', color='#55A868')
save_waveform(gwn,  'gwn.png', color='#8172B2')

# group 2: EEG, EMG, EOG, GWN
save_waveform(S[0], 'artifact_eeg.png', color='#4C72B0')
save_waveform(S[5], 'artifact_emg.png', color='#55A868')
save_waveform(S[4], 'artifact_eog.png', color='#DD8452')
save_waveform(gwn,  'artifact_gwn.png', color='#8172B2')

# ── mixing diagram (standalone) ─────────────────────────────────────────────
A_all = np.load(os.path.join(BASE, 'multichannel_ica', 'A_mixing.npy'))
A = A_all[0]
gwn_col = np.random.randn(16, 1); gwn_col /= np.linalg.norm(gwn_col)
A_ext = np.hstack([A, gwn_col])
S_ext = np.vstack([S, np.random.randn(1, 512)])

labels_mix = ['EEG 1', 'EEG 2', 'EEG 3', 'EEG 4', 'EOG', 'EMG', 'GWN']
colors_mix = ['#4C72B0'] * 4 + ['#DD8452', '#55A868', '#8172B2']
ys = np.linspace(0.88, 0.12, 7)

fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
fig.patch.set_facecolor('white')
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ax.set_title('Signal mixing process', fontsize=11, pad=6)

src_x, box_w, box_h = 0.12, 0.22, 0.08
mix_x, out_x = 0.52, 0.84

for label, yc, col in zip(labels_mix, ys, colors_mix):
    ax.add_patch(mpatches.FancyBboxPatch(
        (src_x - box_w/2, yc - box_h/2), box_w, box_h,
        boxstyle='round,pad=0.01', linewidth=1,
        edgecolor=col, facecolor=col + '22'))
    ax.text(src_x, yc, label, ha='center', va='center',
            fontsize=9, color=col, fontweight='bold')
    ax.annotate('',
        xy=(mix_x - 0.075, 0.5 + (yc - 0.5) * 0.18),
        xytext=(src_x + box_w/2, yc),
        arrowprops=dict(arrowstyle='->', color=col, lw=1.2))

ax.text((src_x + box_w/2 + mix_x - 0.075)/2, 0.5 + 0.12,
        'S (7x512)', ha='center', va='bottom', fontsize=8, color='#555555')

mbox_w, mbox_h = 0.15, 0.14
ax.add_patch(mpatches.FancyBboxPatch(
    (mix_x - mbox_w/2, 0.5 - mbox_h/2), mbox_w, mbox_h,
    boxstyle='round,pad=0.01', linewidth=1.5,
    edgecolor='#555555', facecolor='#55555515'))
ax.text(mix_x, 0.5, 'xA', ha='center', va='center',
        fontsize=13, color='#333333', fontweight='bold')
ax.text(mix_x, 0.5 - mbox_h/2 - 0.03, 'A (16x7)',
        ha='center', va='top', fontsize=7.5, color='#666666')

ax.annotate('',
    xy=(out_x - 0.07, 0.5), xytext=(mix_x + mbox_w/2, 0.5),
    arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))
ax.text((mix_x + mbox_w/2 + out_x - 0.07)/2, 0.5 + 0.03,
        'X (16x512)', ha='center', va='bottom', fontsize=8, color='#555555')

nbox_w, nbox_h = 0.26, 0.12
ax.add_patch(mpatches.FancyBboxPatch(
    (out_x - nbox_w/2, 0.5 - nbox_h/2), nbox_w, nbox_h,
    boxstyle='round,pad=0.015', linewidth=1.5,
    edgecolor='#C44E52', facecolor='#C44E5222'))
ax.text(out_x, 0.5 + 0.015, 'Noisy EEG', ha='center', va='center',
        fontsize=10, color='#C44E52', fontweight='bold')
ax.text(out_x, 0.5 - 0.025, '(ch. 1 shown)', ha='center', va='center',
        fontsize=7, color='#C44E52')

mix_path = os.path.join(OUT_DIR, 'mixing_diagram.png')
plt.savefig(mix_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'Saved → {mix_path}')

# ── A matrix heatmap ─────────────────────────────────────────────────────────
A_plot = np.load(os.path.join(BASE, 'multichannel_ica', 'A_mixing.npy'))[0]  # (16, 6)

fig, ax = plt.subplots(figsize=(3.5, 5), dpi=300)
fig.patch.set_facecolor('white')
im = ax.imshow(A_plot, aspect='auto', cmap='RdBu_r',
               vmin=-np.abs(A_plot).max(), vmax=np.abs(A_plot).max())
ax.set_xticks(range(6))
ax.set_xticklabels(['EEG1', 'EEG2', 'EEG3', 'EEG4', 'EOG', 'EMG'],
                   fontsize=8, rotation=30, ha='right')
ax.set_yticks(range(16))
ax.set_yticklabels([f'ch{i+1}' for i in range(16)], fontsize=7)
ax.set_title('Mixing matrix A  (16×6)', fontsize=10, pad=6)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
a_path = os.path.join(OUT_DIR, 'A_matrix.png')
plt.savefig(a_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'Saved → {a_path}')

# ── multichannel timeseries helper ───────────────────────────────────────────
def save_multichannel(data, filename, title, color='#4C72B0'):
    n_ch = data.shape[0]
    t_x  = np.arange(data.shape[1]) / 512.0
    fig, axes = plt.subplots(n_ch, 1, figsize=(6, 8), dpi=300,
                             sharex=True,
                             gridspec_kw=dict(hspace=0.05,
                                              left=0.12, right=0.98,
                                              top=0.95, bottom=0.06))
    fig.patch.set_facecolor('white')
    for i, ax in enumerate(axes):
        y = data[i]; m = np.max(np.abs(y))
        ax.plot(t_x, y / m if m > 0 else y, color=color, linewidth=0.6)
        ax.set_yticks([])
        ax.set_ylabel(f'ch{i+1}', fontsize=6, rotation=0, labelpad=22,
                      va='center', ha='right')
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)
        if i < n_ch - 1:
            ax.spines['bottom'].set_visible(False)
        else:
            ax.spines['bottom'].set_linewidth(0.5)
            ax.set_xlabel('Time (s)', fontsize=8)
            ax.tick_params(axis='x', labelsize=7)
    axes[0].set_title(title, fontsize=10, pad=6)
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved → {path}')

save_multichannel(
    np.load(os.path.join(BASE, 'multichannel_ica', 'X_noisy.npy'))[0],
    'noisy_eeg_multichannel.png', 'Noisy EEG  X (16 channels)', color='#4C72B0')

save_multichannel(
    np.load(os.path.join(BASE, 'multichannel_ica', 'X_clean.npy'))[0],
    'clean_eeg_multichannel.png', 'Clean EEG  X (16 channels)', color='#2ca02c')
