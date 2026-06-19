"""
ICA EEG 去伪影完整流程
输入: data/multichannel_ica/ 下的仿真多通道数据
输出: 去伪影后的信号 + 定量评估结果
"""
import numpy as np
import mne
from mne_icalabel.iclabel import iclabel_label_components
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 配置 ──────────────────────────────────────────────────
SFREQ        = 512.0
EPOCH_IDX    = 0        # 可视化第几个epoch


def get_ch_names(n_ch):
    all_ch = mne.channels.make_standard_montage('standard_1020').ch_names
    idx = np.linspace(0, len(all_ch) - 1, n_ch, dtype=int)
    return [all_ch[i] for i in idx]

def load_data(data_dir):
    X_noisy   = np.load(os.path.join(data_dir, 'X_noisy.npy'))
    X_clean   = np.load(os.path.join(data_dir, 'X_clean.npy'))
    S_sources = np.load(os.path.join(data_dir, 'S_sources.npy'))
    return X_noisy, X_clean, S_sources


def make_raw(epoch, sfreq=SFREQ):
    """把单个epoch (n_ch, T) 包成带标准montage的 MNE RawArray"""
    n_ch = epoch.shape[0]
    ch_names = get_ch_names(n_ch)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw  = mne.io.RawArray(epoch, info, verbose=False)
    raw.set_montage(mne.channels.make_standard_montage('standard_1020'), verbose=False)
    return raw


def run_ica(raw, random_state=42):
    """在原始数据上 fit ICA；同时返回滤波+CAR副本供 ICLabel 识别用"""
    raw_filt = raw.copy().filter(1., 100., method='fir', fir_window='hamming',
                                 filter_length='10s', verbose=False)
    raw_filt.set_eeg_reference('average', projection=False, verbose=False)
    n_components = raw.info['nchan']   # 等于通道数，MNE 自动降为 n_ch-1
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method='infomax',
        fit_params=dict(extended=True),
        random_state=random_state,
        max_iter=1000,
        verbose=False
    )
    ica.fit(raw, verbose=False)   # fit 在原始数据上
    return ica, raw_filt


ARTIFACT_LABELS = {'muscle artifact', 'eye blink', 'heart beat', 'line noise', 'channel noise', 'other'}
ICLABEL_NAMES   = ['brain', 'muscle artifact', 'eye blink', 'heart beat', 'line noise', 'channel noise', 'other']


def identify_artifact_components(ica, raw):
    """用 ICLabel (onnx backend) 识别伪影成分，返回 exclude 列表"""
    proba = iclabel_label_components(raw, ica, backend='onnx')
    labels = [ICLABEL_NAMES[row.argmax()] for row in proba]
    exclude = []
    for i, (label, row) in enumerate(zip(labels, proba)):
        marker = '→ exclude' if label in ARTIFACT_LABELS else ''
        print(f'  IC{i:02d}: {label:<20s} (p={row.max():.3f}) {marker}')
        if label in ARTIFACT_LABELS:
            exclude.append(i)
    return exclude


def apply_ica(ica, raw, exclude):
    """剔除伪影成分，重建信号"""
    ica.exclude = exclude
    raw_clean = raw.copy()
    ica.apply(raw_clean, verbose=False)
    return raw_clean


def evaluate(raw_noisy, raw_reconstructed, raw_groundtruth):
    """
    定量评估：
    - SNR improvement
    - 逐通道相关系数
    """
    X_in  = raw_noisy.get_data()
    X_out = raw_reconstructed.get_data()
    X_gt  = raw_groundtruth.get_data()

    T = min(X_in.shape[1], X_out.shape[1], X_gt.shape[1])
    X_in, X_out, X_gt = X_in[:, :T], X_out[:, :T], X_gt[:, :T]

    def snr(signal, noise):
        return 10 * np.log10(np.mean(signal**2) / (np.mean(noise**2) + 1e-12))

    snr_before = snr(X_gt, X_in - X_gt)
    snr_after  = snr(X_gt, X_out - X_gt)

    corr = np.mean([abs(np.corrcoef(X_out[i], X_gt[i])[0, 1])
                    for i in range(X_out.shape[0])])

    print(f'  SNR before ICA : {snr_before:.2f} dB')
    print(f'  SNR after  ICA : {snr_after:.2f} dB  (↑{snr_after - snr_before:.2f} dB)')
    print(f'  Mean channel corr (output vs clean): {corr:.4f}')
    return snr_before, snr_after, corr


def plot_comparison(raw_noisy, raw_clean_ica, raw_groundtruth, out_path, epoch_idx):
    """绘制第一个通道的对比波形"""
    ch = 0
    t = np.arange(raw_noisy.get_data().shape[1]) / SFREQ

    T = min(raw_noisy.get_data().shape[1],
            raw_clean_ica.get_data().shape[1],
            raw_groundtruth.get_data().shape[1])

    sig_noisy = raw_noisy.get_data()[ch, :T]
    sig_ica   = raw_clean_ica.get_data()[ch, :T]
    sig_gt    = raw_groundtruth.get_data()[ch, :T]
    t = t[:T]

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    # 上图：三条线叠加
    axes[0].plot(t, sig_noisy, lw=0.8, color='tomato',    alpha=0.7, label='Noisy')
    axes[0].plot(t, sig_ica,   lw=0.8, color='steelblue', alpha=0.9, label='After ICA')
    axes[0].plot(t, sig_gt,    lw=0.8, color='green',     alpha=0.9, label='Ground Truth')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].set_ylabel('Amplitude')

    # 下图：残差（去噪前后与GT的差）
    axes[1].plot(t, sig_noisy - sig_gt, lw=0.7, color='tomato',    alpha=0.8, label='Noisy − GT')
    axes[1].plot(t, sig_ica   - sig_gt, lw=0.7, color='steelblue', alpha=0.8, label='ICA − GT')
    axes[1].axhline(0, color='k', lw=0.5, ls='--')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].set_ylabel('Residual')
    axes[1].set_xlabel('Time (s)')

    fig.suptitle(f'Epoch {epoch_idx} — Channel EEG001')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  Plot saved: {out_path}')


# ── 主流程 ────────────────────────────────────────────────
if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'multichannel_ica')
    out_dir  = os.path.join(data_dir, 'ica_results')
    os.makedirs(out_dir, exist_ok=True)

    X_noisy, X_clean, S_sources = load_data(data_dir)
    print(f'Loaded  X_noisy:{X_noisy.shape}  X_clean:{X_clean.shape}')
    all_snr_before, all_snr_after, all_corr = [], [], []

    for ep in range(X_noisy.shape[0]):
        print(f'\n── Epoch {ep:03d} ──')

        raw_noisy = make_raw(X_noisy[ep])
        raw_gt    = make_raw(X_clean[ep])

        # 1. 滤波+CAR副本上拟合 ICA
        ica, raw_filt = run_ica(raw_noisy)

        # 2. ICLabel 识别伪影成分（用滤波副本）
        exclude = identify_artifact_components(ica, raw_filt)

        # 3. 去除伪影（作用于原始未滤波信号）
        raw_reconstructed = apply_ica(ica, raw_noisy, exclude)

        # 4. 评估（直接和原始 X_clean 比较，无任何预处理）
        snr_b, snr_a, corr = evaluate(raw_noisy, raw_reconstructed, raw_gt)
        all_snr_before.append(snr_b)
        all_snr_after.append(snr_a)
        all_corr.append(corr)

        # 5. 可视化（只保存指定epoch）
        if ep == EPOCH_IDX:
            plot_comparison(
                raw_noisy, raw_reconstructed, raw_gt,
                out_path=os.path.join(out_dir, f'epoch{ep:03d}_comparison.png'),
                epoch_idx=ep
            )

    # ── 汇总统计 ─────────────────────────────────────────
    print('\n══ 全局统计 ══')
    print(f'  SNR before : {np.mean(all_snr_before):.2f} ± {np.std(all_snr_before):.2f} dB')
    print(f'  SNR after  : {np.mean(all_snr_after):.2f} ± {np.std(all_snr_after):.2f} dB')
    print(f'  SNR gain   : {np.mean(all_snr_after) - np.mean(all_snr_before):.2f} dB')
    print(f'  Mean corr  : {np.mean(all_corr):.4f} ± {np.std(all_corr):.4f}')

    np.save(os.path.join(out_dir, 'snr_before.npy'), np.array(all_snr_before))
    np.save(os.path.join(out_dir, 'snr_after.npy'),  np.array(all_snr_after))
    np.save(os.path.join(out_dir, 'corr.npy'),        np.array(all_corr))
    print(f'\nResults saved to {out_dir}')
