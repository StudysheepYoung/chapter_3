"""
多种 ICA 算法对比：FastICA / Infomax(extended) / Picard / JADE / SOBI
输入: data/multichannel_ica/
输出: 汇总表 + 对比图
"""
import numpy as np
import mne
from mne_icalabel.iclabel import iclabel_label_components
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import eigh, svd

# ── 配置 ──────────────────────────────────────────────────
SFREQ        = 512.0
EPOCH_IDX    = 0


def get_ch_names(n_ch):
    all_ch = mne.channels.make_standard_montage('standard_1020').ch_names
    idx = np.linspace(0, len(all_ch) - 1, n_ch, dtype=int)
    return [all_ch[i] for i in idx]

# ══════════════════════════════════════════════════════════
# JADE 实现（基于四阶累积量联合对角化）
# ══════════════════════════════════════════════════════════
def jade_ica(X, n_comp=None):
    """
    JADE ICA.  X: (n_ch, T).  返回 (W, S): W @ X ≈ S.
    参考: Cardoso & Souloumiac (1993).
    """
    n_ch, T = X.shape
    if n_comp is None:
        n_comp = n_ch

    # 白化
    X = X - X.mean(axis=1, keepdims=True)
    cov = X @ X.T / T
    d, E = np.linalg.eigh(cov)
    idx = np.argsort(d)[::-1][:n_comp]
    d, E = d[idx], E[:, idx]
    W_white = (E / np.sqrt(d)).T          # (n_comp, n_ch)
    Z = W_white @ X                        # (n_comp, T)

    # 四阶累积量矩阵集合
    n = n_comp
    CM = np.zeros((n * n, n * n))
    for p in range(n):
        for q in range(p, n):
            zpzq = Z[p] * Z[q]                          # (T,)
            Qpq = (Z * zpzq[np.newaxis, :]) @ Z.T / T  # (n, n)
            if p == q:
                Qpq -= np.eye(n)
            else:
                cross = np.mean(zpzq)
                Qpq -= cross * np.ones((n, n))
            w = 1.0 if p == q else np.sqrt(2)
            CM += w * np.kron(Qpq, Qpq)

    # 联合对角化（用 Jacobi sweeps）
    V = _joint_diag(CM, n)
    W = V.T @ W_white
    S = W @ X
    return W, S


def _joint_diag(CM, n, n_sweeps=100):
    """简化版联合对角化，返回旋转矩阵 V (n×n)"""
    V = np.eye(n)
    for _ in range(n_sweeps):
        for p in range(n - 1):
            for q in range(p + 1, n):
                # 提取 2×2 子问题
                Vp = V[:, p]
                Vq = V[:, q]
                # Givens 旋转角
                g = np.array([
                    CM[p * n + p, q * n + q] - CM[q * n + q, p * n + p],
                    CM[p * n + p, p * n + p] - CM[q * n + q, q * n + q],
                    CM[p * n + q, p * n + q] + CM[q * n + p, q * n + p],
                ])
                ton = g[0] / (g[1] + 1e-12)
                theta = 0.5 * np.arctan2(ton, 1)
                c, s = np.cos(theta), np.sin(theta)
                V[:, p] = c * Vp + s * Vq
                V[:, q] = -s * Vp + c * Vq
    return V


# ══════════════════════════════════════════════════════════
# SOBI 实现（二阶盲辨识，多时延协方差联合对角化）
# ══════════════════════════════════════════════════════════
def sobi_ica(X, lags=None, n_comp=None):
    """
    SOBI ICA.  X: (n_ch, T).  返回 (W, S).
    参考: Belouchrani et al. (1997).
    """
    n_ch, T = X.shape
    if n_comp is None:
        n_comp = n_ch
    if lags is None:
        lags = list(range(1, min(100, T // 4)))

    X = X - X.mean(axis=1, keepdims=True)

    # 白化
    cov0 = X @ X.T / T
    d, E = np.linalg.eigh(cov0)
    idx = np.argsort(d)[::-1][:n_comp]
    d, E = d[idx], E[:, idx]
    W_white = (E / np.sqrt(np.maximum(d, 1e-12))).T
    Z = W_white @ X

    # 多时延协方差矩阵集合
    Rs = []
    for lag in lags:
        R = Z[:, lag:] @ Z[:, :-lag].T / (T - lag)
        Rs.append((R + R.T) / 2)

    # 联合对角化（Jacobi）
    V = _joint_diag_sobi(Rs, n_comp)
    W = V.T @ W_white
    S = W @ X
    return W, S


def _joint_diag_sobi(Ms, n, n_sweeps=30):
    V = np.eye(n)
    for _ in range(n_sweeps):
        for p in range(n - 1):
            for q in range(p + 1, n):
                num = sum(M[p, q] * (M[p, p] - M[q, q]) for M in Ms)
                den = sum((M[p, p] - M[q, q]) ** 2 - M[p, q] ** 2 for M in Ms) + 1e-12
                theta = 0.5 * np.arctan2(2 * num, den)
                c, s = np.cos(theta), np.sin(theta)
                Vp = V[:, p].copy()
                Vq = V[:, q].copy()
                V[:, p] = c * Vp + s * Vq
                V[:, q] = -s * Vp + c * Vq
                for M in Ms:
                    Mp = M[p, :].copy(); Mq = M[q, :].copy()
                    M[p, :] = c * Mp + s * Mq
                    M[q, :] = -s * Mp + c * Mq
                    Mp = M[:, p].copy(); Mq = M[:, q].copy()
                    M[:, p] = c * Mp + s * Mq
                    M[:, q] = -s * Mp + c * Mq
    return V


# ══════════════════════════════════════════════════════════
# 通用工具
# ══════════════════════════════════════════════════════════
def load_data(data_dir):
    X_noisy   = np.load(os.path.join(data_dir, 'X_noisy.npy'))
    X_clean   = np.load(os.path.join(data_dir, 'X_clean.npy'))
    S_sources = np.load(os.path.join(data_dir, 'S_sources.npy'))
    return X_noisy, X_clean, S_sources


def make_raw(epoch, sfreq=SFREQ):
    n_ch = epoch.shape[0]
    ch_names = get_ch_names(n_ch)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw  = mne.io.RawArray(epoch, info, verbose=False)
    raw.set_montage(mne.channels.make_standard_montage('standard_1020'), verbose=False)
    return raw


def run_mne_ica(raw, method):
    n_components = raw.info['nchan']
    fit_params = dict(extended=True) if method == 'infomax' else {}
    ica = mne.preprocessing.ICA(
        n_components=n_components, method=method,
        fit_params=fit_params, random_state=42,
        max_iter=1000, verbose=False
    )
    ica.fit(raw, verbose=False)
    return ica


ARTIFACT_LABELS = {'muscle artifact', 'eye blink', 'heart beat', 'line noise', 'channel noise', 'other'}
ICLABEL_NAMES   = ['brain', 'muscle artifact', 'eye blink', 'heart beat', 'line noise', 'channel noise', 'other']


def _iclabel_exclude(ica, raw):
    """用 ICLabel (onnx) 识别伪影成分，返回 exclude 列表"""
    proba = iclabel_label_components(raw, ica, backend='onnx')
    labels = [ICLABEL_NAMES[row.argmax()] for row in proba]
    return [i for i, label in enumerate(labels) if label in ARTIFACT_LABELS]


def identify_and_remove_mne(ica, raw):
    """用 ICLabel 识别伪影成分，返回重建信号 (n_ch, T)"""
    exclude = _iclabel_exclude(ica, raw)
    ica.exclude = exclude
    raw_out = raw.copy()
    ica.apply(raw_out, verbose=False)
    return raw_out.get_data()


def identify_and_remove_custom(W, X, raw):
    """
    JADE/SOBI: 把分解结果注入 infomax ICA 对象，用 ICLabel 识别，再重建。
    W: (n_comp, n_ch) 解混矩阵
    """
    ica = mne.preprocessing.ICA(
        n_components=W.shape[0], method='infomax',
        fit_params=dict(extended=True), random_state=42, verbose=False
    )
    ica.fit(raw, verbose=False)

    # MNE 内部: sources = unmixing_matrix_ @ pca_components_ * pre_whitener_.T @ X
    # 我们有 W (直接作用于原始数据), 需要反推 unmixing_matrix_:
    # W = unmixing_matrix_ @ pca_components_ * pre_whitener_.T
    # => unmixing_matrix_ = W / pre_whitener_.T @ pinv(pca_components_)
    W_pca = W @ np.linalg.pinv(ica.pca_components_[:W.shape[0]] * ica.pre_whitener_.T)
    ica.unmixing_matrix_ = W_pca

    exclude = _iclabel_exclude(ica, raw)
    ica.exclude = exclude
    raw_out = raw.copy()
    ica.apply(raw_out, verbose=False)
    return raw_out.get_data()


def compute_metrics(X_in, X_out, X_gt):
    T = min(X_in.shape[1], X_out.shape[1], X_gt.shape[1])
    X_in, X_out, X_gt = X_in[:, :T], X_out[:, :T], X_gt[:, :T]

    def snr(s, n):
        return 10 * np.log10(np.mean(s**2) / (np.mean(n**2) + 1e-12))

    snr_b = snr(X_gt, X_in - X_gt)
    snr_a = snr(X_gt, X_out - X_gt)
    corr  = np.mean([abs(np.corrcoef(X_out[i], X_gt[i])[0, 1]) for i in range(X_out.shape[0])])
    return snr_b, snr_a, corr


# ══════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════
METHODS = ['fastica', 'infomax', 'picard', 'jade', 'sobi']

if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'multichannel_ica')
    out_dir  = os.path.join(data_dir, 'ica_compare')
    os.makedirs(out_dir, exist_ok=True)

    X_noisy, X_clean, S_sources = load_data(data_dir)
    n_ep = X_noisy.shape[0]
    print(f'已加载  X_noisy:{X_noisy.shape}  X_clean:{X_clean.shape}')

    results = {m: {'snr_b': [], 'snr_a': [], 'corr': []} for m in METHODS}

    for ep in range(n_ep):
        X_n  = X_noisy[ep]
        X_gt = X_clean[ep]
        S_ep = S_sources[ep]
        raw  = make_raw(X_n)

        for method in METHODS:
            try:
                if method in ('fastica', 'infomax', 'picard'):
                    ica = run_mne_ica(raw, method)
                    X_out = identify_and_remove_mne(ica, raw)
                elif method == 'jade':
                    n_comp = X_n.shape[0]
                    W, _ = jade_ica(X_n, n_comp=n_comp)
                    X_out = identify_and_remove_custom(W, X_n, raw)
                elif method == 'sobi':
                    n_comp = X_n.shape[0]
                    W, _ = sobi_ica(X_n, n_comp=n_comp)
                    X_out = identify_and_remove_custom(W, X_n, raw)

                snr_b, snr_a, corr = compute_metrics(X_n, X_out, X_gt)
                results[method]['snr_b'].append(snr_b)
                results[method]['snr_a'].append(snr_a)
                results[method]['corr'].append(corr)

                # 保存对比图（仅 EPOCH_IDX）
                if ep == EPOCH_IDX:
                    t = np.arange(X_n.shape[1]) / SFREQ
                    ch = 0
                    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
                    axes[0].plot(t, X_n[ch],   lw=0.8, color='tomato',    alpha=0.7, label='含噪信号')
                    axes[0].plot(t, X_out[ch],  lw=0.8, color='steelblue', alpha=0.9, label=f'{method}')
                    axes[0].plot(t, X_gt[ch],   lw=0.8, color='green',     alpha=0.9, label='真实信号')
                    axes[0].legend(fontsize=8); axes[0].set_ylabel('幅值')
                    axes[1].plot(t, X_n[ch] - X_gt[ch],  lw=0.7, color='tomato',    alpha=0.8, label='含噪−真实')
                    axes[1].plot(t, X_out[ch] - X_gt[ch], lw=0.7, color='steelblue', alpha=0.8, label=f'{method}−真实')
                    axes[1].axhline(0, color='k', lw=0.5, ls='--')
                    axes[1].legend(fontsize=8); axes[1].set_ylabel('残差'); axes[1].set_xlabel('时间 (s)')
                    fig.suptitle(f'{method.upper()} — 第 {ep} 段 — 信噪比提升 {snr_a-snr_b:+.2f} dB')
                    plt.tight_layout()
                    fig.savefig(os.path.join(out_dir, f'epoch{ep:03d}_{method}.png'), dpi=150)
                    plt.close(fig)

            except Exception as e:
                print(f'  [{method}] 第 {ep} 段失败: {e}')
                results[method]['snr_b'].append(np.nan)
                results[method]['snr_a'].append(np.nan)
                results[method]['corr'].append(np.nan)

        if (ep + 1) % 20 == 0:
            print(f'  已处理 {ep+1}/{n_ep} 段')

    # ── 汇总表 ────────────────────────────────────────────
    print('\n' + '═' * 62)
    print(f'{"方法":<12} {"去噪前SNR":>10} {"去噪后SNR":>10} {"SNR提升":>10} {"相关系数":>8}')
    print('─' * 62)
    for m in METHODS:
        sb = np.nanmean(results[m]['snr_b'])
        sa = np.nanmean(results[m]['snr_a'])
        co = np.nanmean(results[m]['corr'])
        print(f'{m:<12} {sb:>10.2f} {sa:>10.2f} {sa-sb:>+10.2f} {co:>8.4f}')
    print('═' * 62)

    # ── 汇总对比图（SNR gain 柱状图）────────────────────
    gains = [np.nanmean(results[m]['snr_a']) - np.nanmean(results[m]['snr_b']) for m in METHODS]
    corrs = [np.nanmean(results[m]['corr']) for m in METHODS]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    axes[0].bar(METHODS, gains, color=colors)
    axes[0].axhline(0, color='k', lw=0.8, ls='--')
    axes[0].set_ylabel('信噪比提升 (dB)')
    axes[0].set_title('信噪比改善')
    axes[1].bar(METHODS, corrs, color=colors)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel('平均相关系数')
    axes[1].set_title('输出与真实信号相关性')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'summary_comparison.png'), dpi=150)
    plt.close(fig)
    print(f'\n结果已保存至 {out_dir}')
