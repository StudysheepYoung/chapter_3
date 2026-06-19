# %% 导入与配置
import numpy as np
import mne
from mne_icalabel.iclabel import iclabel_label_components
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Times New Roman', 'Songti SC', 'STSong']
plt.rcParams['axes.unicode_minus'] = False
from scipy.linalg import eigh

SFREQ        = 512.0
EPOCH_IDX    = 0

METHODS      = ['fastica', 'infomax', 'picard', 'jade', 'sobi', 'amuse']

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'multichannel_ica')
OUT_DIR  = os.path.join(DATA_DIR, 'ica_compare')
os.makedirs(OUT_DIR, exist_ok=True)


def get_ch_names(n_ch):
    all_ch = mne.channels.make_standard_montage('standard_1020').ch_names
    idx = np.linspace(0, len(all_ch) - 1, n_ch, dtype=int)
    return [all_ch[i] for i in idx]


# %% ICA 算法实现
# ── JADE ──────────────────────────────────────────────────
def jade_numpy(X, n_comp):
    """JADE ICA，返回 (V, vecs_sub, D, S)，X: (n_chan, n_times)"""
    _, T = X.shape
    cov = X @ X.T / T
    vals, vecs = eigh(cov)
    idx = np.argsort(vals)[::-1][:n_comp]
    vals_pos = np.maximum(vals[idx], 1e-12)
    D = np.diag(1.0 / np.sqrt(vals_pos))
    vecs_sub = vecs[:, idx]
    W_white = D @ vecs_sub.T
    Z = W_white @ X

    CM = np.zeros((n_comp, n_comp * n_comp))
    for p in range(n_comp):
        for q in range(n_comp):
            zp, zq = Z[p], Z[q]
            cum = (zp * zq * Z) @ Z.T / T - np.outer(zp @ zq.T / T * np.ones(n_comp), np.ones(n_comp))
            cum -= np.eye(n_comp) * (zp @ zq.T / T)
            CM[:, p * n_comp + q] = cum[:, p]

    V = np.eye(n_comp)
    for _ in range(100):
        for p in range(n_comp - 1):
            for q in range(p + 1, n_comp):
                g = np.array([CM[p, p * n_comp + p] - CM[q, q * n_comp + q],
                               CM[p, p * n_comp + q] + CM[q, q * n_comp + p]])
                ton = g[0]; toff = g[1]
                theta = 0.5 * np.arctan2(toff, ton + np.sqrt(ton**2 + toff**2))
                c, s = np.cos(theta), np.sin(theta)
                G = np.eye(n_comp)
                G[p, p] = c; G[q, q] = c
                G[p, q] = s; G[q, p] = -s
                V = V @ G
                CM = G.T @ CM.reshape(n_comp, n_comp, n_comp).transpose(1, 0, 2).reshape(n_comp, n_comp * n_comp)
                CM = (G.T @ CM.reshape(n_comp, n_comp, n_comp)).reshape(n_comp, n_comp * n_comp)

    return V, vecs_sub, D, V.T @ Z


# ── SOBI ──────────────────────────────────────────────────
def sobi_numpy(X, n_comp, n_lags=100):
    """SOBI ICA，利用多时延协方差联合对角化，返回 (V, vecs_sub, D, S)"""
    _, T = X.shape
    cov = X @ X.T / T
    vals, vecs = eigh(cov)
    idx = np.argsort(vals)[::-1][:n_comp]
    vals_pos = np.maximum(vals[idx], 1e-12)
    D = np.diag(1.0 / np.sqrt(vals_pos))
    vecs_sub = vecs[:, idx]
    W_white = D @ vecs_sub.T
    Z = W_white @ X

    Rs = []
    for lag in np.arange(1, n_lags + 1):
        R = Z[:, lag:] @ Z[:, :-lag].T / (T - lag)
        Rs.append((R + R.T) / 2)

    V = np.eye(n_comp)
    for _ in range(200):
        for p in range(n_comp - 1):
            for q in range(p + 1, n_comp):
                num = sum(2 * R[p, q] * (R[p, p] - R[q, q]) for R in Rs)
                den = sum((R[p, p] - R[q, q])**2 - 4 * R[p, q]**2 for R in Rs)
                theta = 0.5 * np.arctan2(num, den + 1e-12)
                c, s = np.cos(theta), np.sin(theta)
                G = np.eye(n_comp)
                G[p, p] = c; G[q, q] = c
                G[p, q] = s; G[q, p] = -s
                V = V @ G
                Rs = [G.T @ R @ G for R in Rs]

    return V, vecs_sub, D, V.T @ Z


# ── AMUSE ─────────────────────────────────────────────────
def amuse_numpy(X, n_comp, lag=1):
    """AMUSE ICA，用单时延协方差特征分解，返回 (V, vecs_sub, D, S)"""
    _, T = X.shape
    cov = X @ X.T / T
    vals, vecs = eigh(cov)
    idx = np.argsort(vals)[::-1][:n_comp]
    vals_pos = np.maximum(vals[idx], 1e-12)
    D = np.diag(1.0 / np.sqrt(vals_pos))
    vecs_sub = vecs[:, idx]
    W_white = D @ vecs_sub.T
    Z = W_white @ X

    R_lag = Z[:, lag:] @ Z[:, :-lag].T / (T - lag)
    R_sym = (R_lag + R_lag.T) / 2
    _, V_mat = eigh(R_sym)
    order = np.argsort(np.abs(np.linalg.eigvalsh(R_sym)))[::-1]
    V = V_mat[:, order]

    return V, vecs_sub, D, V.T @ Z


# %% 通用工具
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
    raw.filter(1.0, 100.0, method='iir', verbose=False)
    raw.set_eeg_reference('average', verbose=False)
    return raw


def run_mne_ica(raw, method):
    n_components = min(raw.info['nchan'], 15)
    fit_params = dict(extended=True) if method == 'infomax' else {}
    if method == 'fastica':
        fit_params = dict(tol=1e-4, max_iter=2000)
    ica = mne.preprocessing.ICA(
        n_components=n_components, method=method,
        fit_params=fit_params, random_state=42,
        max_iter=2000, verbose=False
    )
    ica.fit(raw, verbose=False)
    return ica


ARTIFACT_LABELS = {'muscle artifact', 'eye blink', 'heart beat', 'line noise', 'channel noise', 'other'}
ICLABEL_NAMES   = ['brain', 'muscle artifact', 'eye blink', 'heart beat', 'line noise', 'channel noise', 'other']


def _iclabel_exclude(ica, raw):
    proba = iclabel_label_components(raw, ica)
    labels = [ICLABEL_NAMES[row.argmax()] for row in proba]
    for i, (label, prob) in enumerate(zip(labels, proba)):
        marker = ' ← exclude' if label in ARTIFACT_LABELS else ''
        print(f'  IC{i:02d} → {label:<18} ({prob.max()*100:.1f}%){marker}')
    return [i for i, label in enumerate(labels) if label in ARTIFACT_LABELS]


def identify_and_remove_mne(ica, raw):
    exclude = _iclabel_exclude(ica, raw)
    ica.exclude = exclude
    raw_out = raw.copy()
    ica.apply(raw_out, verbose=False)
    return raw_out.get_data()


def identify_and_remove_custom(V, vecs_sub, D, raw):
    """JADE/SOBI/AMUSE: 注入 MNE ICA 对象，用 ICLabel 识别，再重建"""
    n_comp = V.shape[0]
    data = raw.get_data()
    ica = mne.preprocessing.ICA(
        n_components=n_comp, method='fastica', random_state=42, verbose=False
    )
    ica.fit(raw, verbose=False)
    D_inv = np.diag(1.0 / np.diag(D))
    ica.pca_components_  = vecs_sub.T
    ica.unmixing_matrix_ = V.T @ D
    ica.mixing_matrix_   = D_inv @ V
    ica.pca_mean_        = data.mean(axis=1)
    ica.pre_whitener_    = np.ones((data.shape[0], 1))
    exclude = _iclabel_exclude(ica, raw)
    ica.exclude = exclude
    raw_out = raw.copy()
    ica.apply(raw_out, verbose=False)
    return raw_out.get_data()


def compute_metrics(X_in, X_out, X_gt):
    """
    X_in  : 含噪输入  (n_ch, T)
    X_out : 去噪输出  (n_ch, T)
    X_gt  : 真实脑信号 (n_ch, T)

    X_artifact = X_in - X_gt  为真实伪迹投影

    指标定义：
      SNR      = 20*log10( mean_ch( RMS(X_gt) / RMS(noise) ) )   per-channel RMS ratio, then mean
      RMSE     = sqrt( mean((X_out - X_gt)^2) )
      SF       = 10*log10( E[(X_in-X_out)^2] / E[X_in^2] )         屏蔽因子，越大越好
      RNR      = 10*log10( E[(X_out-X_gt)^2] / E[X_artifact^2] )  残余噪声比，越小越好
      LE       = 10*log10( E[(X_out-X_gt)^2] / E[X_gt^2] )        泄漏误差，越小越好
    """
    T = min(X_in.shape[1], X_out.shape[1], X_gt.shape[1])
    X_in, X_out, X_gt = X_in[:, :T], X_out[:, :T], X_gt[:, :T]
    X_artifact = X_in - X_gt
    residual   = X_out - X_gt

    def rms_per_ch(X):
        return np.sqrt(np.mean(X**2, axis=1))  # (n_ch,)

    def rms_ratio_per_ch(num, den):
        r = rms_per_ch(num) / (rms_per_ch(den) + 1e-12)
        return 20 * np.log10(np.mean(r))

    snr_b = rms_ratio_per_ch(X_gt, X_artifact)
    snr_a = rms_ratio_per_ch(X_gt, residual)
    rmse  = np.mean(np.sqrt(np.mean(residual**2, axis=1)))
    sf    = rms_ratio_per_ch(X_in, X_out)
    rnr   = rms_ratio_per_ch(residual, X_artifact)
    le    = rms_ratio_per_ch(residual, X_gt)
    return snr_b, snr_a, rmse, sf, rnr, le


# %% 加载数据
X_noisy, X_clean, S_sources = load_data(DATA_DIR)
n_ep = X_noisy.shape[0]
print(f'已加载  X_noisy:{X_noisy.shape}  X_clean:{X_clean.shape}')

# ── 滤波+CAR 前后对比图（epoch 0, ch 0）────────────────────────────────────
_X_n_orig = X_noisy[EPOCH_IDX].copy()
_X_gt     = X_clean[EPOCH_IDX].copy()

# get filtered-only (before CAR) to show mean drift
_n_ch = _X_n_orig.shape[0]
_info_pre = mne.create_info(ch_names=get_ch_names(_n_ch), sfreq=SFREQ, ch_types='eeg')
_raw_pre = mne.io.RawArray(_X_n_orig.copy(), _info_pre, verbose=False)
_raw_pre.set_montage(mne.channels.make_standard_montage('standard_1020'), verbose=False)
_raw_pre.filter(1.0, 100.0, method='iir', verbose=False)
_X_filtered_only = _raw_pre.get_data()          # after filter, before CAR
_mean_drift = _X_filtered_only.mean(axis=0)     # (T,) — cross-channel mean at each time point

_raw      = make_raw(X_noisy[EPOCH_IDX].copy())
_X_filt   = _raw.get_data()                     # after filter + CAR
_ch = 0
_t  = np.arange(_X_n_orig.shape[1]) / SFREQ

# mean drift plot
fig_car, axes_car = plt.subplots(2, 1, figsize=(10, 5), sharex=True, dpi=150,
                                  gridspec_kw=dict(hspace=0.08, left=0.09,
                                                   right=0.97, top=0.92, bottom=0.09))
fig_car.patch.set_facecolor('white')

axes_car[0].plot(_t, _mean_drift, lw=1.5, color='#e67e22')
axes_car[0].axhline(0, color='k', lw=0.6, ls='--')
axes_car[0].set_ylabel('各通道均值', fontsize=9)
axes_car[0].set_title('跨通道均值漂移（CAR前）— 重参考动机',
                       fontsize=10, pad=6)
for sp in ['top', 'right']: axes_car[0].spines[sp].set_visible(False)

axes_car[1].plot(_t, _X_filtered_only[_ch], lw=1.2, color='#888888', alpha=0.7, label='CAR前')
axes_car[1].plot(_t, _X_filt[_ch],          lw=1.5, color='steelblue', ls='--', marker='s', markevery=40, ms=3.5, label='CAR后')
axes_car[1].set_ylabel('通道1幅值', fontsize=9)
axes_car[1].set_xlabel('时间 (s)', fontsize=9)
axes_car[1].legend(fontsize=8, loc='upper right')
for sp in ['top', 'right']: axes_car[1].spines[sp].set_visible(False)

_car_path = os.path.join(OUT_DIR, f'epoch{EPOCH_IDX:03d}_car_mean_drift.png')
fig_car.savefig(_car_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig_car)
print(f'Saved → {_car_path}')
_ch = 0
_t  = np.arange(_X_n_orig.shape[1]) / SFREQ

fig, ax = plt.subplots(figsize=(10, 3), dpi=150)
fig.patch.set_facecolor('white')
_mask_fc = (_t >= 0.3) & (_t <= 0.5)
ax.plot(_t[_mask_fc], _X_n_orig[_ch][_mask_fc], lw=0.8, color='tomato',    label='含噪 (原始)')
ax.plot(_t[_mask_fc], _X_filt[_ch][_mask_fc],   lw=0.8, color='steelblue', ls='--', marker='s', markevery=10, ms=3.5, label='滤波+CAR后')
ax.plot(_t[_mask_fc], _X_gt[_ch][_mask_fc],     lw=0.8, color='green',              marker='o', markevery=10, ms=3.5, label='真实信号')
ax.set_ylabel('幅值', fontsize=9)
ax.set_xlabel('时间 (s)', fontsize=9)
ax.legend(fontsize=8, loc='upper right')
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
ax.set_title(f'滤波+CAR效果 — 第 {EPOCH_IDX} 段, 通道{_ch+1}', fontsize=11)
out_path = os.path.join(OUT_DIR, f'epoch{EPOCH_IDX:03d}_filter_car.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'Saved → {out_path}')

results = {m: {'snr_b': [], 'snr_a': [], 'rmse': [], 'sf': [], 'rnr': [], 'le': []} for m in METHODS}
residuals_epoch0 = {}   # collect residuals for combined plot


# %% 主循环：逐 epoch 逐方法运行 ICA
for ep in range(n_ep):
    X_n  = X_noisy[ep].copy()
    X_gt = X_clean[ep]
    raw  = make_raw(X_n.copy())
    X_filtered = raw.get_data()   # 滤波+CAR 后的 noisy，自定义算法在此基础上分解
    n_comp = min(X_n.shape[0], 15)

    for method in METHODS:
        try:
            if method in ('fastica', 'infomax', 'picard'):
                ica = run_mne_ica(raw, method)
                X_out = identify_and_remove_mne(ica, raw)
                if method == 'fastica' and ep == EPOCH_IDX:
                    sources = ica.get_sources(raw).get_data()  # (n_comp, T)
                    n_comp_plot = sources.shape[0]
                    t_ic = np.arange(sources.shape[1]) / SFREQ
                    fig_ic, axes_ic = plt.subplots(n_comp_plot, 1,
                                                   figsize=(12, n_comp_plot * 0.55),
                                                   sharex=True,
                                                   gridspec_kw=dict(hspace=0.05,
                                                                    left=0.08, right=0.98,
                                                                    top=0.95, bottom=0.04))
                    fig_ic.patch.set_facecolor('white')
                    for i, ax_ic in enumerate(axes_ic):
                        y = sources[i]; m = np.max(np.abs(y))
                        col = '#C44E52'
                        ax_ic.plot(t_ic, y / m if m > 0 else y, lw=0.6, color=col)
                        ax_ic.set_yticks([])
                        ax_ic.set_ylabel(f'IC{i:02d}', fontsize=6, rotation=0,
                                         labelpad=28, va='center', ha='right')
                        for sp in ['top', 'right', 'left']:
                            ax_ic.spines[sp].set_visible(False)
                        if i < n_comp_plot - 1:
                            ax_ic.spines['bottom'].set_visible(False)
                        else:
                            ax_ic.spines['bottom'].set_linewidth(0.5)
                            ax_ic.set_xlabel('时间 (s)', fontsize=8)
                            ax_ic.tick_params(axis='x', labelsize=7)
                    axes_ic[0].set_title(
                        f'FastICA — {n_comp_plot} 个成分  '
                        f'(红色=已排除, 第 {EPOCH_IDX} 段)', fontsize=10, pad=6)
                    fig_ic.savefig(os.path.join(OUT_DIR, f'epoch{EPOCH_IDX:03d}_fastica_components.png'),
                                   dpi=150, bbox_inches='tight', facecolor='white')
                    plt.close(fig_ic)
                    print(f'Saved → {os.path.join(OUT_DIR, f"epoch{EPOCH_IDX:03d}_fastica_components.png")}')
            elif method == 'jade':
                V, vecs_sub, D, _ = jade_numpy(X_filtered, n_comp)
                X_out = identify_and_remove_custom(V, vecs_sub, D, raw)
            elif method == 'sobi':
                V, vecs_sub, D, _ = sobi_numpy(X_filtered, n_comp)
                X_out = identify_and_remove_custom(V, vecs_sub, D, raw)
            elif method == 'amuse':
                V, vecs_sub, D, _ = amuse_numpy(X_filtered, n_comp)
                X_out = identify_and_remove_custom(V, vecs_sub, D, raw)

            snr_b, snr_a, rmse, sf, rnr, le = compute_metrics(X_n, X_out, X_gt)
            results[method]['snr_b'].append(snr_b)
            results[method]['snr_a'].append(snr_a)
            results[method]['rmse'].append(rmse)
            results[method]['sf'].append(sf)
            results[method]['rnr'].append(rnr)
            results[method]['le'].append(le)

            if ep == EPOCH_IDX:
                t = np.arange(X_n.shape[1]) / SFREQ
                ch = 0
                t_show = 0.5          # show only first 0.5 s for local detail
                mask = t <= t_show

                # amplitude plot
                fig_amp, ax_amp = plt.subplots(figsize=(8, 3), dpi=150)
                fig_amp.patch.set_facecolor('white')
                ax_amp.plot(t[mask], X_n[ch][mask],   lw=2.0, color='tomato',    alpha=0.7, label='含噪 (原始)')
                ax_amp.plot(t[mask], X_out[ch][mask], lw=2.0, color='steelblue', alpha=0.9, ls='--', marker='s', markevery=20, ms=3.5, label=f'{method}')
                ax_amp.plot(t[mask], X_gt[ch][mask],  lw=2.0, color='green',     alpha=0.9,           marker='o', markevery=20, ms=3.5, label='真实信号')
                ax_amp.legend(fontsize=8); ax_amp.set_ylabel('幅值', fontsize=9)
                ax_amp.set_xlabel('时间 (s)', fontsize=9)
                for sp in ['top', 'right']: ax_amp.spines[sp].set_visible(False)
                fig_amp.suptitle(f'{method.upper()} — 第 {ep} 段 — 信噪比提升 {snr_a-snr_b:+.2f} dB', fontsize=10)
                fig_amp.tight_layout()
                fig_amp.savefig(os.path.join(OUT_DIR, f'epoch{ep:03d}_{method}.png'), dpi=150,
                                bbox_inches='tight', facecolor='white')
                plt.close(fig_amp)

                # store residual for combined plot
                residuals_epoch0[method] = (X_out[ch] - X_gt[ch]).copy()
                if ep == EPOCH_IDX and method not in residuals_epoch0:
                    pass  # already stored above

                # PSD comparison
                from scipy.signal import welch
                _nperseg = min(256, X_n.shape[1])
                f_n,  p_n   = welch(X_n[ch],   fs=SFREQ, nperseg=_nperseg)
                f_out, p_out = welch(X_out[ch], fs=SFREQ, nperseg=_nperseg)
                f_gt, p_gt  = welch(X_gt[ch],  fs=SFREQ, nperseg=_nperseg)

                fig_psd, ax_psd = plt.subplots(figsize=(7, 4), dpi=150)
                fig_psd.patch.set_facecolor('white')
                ax_psd.semilogy(f_n,   p_n,   lw=1.0, color='tomato',    alpha=0.7, label='含噪 (原始)')
                ax_psd.semilogy(f_out, p_out, lw=1.0, color='steelblue', alpha=0.9, ls='--', marker='s', markevery=20, ms=3.5, label=f'{method}')
                ax_psd.semilogy(f_gt,  p_gt,  lw=1.0, color='green',     alpha=0.9,           marker='o', markevery=20, ms=3.5, label='真实信号')
                ax_psd.set_xlim(0, SFREQ / 2)
                ax_psd.set_xlabel('频率 (Hz)', fontsize=9)
                ax_psd.set_ylabel('功率谱密度 (V²/Hz)', fontsize=9)
                ax_psd.legend(fontsize=8)
                for sp in ['top', 'right']: ax_psd.spines[sp].set_visible(False)
                ax_psd.set_title(f'{method.upper()} 功率谱 — 第 {ep} 段, 通道{ch+1}', fontsize=10)
                fig_psd.tight_layout()
                fig_psd.savefig(os.path.join(OUT_DIR, f'epoch{ep:03d}_{method}_psd.png'),
                                dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig_psd)

        except Exception as e:
            print(f'  [{method}] 第 {ep} 段失败: {e}')
            for k in ('snr_b', 'snr_a', 'rmse', 'sf', 'rnr', 'le'):
                results[method][k].append(np.nan)

    if (ep + 1) % 20 == 0:
        print(f'  已处理 {ep+1}/{n_ep} 段')


# %% combined residual plot (all 6 methods, epoch 0)
if residuals_epoch0:
    _t_r  = np.arange(X_noisy.shape[2]) / SFREQ
    _mask = _t_r <= 0.2
    _noisy_res = (X_noisy[EPOCH_IDX][0] - X_clean[EPOCH_IDX][0])
    _method_styles = [
        dict(lw=1.0, ls='--', color='#e74c3c', marker='o',  markevery=(2,  18), ms=3.5),
        dict(lw=1.0, ls='--', color='#3498db', marker='s',  markevery=(5,  22), ms=3.5),
        dict(lw=1.0, ls='--', color='#2ecc71', marker='^',  markevery=(1,  27), ms=3.5),
        dict(lw=1.0, ls='--', color='#f39c12', marker='D',  markevery=(8,  20), ms=3.5),
        dict(lw=1.0, ls='--', color='#9b59b6', marker='v',  markevery=(3,  25), ms=3.5),
        dict(lw=1.0, ls='--', color='#1abc9c', marker='P',  markevery=(11, 30), ms=3.5),
    ]

    fig_cr, ax_cr = plt.subplots(figsize=(10, 4), dpi=150)
    fig_cr.patch.set_facecolor('white')
    _mask = _t_r <= 0.1
    ax_cr.plot(_t_r[_mask], _noisy_res[_mask], lw=1.5, color='#888888', alpha=0.6, label='含噪−真实', zorder=1)
    for (m, res), sty in zip(residuals_epoch0.items(), _method_styles):
        ax_cr.plot(_t_r[_mask], res[_mask], label=f'{m}−真实', **sty)
    ax_cr.axhline(0, color='k', lw=0.8, ls='--')
    ax_cr.legend(fontsize=8, ncol=4, loc='upper right')
    ax_cr.set_ylabel('残差', fontsize=9)
    ax_cr.set_xlabel('时间 (s)', fontsize=9)
    for sp in ['top', 'right']: ax_cr.spines[sp].set_visible(False)
    fig_cr.suptitle(f'残差对比 — 所有方法 — 第 {EPOCH_IDX} 段, 通道1', fontsize=10)
    fig_cr.tight_layout()
    _cr_path = os.path.join(OUT_DIR, f'epoch{EPOCH_IDX:03d}_residual_all.png')
    fig_cr.savefig(_cr_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig_cr)
    print(f'Saved → {_cr_path}')

# %% 汇总表
print('\n' + '═' * 90)
print(f'{"方法":<12} {"去噪前SNR":>8} {"去噪后SNR":>8} {"SNR提升":>8} {"RMSE":>10} {"SF":>8} {"RNR":>8} {"LE":>8}')
print('─' * 80)
for m in METHODS:
    sb  = np.nanmean(results[m]['snr_b'])
    sa  = np.nanmean(results[m]['snr_a'])
    rm  = np.nanmean(results[m]['rmse'])
    sf  = np.nanmean(results[m]['sf'])
    rnr = np.nanmean(results[m]['rnr'])
    le  = np.nanmean(results[m]['le'])
    print(f'{m:<12} {sb:>8.2f} {sa:>8.2f} {sa-sb:>+8.2f} {rm:>10.4f} {sf:>8.2f} {rnr:>8.2f} {le:>8.2f}')
print('═' * 80)


# %% 汇总对比图
gains = [np.nanmean(results[m]['snr_a']) - np.nanmean(results[m]['snr_b']) for m in METHODS]
sfs   = [np.nanmean(results[m]['sf'])    for m in METHODS]
rnrs  = [np.nanmean(results[m]['rnr'])   for m in METHODS]
les   = [np.nanmean(results[m]['le'])    for m in METHODS]
rmses = [np.nanmean(results[m]['rmse'])  for m in METHODS]

fig, axes = plt.subplots(1, 5, figsize=(22, 4))
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

axes[0].bar(METHODS, gains, color=colors)
axes[0].axhline(0, color='k', lw=0.8, ls='--')
axes[0].set_ylabel('信噪比提升 (dB)'); axes[0].set_title('信噪比提升')

axes[1].bar(METHODS, sfs, color=colors)
axes[1].set_ylabel('SF (dB)'); axes[1].set_title('屏蔽因子')

axes[2].bar(METHODS, rnrs, color=colors)
axes[2].set_ylabel('RNR (dB)'); axes[2].set_title('残余噪声比')

axes[3].bar(METHODS, les, color=colors)
axes[3].set_ylabel('LE (dB)'); axes[3].set_title('泄漏误差')

axes[4].bar(METHODS, rmses, color=colors)
axes[4].set_ylabel('RMSE'); axes[4].set_title('均方根误差')

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'summary_comparison.png'), dpi=150)
plt.close(fig)
print(f'\n结果已保存至 {OUT_DIR}')

# %%
