import numpy as np
import os

# ── 配置 ──────────────────────────────────────────────────
N_EEG_SRC  = 4     # 每个epoch中EEG独立源的数量
N_CHANNELS = 16    # 模拟通道数（过完备混合，>= N_EEG_SRC + 2）
N_EPOCHS   = 10   # 生成的epoch数量
T          = 512   # 每段时间点数
EOG_SNR_DB  = -1  # EOG 相对 EEG 源的 SNR (dB)
EMG_SNR_DB  = -1  # EMG 相对 EEG 源的 SNR (dB)
NOISE_SNR_DB = -1   # 高斯白噪声相对混合后电极信号的 SNR (dB)
SEED       = 42


def make_mixing_matrix(n_ch, n_src, rng):
    """
    生成随机混合矩阵并列归一化，模拟头皮传导。
    列归一化保证每个源的传导强度可比。
    """
    A = np.random.randn(n_ch, n_src)  # 生成标准正态分布的随机数
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    return A


def generate_multichannel(EEG_all, EOG_all, EMG_all,
                          n_eeg_src=N_EEG_SRC,
                          n_channels=N_CHANNELS,
                          n_epochs=N_EPOCHS,
                          eog_snr_db=EOG_SNR_DB,
                          emg_snr_db=EMG_SNR_DB,
                          noise_snr_db=NOISE_SNR_DB,
                          seed=SEED):
    """
    生成多通道仿真EEG数据集，用于ICA算法验证。

    混合模型：
        S  (n_src × T)   = [eeg_1..eeg_K, eog, emg]
        X  (n_ch × T)    = A @ S   ← 模拟头皮记录
        X_clean (n_ch×T) = A[:, :K] @ S[:K]  ← ground truth（无伪影）

    返回：
        X_all       : (n_epochs, n_ch, T)   带噪多通道信号
        X_clean_all : (n_epochs, n_ch, T)   对应干净多通道信号
        A_all       : (n_epochs, n_ch, n_src) 每个epoch的混合矩阵
        S_all       : (n_epochs, n_src, T)  每个epoch的源信号
    """
    assert n_channels >= n_eeg_src + 2, \
        "n_channels 必须 >= n_eeg_src + 2 (EOG + EMG)"

    rng = np.random.default_rng(seed)
    n_src = n_eeg_src + 2  # EEG源 + EOG + EMG

    X_all, X_clean_all, A_all, S_all = [], [], [], []

    for ep in range(n_epochs):
        # ── 随机采样源信号 ────────────────────────────────
        idx_eeg = rng.choice(EEG_all.shape[0], size=n_eeg_src, replace=False)
        idx_eog = rng.integers(0, EOG_all.shape[0])
        idx_emg = rng.integers(0, EMG_all.shape[0])

        eeg_srcs = EEG_all[idx_eeg]          # (n_eeg_src, T)
        eog_src  = EOG_all[idx_eog]           # (T,)
        emg_src  = EMG_all[idx_emg]           # (T,)

        # ── SNR控制：分别缩放EOG/EMG幅度 ─────────────────
        eeg_rms = np.sqrt(np.mean(eeg_srcs ** 2))

        eog_target = eeg_rms / (10 ** (eog_snr_db / 10))
        eog_src = eog_src * (eog_target / (np.sqrt(np.mean(eog_src ** 2)) + 1e-12))

        emg_target = eeg_rms / (10 ** (emg_snr_db / 10))
        emg_src = emg_src * (emg_target / (np.sqrt(np.mean(emg_src ** 2)) + 1e-12))

        # ── 组装源矩阵 S ──────────────────────────────────
        S = np.vstack([eeg_srcs,
                       eog_src[np.newaxis, :],
                       emg_src[np.newaxis, :]])   # (n_src, T)

        # ── 随机混合矩阵 A ────────────────────────────────
        A = make_mixing_matrix(n_channels, n_src, rng)  # (n_ch, n_src)

        # ── 混合 ──────────────────────────────────────────
        X       = A @ S                          # (n_ch, T) 带噪

        # ── 叠加高斯白噪声 ────────────────────────────────
        x_rms = np.sqrt(np.mean(X ** 2))
        noise_target = x_rms / (10 ** (noise_snr_db / 10))
        white_noise = rng.standard_normal(X.shape)
        white_noise *= noise_target / (np.sqrt(np.mean(white_noise ** 2)) + 1e-12)
        X = X + white_noise

        X_clean = A[:, :n_eeg_src] @ S[:n_eeg_src]  # (n_ch, T) 仅EEG成分，无白噪声

        X_all.append(X)
        X_clean_all.append(X_clean)
        A_all.append(A)
        S_all.append(S)

    return (np.array(X_all),
            np.array(X_clean_all),
            np.array(A_all),
            np.array(S_all))


if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    out_dir  = os.path.join(data_dir, 'multichannel_ica')
    os.makedirs(out_dir, exist_ok=True)

    EEG_all = np.load(os.path.join(data_dir, 'EEG_all_epochs.npy'))
    EOG_all = np.load(os.path.join(data_dir, 'EOG_all_epochs.npy'))
    EMG_all = np.load(os.path.join(data_dir, 'EMG_all_epochs.npy'))
    print(f'Loaded  EEG:{EEG_all.shape}  EOG:{EOG_all.shape}  EMG:{EMG_all.shape}')

    X_all, X_clean_all, A_all, S_all = generate_multichannel(
        EEG_all, EOG_all, EMG_all)

    print(f'X_all       : {X_all.shape}      <- 带噪多通道输入')
    print(f'X_clean_all : {X_clean_all.shape} <- ground truth干净信号')
    print(f'A_all       : {A_all.shape}       <- 混合矩阵（ICA验证用）')
    print(f'S_all       : {S_all.shape}       <- 源信号（ICA验证用）')

    np.save(os.path.join(out_dir, 'X_noisy.npy'),  X_all)
    np.save(os.path.join(out_dir, 'X_clean.npy'),  X_clean_all)
    np.save(os.path.join(out_dir, 'A_mixing.npy'), A_all)
    np.save(os.path.join(out_dir, 'S_sources.npy'), S_all)
    print(f'Saved to {out_dir}')
