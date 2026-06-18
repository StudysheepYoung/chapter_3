# %% 导入与配置
import matplotlib
matplotlib.use('Agg')

import os
import glob
import numpy as np
import mne
from mne import find_events, Epochs
import matplotlib.pyplot as plt

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(SCRIPT_DIR, "batch_preprocessing_results")

EVENT_ID     = 1000000000
TMIN, TMAX   = -0.1, 0.3
BASELINE     = (-0.1, 0)
REJECT       = dict(mag=10e-12)


# %% 辅助函数
def build_epochs_and_evoked(raw, tmin, tmax, baseline, reject):
    events = find_events(raw, stim_channel='Trigger', verbose=False)
    epochs = Epochs(raw, events, EVENT_ID, tmin=tmin, tmax=tmax,
                    baseline=baseline, detrend=1, reject=reject,
                    preload=True, verbose=False)
    return epochs, epochs.average()


def compute_snr_from_evoked(evoked, sig_tmin=0.0, sig_tmax=0.2,
                             noise_tmin=-0.1, noise_tmax=0.0):
    picks = mne.pick_types(evoked.info, meg=True, exclude='bads')
    data  = evoked.data[picks]
    times = evoked.times

    def rms(mask):
        return np.mean(np.sqrt(np.mean(data[:, mask] ** 2, axis=1)))

    rms_signal = rms((times >= sig_tmin) & (times <= sig_tmax))
    rms_noise  = rms((times >= noise_tmin) & (times < noise_tmax))
    snr_linear = rms_signal / rms_noise if rms_noise > 0 else np.nan
    snr_db     = 20 * np.log10(snr_linear) if snr_linear > 0 else np.nan
    return snr_linear, snr_db, rms_noise


# %% 读取第一个 FIF 文件
fif_files = sorted(glob.glob(os.path.join(DATA_DIR, "*", "*.fif")))
print(f"找到 {len(fif_files)} 个 .fif 文件，使用第一个: {fif_files[0]}")

raw = mne.io.read_raw_fif(fif_files[0], preload=True, verbose=False)

raw_before = raw.copy()
_, evoked_before = build_epochs_and_evoked(raw_before, TMIN, TMAX, BASELINE, REJECT)
snr_lin_before, snr_db_before, noise_rms_before = compute_snr_from_evoked(evoked_before)
print(f"SNR before: {snr_lin_before:.4f} ({snr_db_before:.2f} dB)\n")

fig = evoked_before.plot(spatial_colors=True, show=False, time_unit='s')
fig.savefig(os.path.join(SCRIPT_DIR, 'evoked_before.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
print("已保存 evoked_before.png")
