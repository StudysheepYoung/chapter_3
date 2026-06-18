# %% 导入与配置
import matplotlib
matplotlib.use('Agg')

import os
import numpy as np
import mne
import scipy.io as scio
from mne import find_events, Epochs
import matplotlib.pyplot as plt

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(SCRIPT_DIR, "20241214 104044.basedata")
SENSOR_PATH  = os.path.join(SCRIPT_DIR, "sensors_mecg64.mat")

TMIN, TMAX   = -0.1, 0.6
BASELINE     = (-0.1, 0)
REJECT       = dict(mag=3e-9)


# %% 辅助函数
def read_meg_data(data_path, sensor_path):
    fs = 1000
    n_record_chans = 66
    file_id = open(data_path, "rb")
    baseDate_data_0 = np.fromfile(file_id, dtype=np.float32)
    baseDate_data = baseDate_data_0[512:]
    General_Time_In_Seconds = len(baseDate_data) // n_record_chans // fs
    Single_Sensor_Data_Length = General_Time_In_Seconds * fs
    file_id.close()
    read_raw_data = np.zeros((n_record_chans, Single_Sensor_Data_Length))
    for channel_index in range(n_record_chans):
        for time_seconds in range(General_Time_In_Seconds):
            read_raw_data[channel_index, time_seconds * fs:(time_seconds + 1) * fs] = \
                baseDate_data[channel_index * fs + (time_seconds * n_record_chans * fs):
                              (channel_index + 1) * fs + (time_seconds * n_record_chans * fs)]

    use_chans = 65
    raw_data = read_raw_data[:use_chans, :]
    raw_data[:-1, :] = raw_data[:-1, :] * 1e-12

    num_chans_data = 64
    sensor_info = scio.loadmat(sensor_path)
    label = list(sensor_info['ch_names'])
    label = [lab.strip() for lab in label]
    pos = sensor_info['pos']
    ori = sensor_info['ori']

    raw_info = mne.create_info(
        ch_names=label + ['Trigger'],
        ch_types=['eeg' for _ in range(64)] + ['stim'],
        sfreq=1000)

    raw = mne.io.RawArray(raw_data, raw_info)
    dic = {label[i]: pos[i, :] for i in range(num_chans_data)}
    montage = mne.channels.make_dig_montage(ch_pos=dic, coord_frame='head')
    raw = raw.set_montage(montage)

    for j, ch_name in enumerate(raw.info['ch_names']):
        if ch_name != 'Trigger':
            raw.info['chs'][j]['kind'] = mne.io.constants.FIFF.FIFFV_MEG_CH
            raw.info['chs'][j]['unit'] = mne.io.constants.FIFF.FIFF_UNIT_T
            raw.info['chs'][j]['coil_type'] = mne.io.constants.FIFF.FIFFV_COIL_QUSPIN_ZFOPM_MAG2
            raw.info['chs'][j]['loc'][3:12] = np.array([1., 0., 0., 0., 1., 0., 0., 0., 1.])
            Z_orient = mne._fiff.tag._loc_to_coil_trans(raw.info['chs'][j]['loc'])[:3, :3]
            find_Rotation = mne.transforms._find_vector_rotation(Z_orient[:, 2], ori[j, :])
            raw.info['chs'][j]['loc'][3:12] = np.dot(find_Rotation, Z_orient).T.ravel()

    return raw


def build_epochs_and_evoked(raw, tmin, tmax, baseline, reject):
    events = find_events(raw, stim_channel='Trigger', verbose=False)
    events[:, 0] += 400
    event_id = int(np.unique(events[:, 2])[0])
    epochs = Epochs(raw, events, event_id, tmin=tmin, tmax=tmax,
                    baseline=baseline, detrend=1, reject=reject,
                    preload=True, verbose=False)
    return epochs, epochs.average()


def compute_snr_from_evoked(evoked, sig_tmin=0.05, sig_tmax=0.15, noise_tmin=-0.1, noise_tmax=0.0):
    picks = mne.pick_types(evoked.info, meg=True, exclude='bads')
    data = evoked.data[picks]
    times = evoked.times

    def rms(mask):
        return np.mean(np.sqrt(np.mean(data[:, mask] ** 2, axis=1)))

    rms_signal = rms((times >= sig_tmin) & (times <= sig_tmax))
    rms_noise  = rms((times >= noise_tmin) & (times < noise_tmax))
    snr_linear = rms_signal / rms_noise if rms_noise > 0 else np.nan
    snr_db     = 20 * np.log10(snr_linear) if snr_linear > 0 else np.nan
    return snr_linear, snr_db, rms_noise


# %% 读取数据，计算去噪前基线
print("读取原始数据...")
raw = read_meg_data(DATA_PATH, SENSOR_PATH)

raw_before = raw.copy()
raw_before.filter(1.0, 40.0, picks='meg', verbose=False)
_, evoked_before = build_epochs_and_evoked(raw_before, TMIN, TMAX, BASELINE, REJECT)
snr_lin_before, snr_db_before, noise_rms_before = compute_snr_from_evoked(evoked_before)
print(f"SNR before: {snr_lin_before:.4f} ({snr_db_before:.2f} dB)\n")

fig = evoked_before.plot(spatial_colors=True, show=False, time_unit='s')
fig.savefig(os.path.join(SCRIPT_DIR, 'evoked_before.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
print("已保存 evoked_before.png")
