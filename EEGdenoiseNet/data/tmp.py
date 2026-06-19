import numpy as np

data = np.load('/Users/luckyyoung/Desktop/data/EEGdenoiseNet/data/EEG_all_epochs.npy')
print(data.shape)

print(data[0,:])