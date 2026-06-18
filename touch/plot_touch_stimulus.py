"""
电刺激范式时序示意图（方波形式）
- DS7A 电刺激仪，正中神经，200 us，ISI = 1 s，300 trials
"""

import numpy as np
import matplotlib.pyplot as plt
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stimulus_plots')
os.makedirs(OUT_DIR, exist_ok=True)

ISI    = 1.0       # 刺激间隔 s
DUR    = 0.0002    # 脉冲宽度 200 us
N_SHOW = 4

t_pts, y_pts = [], []
for i in range(N_SHOW):
    onset  = i * ISI
    offset = onset + DUR
    t_pts += [onset, onset, offset, offset]
    y_pts += [0,     1,     1,      0     ]
t_pts.append(N_SHOW * ISI)
y_pts.append(0)

t_arr = np.array(t_pts)
y_arr = np.array(y_pts)

fig, ax = plt.subplots(figsize=(9, 2.2), dpi=300)
fig.patch.set_facecolor('white')
fig.subplots_adjust(left=0.08, right=0.97, top=0.80, bottom=0.22)

ax.plot(t_arr, y_arr, color='#2166AC', linewidth=1.5)

ax.set_xlim(-0.05, N_SHOW * ISI)
ax.set_ylim(0, 1.3)
ax.set_xlabel('Time (s)', fontsize=10)
ax.set_yticks([0, 1])
ax.set_yticklabels(['0', '1'], fontsize=9)
ax.set_title('Electrical stimulus sequence  (DS7A, 200 \u03bcs, ISI = 1 s)', fontsize=10)
ax.tick_params(axis='x', labelsize=9)
for sp in ['top', 'right']:
    ax.spines[sp].set_visible(False)

path = os.path.join(OUT_DIR, 'stimulus_sequence.png')
fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'Saved -> {path}')
