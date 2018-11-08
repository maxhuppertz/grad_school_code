import matplotlib.pyplot as plt
import numpy as np

r_min = 0
r_max = .05
r_num = 100
r = np.array(np.linspace(r_min, r_max+1, r_num), ndmin=2)

s_min = 0
s_max = 15
s_num = 100
s = np.array(np.linspace(s_min, s_max, s_num), ndmin=2)

beta_min = ( (r.transpose() @ np.ones(r.shape)) @ np.exp( r.transpose() @ s ) ) / ( 1 + r.transpose() @ s )
#beta_min[np.where(beta_min > 5000)] = 2
#beta_min[np.where(beta_min <= .1)] = 1

fig, ax = plt.subplots()

heatmap = ax.imshow(beta_min, cmap='Reds', interpolation='nearest')

cbar = ax.figure.colorbar(heatmap, ax=ax)

n_ticks=10
ax.set_xticks(np.linspace(0, beta_min.shape[0], n_ticks))
ax.set_yticks(np.linspace(0, beta_min.shape[1], n_ticks))
ax.set_xticklabels(np.linspace(s_min, s_max, n_ticks))
ax.set_yticklabels(np.linspace(r_min, r_max, n_ticks))
#cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

plt.show()
