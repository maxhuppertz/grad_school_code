import matplotlib.pyplot as plt
import numpy as np

# Set graph options
#plt.rc('font', **{'family': 'serif', 'serif': ['lmodern']})
#plt.rc('text', usetex=True)

# Create vector of interest rates
r_min = 0
r_max = .1
r_num = 101
r = np.array(np.linspace(r_min, r_max, r_num), ndmin=2)

# Create vector of ideal years of schooling
s_min = 0
s_max = 15
s_num = 101
s = np.array(np.linspace(s_min, s_max, s_num), ndmin=2)

# Calculate implied minimum Mincer's betas
beta_min = (r.transpose() @ np.ones(r.shape)) * np.exp(r.transpose() @ s) * (1 + r.transpose() @ s)**(-1)

# Set up plot
fig, ax = plt.subplots()

# Make the heat map
heatmap = ax.imshow(beta_min, cmap='Reds', interpolation='nearest')

# Make a color bar
cbar = ax.figure.colorbar(heatmap, ax=ax)

# Select how many ticks to use
n_ticks=11

# Set ticks
ax.set_xticks(np.linspace(0, beta_min.shape[0]-1, n_ticks))
ax.set_yticks(np.linspace(0, beta_min.shape[1]-1, n_ticks))

# Set tick labels
ax.set_xticklabels(np.linspace(s_min, s_max, n_ticks))
ax.set_yticklabels(np.linspace(r_min, r_max, n_ticks))

# Label color bar
cbar.ax.set_ylabel(r'Implied minimum $\beta$', rotation=-90, va="bottom")

# Set axis labels
ax.set_xlabel(r'$s$')
ax.set_ylabel(r'$r$', rotation=0)

fig.tight_layout()

plt.show()
