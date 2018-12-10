# Import necessary modules
import matplotlib.pyplot as plt
import numpy as np
from os import chdir, mkdir, path, mkdir
from scipy.stats import norm

# Set graph options
plt.rc('font', **{'family': 'serif', 'serif': ['lmodern']})
plt.rc('text', usetex=True)

# Specify name for main directory (just uses the file's directory)
mdir = path.dirname(path.abspath(__file__)).replace('\\', '/')

# Set figures directory (doesn't need to exist)
fdir = '/figures'

# Create the figures directory if it doesn't exist
if not path.isdir(mdir+fdir):
    mkdir(mdir+fdir)

# Change directory to figures
chdir(mdir+fdir)

# Define PDFs
def fX(x, mu=0, sigma2=1): return norm.pdf((np.log(x) - mu)/np.sqrt(sigma2)) / x
def fY(y, mu=0, sigma2=1): return fX(y, mu, sigma2) * (1 + np.sin(2 * np.pi * np.log(y)))

# Set up values for the horizontal axis
x = np.linspace(10**(-10), 1000, 10000)

# Select parameters mu and sigma^2
mu = 5
sigma2 = 1

# Set up a plot
fig, ax = plt.subplots(figsize=(6.5, 4.5))

# Plot PDFs
ax.plot(x, fX(x, mu, sigma2), label='$f_X(x)$', color='b', linestyle='-')
ax.plot(x, fY(x, mu, sigma2), label='$f_Y(x)$', color='g', linestyle='-.')

# Set axis limits
ax.set_xlim(0, np.max(x))
ax.set_ylim(0)

# Set axis labels
ax.set_xlabel('$x$', fontsize=11)

# Enable legend
ax.legend(fontsize=11)

# Get rid of unnecessary whitespace
fig.tight_layout()

# Save figure and close
plt.savefig('q3_densities.pdf')
plt.close()
