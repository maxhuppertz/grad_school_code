################################################################################
### EECS 545, problem set 4 question 5
### Gaussian process in 1D
################################################################################

################################################################################
### 1: Load packages, set directories and files, set graph options
################################################################################

# Import necessary packages
import matplotlib.pyplot as plt
import numpy as np
import os  # Only needed to set main directory
from eecs545_ps4funcs import conditional_distribution, kernel_cov
from inspect import getsourcefile  # Only needed to set main directory

# Specify name for main directory. (This just uses the file's directory.) I used
# to use os.path.abspath(__file__), but apparently, it may be a better idea to
# use getsourcefile() instead of __file__ to make sure this runs on different
# OSs. The getsourcefile(object) function checks which file defined the object
# it is applied to. But since the object I give it is an inline function lambda,
# which was created in this file, it points to this file. The .replace() just
# ensures compatibility with Windows.
mdir = (
    os.path.dirname(os.path.abspath(getsourcefile(lambda:0))).replace('\\', '/')
    )

# Make sure I'm in the main directory
os.chdir(mdir)

# Set figures directory (doesn't have to exist)
fdir = 'figures'

# Set file names for plots
fn_prior = 'gaussian_processes_prior.pdf'  # Prior process samples
fn_posterior = 'gaussian_process_posterior.pdf'  # Posterior process samples

# Set graph options
plt.rc('font', **{'family': 'Latin Modern Roman', 'serif': ['lmodern']})
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{lmodern}')

################################################################################
### 2: Plot prior process samples
################################################################################

# Print a message to indicate that the program has started
print('\nStarted')

# Create the figures directory if it doesn't exist
if not os.path.isdir(mdir+'/'+fdir):
    os.mkdir(mdir+'/'+fdir)

# Change to the figures directory
os.chdir(mdir+'/'+fdir)

# Set random number generator's seed
np.random.seed(0)

# Set number of points to draw
n = 100

# Generate x, as a proper row vector (to go along with the course convention of
# the feature matrix being d by n)
X = np.array(np.linspace(start=-5, stop=5, num=n), ndmin=2)

# Get the mean vector of X
mu = np.zeros(shape=(n,1))

# Set up a list of different kernel parameters to use
sigma2s = [.3, .5, 1]

# Specify how many samples to plot per parameter
nseries = 3

# Set up a figure
fig, ax = plt.subplots(nrows=len(sigma2s), ncols=1, figsize=(6.5,6.5))

# Go through all kernel parameters
for i, sigma2 in enumerate(sigma2s):
    # Get the covariance matrix
    Sigma = kernel_cov(X, args=[sigma2])

    # Go through all samples to plot
    for series in range(nseries):
        # Draw a sample
        y = np.random.multivariate_normal(mean=mu[:,0], cov=Sigma)

        # Plot it
        ax[i].plot(X[0,:], y, alpha=.8)

    # Set title and vertical axis label
    ax[i].set_title(r'$\sigma^2 = {:8.1f}$'.format(sigma2), fontsize=11)
    ax[i].set_ylabel('$y$', fontsize=11)

    # Change vertical axis limits
    ax[i].set_ylim(-4, 4)

    # Check whether this is the last plot
    if i == len(sigma2s)-1:
        # If so, add a horizontal axis label
        ax[i].set_xlabel('$x$', fontsize=11)

# Get rid of unnnecessary whitespace
fig.tight_layout()

# Save and close the figure (the bbox_inches='tight' helps remove even more
# unwanted whitespace)
plt.savefig(fn_prior, bbox_inches='tight')
plt.close()

################################################################################
### 2: Plot posterior mean and process samples
################################################################################

# Set random number generator's seed
np.random.seed(0)

# Specify training data labels Sy and training data features Sx
Sy = np.array([-1.3, 2.4, -2.5, -3.3, .3], ndmin=2).T
Sx = np.array([2, 5.2, -1.5, -.8, .3], ndmin=2)

# Combine the existing X vector with new features
X_comb = np.concatenate([X, Sx], axis=1)

# Combine the mean vector for both
mu_comb = np.zeros(shape=(n+Sx.shape[1],1))

# Get indices of elements which are not conditioned on (everything but the
# training data
I = range(X.shape[1])

# Specify how many samples to plot per kernel parameter
nseries = 5

# Set up a figure
fig, ax = plt.subplots(nrows=len(sigma2s), ncols=1, figsize=(6.5,6.5))

# Go through all kernel parameters
for i, sigma2 in enumerate(sigma2s):
    # Get the covariance matrix
    Sigma_comb = kernel_cov(X_comb, args=[sigma2])

    # Get the conditional mean and covariance matrix
    mu_cond, Sigma_cond = (
        conditional_distribution(I=I, U=Sy, mu=mu_comb, Sigma=Sigma_comb)
    )

    # Go through all samples to plot
    for series in range(nseries):
        # Draw the sample
        y = np.random.multivariate_normal(mean=mu_cond[:,0], cov=Sigma_cond)

        # Plot it (the zorder makes it so that I can select what gets plotted on
        # top of what when combining different elements in a single graph)
        ax[i].plot(X[0,:], y, zorder=0, alpha=.8)

    # Add training data to the plot
    ax[i].scatter(Sx[0,:], Sy[:,0], marker='^', linewidth=3, color='black',
                  zorder=1)

    # Add thick gray line for the posterior mean
    ax[i].plot(X[0,:], mu_cond, linewidth=7, zorder=-1, color='grey', alpha=.5)

    # Set title and axis label
    ax[i].set_title(r'$\sigma^2 = {:8.1f}$'.format(sigma2), fontsize=11)
    ax[i].set_ylabel('$y$', fontsize=11)

    # Change axis limits
    ax[i].set_ylim(-4, 4)

    # If this is the last figure...
    if i == len(sigma2s)-1:
        # ... add a horizontal axis label
        ax[i].set_xlabel('$x$', fontsize=11)

# Get rid of unnecessary whitespace
fig.tight_layout()

# Save and close the figure (the bbox_inches='tight' helps remove even more
# unwanted whitespace)
plt.savefig(fn_posterior, bbox_inches='tight')
plt.close()

# Print a message to indicate that the program has finished
print('\nDone')
