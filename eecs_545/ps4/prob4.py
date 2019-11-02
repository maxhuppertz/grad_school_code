################################################################################
### EECS 545, problem set 4 question 4
### Multivariate normal distribution
################################################################################

################################################################################
### 1: Load packages, set directories and files, set graph options
################################################################################

# Import necessary packages
import matplotlib.pyplot as plt
import numpy as np
import os  # Only needed to set main directory
from eecs545_ps4funcs import (multivariate_gaussian, marginal_distribution,
                              conditional_distribution)
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
fn_dplot = 'marginal_density.pdf'  # Marginal density
fn_cplot = 'conditional_density.pdf'  # Condtional contour plot

# Set graph options
plt.rc('font', **{'family': 'Latin Modern Roman', 'serif': ['lmodern']})
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{lmodern}')

################################################################################
### 2: Plot marginal density
################################################################################

# Print a message to indicate that the program has started
print('\nStarted')

# Create the figures directory if it doesn't exist
if not os.path.isdir(mdir+'/'+fdir):
    os.mkdir(mdir+'/'+fdir)

# Change to the figures directory
os.chdir(mdir+'/'+fdir)

# Set up mean and covariance matrix
mu = np.zeros(shape=(2,1))
Sigma = np.array([[1.0, 0.5],
                  [0.5, 1.0]])

# Specify which indices to extract
I = [0]

# Make a list of x values to plot over
xmax = 4.5
X = np.linspace(start=-xmax, stop=xmax, num=1000)

# Get the marginal PDF for each of those values
px = [multivariate_gaussian(x, *marginal_distribution(I, mu, Sigma)) for x in X]

# Set up a figure
fig, ax = plt.subplots(figsize=(6.5, 3.5))

# Plot the density
ax.plot(X, px)

# Label the axes
ax.set_xlabel(r'$x$', fontsize=11)
ax.set_ylabel(r'$p_{X_1}(x)$', fontsize=11)

# Get rid of unnecessary whitespace
fig.tight_layout()

# Save and close the figure (the bbox_inches='tight' helps remove even more
# unwanted whitespace)
plt.savefig(fn_dplot, bbox_inches='tight')
plt.close()

################################################################################
### 3: Plot conditional density
################################################################################

# Set up mean and covariance matrix
mu = np.array([[0.5], [0.0], [-0.5], [0.0]])
Sigma = np.array([[1.0, 0.5, 0.0, 0.0],
                  [0.5, 1.0, 0.0, 1.5],
                  [0.0, 0.0, 2.0, 0.0],
                  [0.0, 1.5, 0.0, 4.0]])

# Set indices of variables which are not conditioned on
I = [0, 3]

# Set values for conditioning variables
U = np.array([[0.1], [-0.2]])

# Get the conditional mean and covariance matrix
mu_cond, Sigma_cond = conditional_distribution(I, U, mu, Sigma)

# Make a grid of points to calculate densities over
xmax = 3.5
x1 = np.linspace(start=-xmax, stop=xmax, num=1000)
x4 = np.linspace(start=-xmax, stop=xmax, num=1000)
X1, X4 = np.meshgrid(x1, x4)

# Calculate the corresponding grid of densities
px = np.array(
    [
        [multivariate_gaussian([X1[i,j], X4[i,j]], mu_cond, Sigma_cond)
         for j in np.arange(X1.shape[0])]
        for i in np.arange(X1.shape[1])
    ]
)

# Set up a figure
fig, ax = plt.subplots(figsize=(4.5, 4.5))

# Plot the density contours
cs = ax.contour(X1, X4, px, levels=15)

# Label the axes
ax.set_xlabel(r'$X_1$', fontsize=11)
ax.set_ylabel(r'$X_4$', fontsize=11)
#ax.clabel(cs)  # Add labels to contour lines (overcrowds the plot a little)

# Get rid of unnecessary whitespace
fig.tight_layout()

# Save and close the figure (the bbox_inches='tight' helps remove even more
# unwanted whitespace)
plt.savefig(fn_cplot, bbox_inches='tight')
plt.close()

# Print a message to indicate that the program has finished
print('\nDone')
