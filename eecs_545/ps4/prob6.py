################################################################################
### EECS 545, problem set 4 question 6
### LDA and QDA analysis
################################################################################

################################################################################
### 1: Load packages, set directories and files, set graph options
################################################################################

# Import necessary packages
import matplotlib.pyplot as plt
import numpy as np
import os  # Only needed to set main directory
from eecs545_ps4funcs import discriminant_bound
from inspect import getsourcefile  # Only needed to set main directory
from sklearn.datasets import load_iris

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
fn_lda = 'LDA_boundary.pdf'  # LDA classifier boundary
fn_qda = 'QDA_boundary.pdf'  # QDA classifier boundary

# Set graph options
plt.rc('font', **{'family': 'Latin Modern Roman', 'serif': ['lmodern']})
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{lmodern}')

################################################################################
### 2: Plot LDA boundary
################################################################################

# Print a message to indicate that the program has started
print('\nStarted')

# Create the figures directory if it doesn't exist
if not os.path.isdir(mdir+'/'+fdir):
    os.mkdir(mdir+'/'+fdir)

# Change to the figures directory
os.chdir(mdir+'/'+fdir)

# Load IRIS data
iris = load_iris()

# Get features for data classified as 0 and data classified as 1
data_0, data_1 = iris.data[:,1:3][:50], iris.data[:,1:3][50:100]

# Enforce the course convention of having X be d by n
data_0, data_1 = data_0.T, data_1.T

# Get the LDA classifier boundary (since this is a line, I can set nbound=2)
plotx, ploty = discriminant_bound(X0=data_0, X1=data_1, nbound=2)

# Set up a figure
fig, ax = plt.subplots(figsize=(6.5, 3.5))

# Plot the data
ax.scatter(data_0[0,:], data_0[1,:], label=r'$y = 0$')
ax.scatter(data_1[0,:], data_1[1,:], label = r'$y = 1$')

# Plot the classifier boundary
ax.plot(plotx, ploty, color='black', linestyle='-.')

# Add axis labels
ax.set_xlabel('$x_1$', fontsize=11)
ax.set_ylabel('$x_2$', fontsize=11)

# Add a legend
ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.15), fancybox=False,
          edgecolor='black', framealpha=1, ncol=5)

# Get rid of unnecessary whitespace
fig.tight_layout()

# Save and close the figure (the bbox_inches='tight' helps remove even more
# unwanted whitespace)
plt.savefig(fn_lda, bbox_inches='tight')
plt.close()

################################################################################
### 2: Plot QDA boundary
################################################################################

# Get grid of points to plot over and classifier values for each point
X1, X2, Y = discriminant_bound(X0=data_0, X1=data_1, common_cov=False)

# Set up a figure
fig, ax = plt.subplots(figsize=(6.5, 3.5))

# Plot the data
ax.scatter(data_0[0,:], data_0[1,:], label=r'$y = 0$')
ax.scatter(data_1[0,:], data_1[1,:], label = r'$y = 1$')

# Plot the classifier boundary, which are points which achieve a classifier
# value of 0
ax.contour(X1, X2, Y, levels=[0], colors='black', linestyles='-.')

# Add axis labels
ax.set_xlabel(r'$x_1$', fontsize=11)
ax.set_ylabel(r'$x_2$', fontsize=11)

# Add a legend
ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.15), fancybox=False,
          edgecolor='black', framealpha=1, ncol=5)

# Get rid of unnecessary whitespace
fig.tight_layout()

# Save and close the figure (the bbox_inches='tight' helps remove even more
# unwanted whitespace)
plt.savefig(fn_qda, bbox_inches='tight')
plt.close()

# Print a message to indicate that the program has finished
print('\nDone')
