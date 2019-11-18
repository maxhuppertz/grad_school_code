################################################################################
### EECS 545, problem set 5 question 3
### Eigenfaces
################################################################################

################################################################################
### 1: Load packages, set directories and files, set graph options
################################################################################

# Import necessary packages
import matplotlib.pyplot as plt
import numpy as np
import os  # Only needed to set main directory
import scipy.io as sio
import time
from eecs545_ps5funcs import pca
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
fn_evs = 'eigenvalues.pdf'  # Sorted eigenvalues
fn_efc = 'eigenfaces.pdf'  # Eigenfaces

# Set graph options
plt.rc('font', **{'family': 'Latin Modern Roman', 'serif': ['lmodern']})
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{lmodern}')

################################################################################
### 2: Run PCA
################################################################################

# Load Yale data
yale = sio.loadmat('yalefaces.mat')

# Get face data
yalefaces = yale['yalefaces']

# Get dimensions of images
imgx = yalefaces.shape[0]
imgy = yalefaces.shape[1]

# Get the number of features d and instances n of vectorized images
d = imgx * imgy
n = yalefaces.shape[2]

# Reshape the images into vectors
X = yalefaces.reshape((d, n))

# Specify how many principal components to extract at most
Kmax = 500

# Make a list of principal components, from 1 to Kmax
K = np.arange(start=1, stop=Kmax, step=1)

# Get the fraction of variance explained by each component R, and eigenvectors U
R, L, U = pca(X, K)

# Check for which k the fraction of the total variance explained by PCA is above
# 95 percent and above 99 percent, respectively (I have to add one to deal with
# Python's zero indexing)
g95 = np.amin(np.argwhere(R > .95)) + 1
g99 = np.amin(np.argwhere(R > .99)) + 1

# Calculate the percentage reduction in dimensionality for each case
r95 = np.round((1 - (g95 / d)) * 100, 2)
r99 = np.round((1 - (g99 / d)) * 100, 2)

# Print the results
print('\nk needed to explain 95% of variance: {}'.format(g95))
print('Dimensionality reduction: {:3.2f}%'.format(r95))
print('\nk needed to explain 99% of variance: {}'.format(g99))
print('Dimensionality reduction: {:3.2f}%'.format(r99))

################################################################################
### 3: Plot sorted eigenvalues
################################################################################

# Create the figures directory if it doesn't exist
if not os.path.isdir(mdir+'/'+fdir):
    os.mkdir(mdir+'/'+fdir)

# Change to the figures directory
os.chdir(mdir+'/'+fdir)

# Set up a figure
fig, ax = plt.subplots(figsize=(6.5, 4.5))

# Plot sorted eigenvalues
ax.semilogy(L)

# Label the axes
ax.set_xlabel(r'Eigenvalue', fontsize=11)

# Get rid of unnecessary whitespace
fig.tight_layout()

# Save and close the figure (the bbox_inches='tight' helps remove even more
# unwanted whitespace)
plt.savefig(fn_evs, bbox_inches='tight')
plt.close()

################################################################################
### 4: Plot eigenfaces
################################################################################

# Specify how many eigenfaces to plot
nplot = 19

# Set a color map for the plot
colmp = 'gray'

# Reshape the first nplot eigenfaces into images
efc = U[:, 0:nplot].reshape((imgx, imgy, nplot))

# Get the sample mean as an image
mu = (X @ np.ones(shape=(n,1)) / n).reshape((imgx, imgy))

# Add that to the eigenfaces, as the first one
efc = np.insert(efc, 0, mu, axis=2)

# Set up another figure
fig, ax = plt.subplots(nrows=4, ncols=5, figsize=(6.5, 7))

# Specify a list of special suffixes for different numbers (for plot titles)
sufx = {1: 'st', 2: 'nd', 3: 'rd'}

# Go through all eigenfaces
for i, ax in enumerate(fig.axes):
    # Get the i-th eigenface
    img = efc[:,:,i]

    # Plot the image
    ax.imshow(img, cmap=colmp)

    # Remove x and y ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Check whether a special suffix has to be used
    if i in sufx.keys():
        # If so, use it
        suf = sufx[i]
    else:
        # Else, just use 'th'
        suf = 'th'

    # Add a title to the figure
    ax.set_title(str(i) + suf + ' eigenface', fontsize=11)

# Get rid of unnecessary whitespace
fig.tight_layout()

# Save and close the figure (the bbox_inches='tight' helps remove even more
# unwanted whitespace)
plt.savefig(fn_efc, bbox_inches='tight')
plt.close()
