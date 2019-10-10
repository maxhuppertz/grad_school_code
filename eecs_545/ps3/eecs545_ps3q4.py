################################################################################
### EECS 545, problem set 3 question 4
### Implement coordinate descent for ridge regression
################################################################################

################################################################################
### 1: Load packages, set directories and files, set graph options
################################################################################

# Import necessary packages
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os  # Only needed to set main directory
import pandas as pd
from eecs545_ps3funcs import coorddesc, regls, ols_mse
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

# Set body fat data set (has to exist in mdir)
fn_dset = 'boston-corrected.csv'

# Set figures directory (doesn't have to exist)
fdir = 'figures'

# Set file names for plots
fn_wplot = 'cd_weights.pdf'  # Weights across iterations
fn_mseplot = 'cd_mse.pdf'  # MSE across iterations

# Set graph options
plt.rc('font', **{'family': 'serif', 'serif': ['lmodern']})
plt.rc('text', usetex=True)

################################################################################
### 2: Load data
################################################################################

# Load the data set
data = pd.read_csv(fn_dset)

# Get a list of features
Xvars = [x for x in data.columns if x != 'CMEDV']

# Get responses and features
y = np.array(data['CMEDV'], ndmin=2).T
X = data[Xvars].values

# Following the course convention, make sure X is d by n
X = X.T

# Get number of features d and number of observations n
d, n = X.shape

# Specify number of folds to use for cross validation and holdout sample (the
# last fold will be used as a holdout sample)
k = 11

# Get the number of observations per fold m
m = np.int(n / k)

# Set number of training data instances (uses the first n_tr instances)
n_tr = m * (k - 1)

# Split data into training and test sample
X_tr = X[:, 0:n_tr]
y_tr = y[0:n_tr, :]
X_te = X[:, n_tr:]
y_te = y[n_tr:, :]

################################################################################
### 3: Get CD estimates
################################################################################

# Get coordinate descent weights w_hat, history of the weights w_tr, and
# objective function at each iteration L_tr
w_hat_cd, w_tr, L_tr = coorddesc(y_tr, X_tr)

# Get the MSE in the test data for the estimated SGD weights
mse_cd = ols_mse(y_te, X_te, w_hat_cd, demean=True, sdscale=True, addicept=True)

# Get the closed form solution
w_hat_cf, _, mse_cf = regls(y_tr, X_tr, l=100, y_te=y_te, X_te=X_te)

# Get the MSE in the training data for the closed form solution
L_cf = ols_mse(y_tr, X_tr, w_hat_cf, demean=True, sdscale=True, addicept=True)

# Add an intercept label to the feature list
Xvars = ['Intercept'] + Xvars

# Print a header for the results
print('\nWeights vectors')
print('\n{:<10} {:>8} {:>8}'.format('Variable', 'CD', 'CF'))

# Go through all variables
for i, var in enumerate(Xvars):
    # Print the variable name, and formatted meand and standard deviation
    print('{:<10}'.format(var),
          '{:8.4f}'.format(np.round(w_hat_cd[i,0], 4)),
          '{:8.4f}'.format(np.round(w_hat_cf[i,0], 4))
    )

# Print the MSEs
print('\nGD MSE in the training data: {:8.4f}'.format(np.round(L_tr[-1], 4)))
print('GD MSE in the test data: {:8.4f}'.format(np.round(mse_cd, 4)))
print('\nCF MSE in the training data: {:8.4f}'.format(np.round(L_cf, 4)))
print('CF MSE in the test data: {:8.4f}'.format(np.round(mse_cf, 4)))

################################################################################
### 4: Plot MSE and weights across iterations
################################################################################

# Create the figures directory if it doesn't exist
if not os.path.isdir(mdir+'/'+fdir):
    os.mkdir(mdir+'/'+fdir)

# Change to the figures directory
os.chdir(mdir+'/'+fdir)

# Get a color map
cmap = matplotlib.cm.get_cmap('jet')

# Set up a figure for the weights across iterations plot
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6.5, 7))

# Go through all weight dimensions
for i in range(w_tr.shape[0]):
    # Plot weights for dimension i across iterations
    ax[0].plot(w_tr[i,:], label=Xvars[i], color=cmap(i/w_tr.shape[0]), alpha=.8)

# Label the y axis
ax[0].set_ylabel('Weight', fontsize=11)

# Add a legend
fig.legend(loc='upper center', bbox_to_anchor=(.5, 1), fancybox=False,
           edgecolor='black', framealpha=1, ncol=5)

# Go through all weight dimensions
for i in range(w_tr.shape[0]):
    # Plot weights for dimension i across iterations
    ax[1].plot(w_tr[i,:], label=Xvars[i], color=cmap(i/w_tr.shape[0]), alpha=.8)

ax[1].set_ylim((-4, 4))

# Label the axes
ax[1].set_xlabel('Iteration', fontsize=11)
ax[1].set_ylabel('Weight', fontsize=11)

# Get rid of unnecessary whitespace
#fig.tight_layout()

# Save and close the figure (the bbox_inches='tight' helps remove even more
# unwanted whitespace)
plt.savefig(fn_wplot)#, bbox_inches='tight')
plt.close()

# Set up a figure for the MSE across iterations plot
fig, ax = plt.subplots(figsize=(6.5, 3.5))

# Plot the MSE across iterations
ax.plot(L_tr, color='blue', alpha=.8)

# Label the axes
ax.set_xlabel('Iteration', fontsize=11)
ax.set_ylabel('MSE', fontsize=11)

# Save and close the figure (the bbox_inches='tight' helps remove even more
# unwanted whitespace)
plt.savefig(fn_mseplot, bbox_inches='tight')
plt.close()
