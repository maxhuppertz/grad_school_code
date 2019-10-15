################################################################################
### EECS 545, problem set 3 question 5
### Implement coordinate descent for LASSO
################################################################################

################################################################################
### 1: Load packages, set seed, set directories and files, set graph options
################################################################################

# Import necessary packages
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os  # Only needed to set main directory
import pandas as pd
from eecs545_ps3funcs import coorddesc, cdupdate_lasso, ols_mse, lasso_mse
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
fn_wplot = 'lasso_cd_weights.pdf'  # Weights across iterations
fn_mseplot = 'lasso_cd_mse.pdf'  # MSE across iterations
fn_cvplot = 'lasso_cv_mse.pdf'  # Penalty term vs. CV MSE

# Set graph options
plt.rc('font', **{'family': 'Latin Modern Roman', 'serif': ['lmodern']})
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{lmodern}')

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

# Set penalty weight
l = 100

# Set initial weights vector
w0 = np.ones(shape=(X.shape[0]+1, 1))

# Get coordinate descent weights w_hat, history of the weights w_tr, and
# objective function at each iteration L_tr
w_hat_cd, w_tr, L_tr = coorddesc(y_tr, X_tr, w0=w0, K=700, args_u=[l],
                                 update_i=cdupdate_lasso)

# Get the MSE in the test data for the estimated CD weights
mse_cd = ols_mse(y_te, X_te, w_hat_cd, demean=True, sdscale=True, addicept=True)

# Add an intercept label to the feature list
Xvars = ['Intercept'] + Xvars

# Print a header for the results
print('\nWeights vectors')
print('\n{:<10} {:>8}'.format('Variable', 'CD'))

# Go through all variables
for i, var in enumerate(Xvars):
    # Print the variable name, and formatted meand and standard deviation
    print('{:<10}'.format(var),
          '{:8.4f}'.format(np.round(w_hat_cd[i,0], 4))
    )

# Print the MSEs
print('\nCD MSE in the training data: {:8.4f}'.format(np.round(L_tr[-1], 4)))
print('CD MSE in the test data: {:8.4f}'.format(np.round(mse_cd, 4)))

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

# Add a horizontal line at zero
ax[0].axhline(0, color='black', linewidth=.8, linestyle='--')

# Go through all weight dimensions
for i in range(w_tr.shape[0]):
    # Plot weights for dimension i across iterations
    ax[0].plot(w_tr[i,:], label=Xvars[i], color=cmap(i/w_tr.shape[0]), alpha=.8)

# Label the y axis
ax[0].set_ylabel('Weight', fontsize=11)

# Add a legend
fig.legend(loc='upper center', bbox_to_anchor=(.5, 1), fancybox=False,
           edgecolor='black', framealpha=1, ncol=5)

# Add a horizontal line at zero
ax[1].axhline(0, color='black', linewidth=.8, linestyle='--')

# Go through all weight dimensions except the intercept
for i in range(1, w_tr.shape[0]):
    # Plot weights for dimension i across iterations
    ax[1].plot(w_tr[i,:], label=Xvars[i], color=cmap(i/w_tr.shape[0]), alpha=.8)

# Label the axes
ax[1].set_xlabel('Iteration', fontsize=11)
ax[1].set_ylabel('Weight', fontsize=11)

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

################################################################################
### 5: Plot cross validated MSE
################################################################################

# Make a list of penalty terms to use
L = np.linspace(0, 100, 1000)

# Make a vector of cross validation indices. Note that due to Python's indexing,
# np.arange(1, k) produces a range of numbers 1, 2, ..., k-1, so this vector is
# for the train and validate data only.
cvidx = np.kron(np.array(np.arange(1, k), ndmin=2).T, np.ones(shape=(m, 1)))

# Set up a list of MSEs for each cross validation sample
cv_mse = []

# Go through all penalty weights
for l in L:
    # Set up a vector for cross validated MSEs across samples
    cv_mse_l = np.empty(shape=(k-1, 1))

    # Go through all validation samples in the training data
    for idx in np.arange(1, k):
        # Make an indicator for the current validation sample
        vldsamp = cvidx[:,0] == idx

        # Estimate weights based on all training data except the current
        # validation sample, and get the MSE in the validation sample
        _, _, _, _, cv_mse_l[idx-1, 0] = (
            coorddesc(y_tr=y_tr[~vldsamp, :], X_tr=X_tr[:, ~vldsamp], w0=w0,
                      y_te=y_tr[vldsamp, :], X_te=X_tr[:, vldsamp], args_u=[l],
                      K=700, update_i=cdupdate_lasso)
        )

    # Calculate mean MSE across all validation samples
    mean_cv_mse = np.ones(shape=(1, k-1)) @ cv_mse_l / (k-1)

    # Add that mean MSE to the list
    cv_mse.append(mean_cv_mse[0,0])

# Find lambda which minimzes the cross validated MSE
idx_lmin = np.argmin(cv_mse)
lstar = L[idx_lmin]

# Get associated MSE in the test data
_, _, _, _, msestar = (
    coorddesc(y_tr=y_tr, X_tr=X_tr, w0=w0, y_te=y_te, X_te=X_te, args_u=[lstar],
              K=700, update_i=cdupdate_lasso)
    )

# Display the results
print('\nOptimal cross-validated lambda: {:2.2f}'.format(lstar, 4))
print('Associated MSE in the test data: {:2.4f}'.format(np.round(msestar, 4)))

# Set up a figure
fig, ax = plt.subplots(figsize=(6.5, 3.5))

# Plot the effective degrees of freedom, and save the result so it can be added
# to the legend entry later on
ax.plot(L, cv_mse, color='blue', alpha=.8)

# Add a dashed line at the optimal lambda
ax.axvline(lstar, dashes=[6, 2], color='grey')

# Label the axes
ax.set_xlabel(r'Penalty term $\lambda$', fontsize=11)
ax.set_ylabel(r'Average MSE', fontsize=11)

# Get rid of unnecessary whitespace
fig.tight_layout()

# Save and close the figure (the bbox_inches='tight' helps remove even more
# unwanted whitespace)
plt.savefig(fn_cvplot, bbox_inches='tight')
plt.close()
