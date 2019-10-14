################################################################################
### EECS 545, problem set 3 question 1
### Implement cross validated ridge regression
################################################################################

################################################################################
### 1: Load packages, set directories and files, set graph options
################################################################################

# Import necessary packages
import matplotlib.pyplot as plt
import numpy as np
import os  # Only needed to set main directory
import pandas as pd
from eecs545_ps3funcs import effdf, regls
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
fn_effdfplot = 'effective_df.pdf'  # Effective DF vs. MSE
fn_cvplot = 'cv_mse.pdf'  # Penalty term vs. CV MSE

# Set graph options
plt.rc('font', **{'family': 'Latin Modern Roman', 'serif': ['lmodern']})
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{lmodern}')

################################################################################
### 2: Load data, display means
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

# Calculate sample means and standard deviations
mu_X_tr = X_tr @ np.ones(shape=(n_tr, 1)) / n_tr
sigma_X_tr = np.array(np.std(X_tr, axis=1, ddof=1), ndmin=2).T

# Print a header for the results
print('\n{:<10} {:>8} {:>8}'.format('Variable', 'Mean', 'SD'))

# Go through all variables
for i, var in enumerate(Xvars):
    # Print the variable name, and formatted meand and standard deviation
    print('{:<10}'.format(var),
          '{:8.4f}'.format(np.round(mu_X_tr[i,0], 4)),
          '{:8.4f}'.format(np.round(sigma_X_tr[i,0], 4))
    )

################################################################################
### 3: Plot effective degrees of freedom and MSE
################################################################################

# Create the figures directory if it doesn't exist
if not os.path.isdir(mdir+'/'+fdir):
    os.mkdir(mdir+'/'+fdir)

# Change to the figures directory
os.chdir(mdir+'/'+fdir)

# Make a list of penalty terms to use
L = np.arange(0, 21, .1)

# Calculate MSEs
mse = [regls(y_tr, X_tr, l, y_te=y_tr, X_te=X_tr)[2] for l in L]

# Set up a figure
fig, ax1 = plt.subplots(figsize=(6.5, 3.5))

# Plot the effective degrees of freedom, and save the result so it can be added
# to the legend entry later on
leg1 = ax1.plot(L, [effdf(X_tr, l) for l in L], color='blue', alpha=.8,
                label=r'Effective DF')

# Label the axes
ax1.set_xlabel(r'Penalty term $\lambda$', fontsize=11)
ax1.set_ylabel(r'Effective DF', fontsize=11, color='blue', alpha=.8)

# Set up a second set of axes (this clones the x axis, but creates a new y axis)
ax2 = ax1.twinx()

# Plot the MSE on the second y axis, and save the result
leg2 = ax2.plot(L, mse, color='green', alpha=.8, linestyle='-.', label=r'MSE')

# Label the new y axis
ax2.set_ylabel('MSE', fontsize=11, color='green', alpha=.8)

# Add both legend entries together
legs = leg1 + leg2

# Add the lines and labels to the legend
ax1.legend(legs, [l.get_label() for l in legs], bbox_to_anchor=(.5, 1),
           loc='upper center', ncol=2, fancybox=False, edgecolor='black',
           framealpha=1)

# Get rid of unnecessary whitespace
fig.tight_layout()

# Save and close the figure (the bbox_inches='tight' helps remove even more
# unwanted whitespace)
plt.savefig(fn_effdfplot, bbox_inches='tight')
plt.close()

################################################################################
### 4: Get MSE for standard OLS
################################################################################

# Get MSE for standard OLS
_, _, mse_ols = regls(y_tr, X_tr, l=0, y_te=y_te, X_te=X_te)

# Display the result
print('\nOLS MSE in the test data:', '{:4.4f}'.format(np.round(mse_ols, 4)))

################################################################################
### 5: Plot cross validated MSE
################################################################################

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

    # Go through all cross validation folds
    for idx in np.arange(1, k):
        # Make an indicator for the current sample
        vldsamp = cvidx[:,0] == idx

        # Get the MSE for the current cross validation fold, and add it to the
        # list
        _, _, cv_mse_l[idx-1, 0] = (
            regls(y_tr=y_tr[~vldsamp, :], X_tr=X_tr[:, ~vldsamp], l=l,
                  y_te=y_tr[vldsamp, :], X_te=X_tr[:, vldsamp])
            )

    # Calculate mean MSE across all cross validation samples
    mean_cv_mse = np.ones(shape=(1, k-1)) @ cv_mse_l / (k-1)

    # Add that mean MSE to the list
    cv_mse.append(mean_cv_mse[0,0])

# Find lambda which minimzes the cross validated MSE
idx_lmin = np.argmin(cv_mse)
lstar = L[idx_lmin]

# Get associated MSE in the test data
_, _, msestar = regls(y_tr=y_tr, X_tr=X_tr, l=lstar, y_te=y_te, X_te=X_te)

# Display the results
print('\nOptimal cross-validated lambda: {:2.2f}'.format(lstar, 4))
print('Effective degrees of freedom: {:2.2f}'.format(effdf(X_tr, lstar)))
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
