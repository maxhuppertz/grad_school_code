################################################################################
### EECS 545, problem set 3 question 3
### Implement subgradient descent and stochastic subgradient descent for the
### optimal soft margin hyperplane
################################################################################

################################################################################
### 1: Load packages, set seed, set directories and files, set graph options
################################################################################

# Import necessary packages
import matplotlib.pyplot as plt
import numpy as np
import os  # Only needed to set main directory
import scipy.io as sio
from eecs545_ps3funcs import (graddesc, stochgraddesc, pen_hingeloss,
                              subgradient_hinge, classline)
from inspect import getsourcefile  # Only needed to set main directory

# Set random number generator's seed
np.random.seed(0)

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
fn_dset = 'nuclear.mat'

# Set figures directory (doesn't have to exist)
fdir = 'figures'

# Set file names for plots
fn_lossplot_gd = 'subgd_loss.pdf'  # Weights across iterations (GD)
fn_classplot_gd = 'subgd_line.pdf'  # MSE across iterations (GD)
fn_lossplot_sgd = 'stochsubgd_loss.pdf'  # Weights across iterations (SGD)
fn_classplot_sgd = 'stochsubgd_line.pdf'  # MSE across iterations (SGD)

# Set graph options
plt.rc('font', **{'family': 'Latin Modern Roman', 'serif': ['lmodern']})
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{lmodern}')

################################################################################
### 2: Load data
################################################################################

# Load the data set
data = sio.loadmat(fn_dset)

# Get features and labels (the transpose enforces the course convetion of y
# being n by 1)
X = data['x']
y = data['y'].T

################################################################################
### 3: Estimate optimal soft margin hyperplane
################################################################################

# Set step size parameter and step size scaling parameter (see docstrings for
# graddesc and stochgraddesc)
eta = 100
etasc = 1

# Set penalty term
l = .001

# Decide whether to add an intercept
icept = True

# Set up number subgradient descent iterations and stochastic SubGD epochs
K_gd = 3500
K_sgd = 35

# Set backtracking parameters for both algorithms
bt_gd = True
bt_sgd = True

# Select at which iteration or epoch to start backtracking
bts_gd = 1
bts_sgd = 10

# Estimate hyperplane parameters using SubGD
w_hat_gd, L_gd = graddesc(y, X, args=[l, icept], K=K_gd,
                          gradient=subgradient_hinge, demean=False,
                          sdscale=False, objfun=pen_hingeloss, eta=eta,
                          etasc=etasc, avggrad=False, addicept=icept,
                          backtrack=bt_gd, bstart=bts_gd)

# Get the index of lowest value of the objective function
idx_gd = np.argmin(L_gd)

# Get the associated function value
Lmin_gd = L_gd[idx_gd]

# Re-seed the random number generator
np.random.seed(0)

# Estimate hyperplane parameters using stochastic SubGD
w_hat_sgd, L_sgd = stochgraddesc(y, X, args=[l, icept], K=K_sgd, avggrad=True,
                                 gradient=subgradient_hinge, demean=False,
                                 sdscale=False, objfun=pen_hingeloss, eta=eta,
                                 etasc=etasc, addicept=icept, backtrack=bt_sgd,
                                 bstart=bts_sgd)

# Get the index of lowest value of the objective function
idx_sgd = np.argmin(L_sgd)

# Get the associated function value
Lmin_sgd = L_sgd[idx_sgd]

# Make a list of features
Xvars = ['Intercept', 'Total energy', 'Tail energy']

# Print a header for the results
print('\nHyperplane parameters')
print('\n{:<15} {:>8} {:>8}'.format('Variable', 'GD', 'SGD'))

# Go through all variables
for i, var in enumerate(Xvars):
    # Print the variable name and associated parameter
    print('{:<15}'.format(var),
          '{:8.4f}'.format(np.round(w_hat_gd[i,0], 4)),
          '{:8.4f}'.format(np.round(w_hat_sgd[i,0], 4))
    )

# Print the minimum achieved loss
print('\nSubGD min. average hinge loss: {:4.4f}'.format(np.round(Lmin_gd, 4)))
print('Achieved at iteration', idx_gd)
print('SubGD min. average hinge loss at final weights: {:4.4f}'.format(
    np.round(L_gd[-1], 4)))

# Print the minimum achieved loss
print('\nStSubGD min. average hinge loss: {:4.4f}'.format(
    np.round(Lmin_sgd, 4)))
print('Achieved at iteration', idx_sgd)
print('StSubGD min. average hinge loss at final weights: {:4.4f}'.format(
    np.round(L_sgd[-1], 4)))

################################################################################
### 4: Plot the results
################################################################################

# Create the figures directory if it doesn't exist
if not os.path.isdir(mdir+'/'+fdir):
    os.mkdir(mdir+'/'+fdir)

# Change to the figures directory
os.chdir(mdir+'/'+fdir)

################################################################################
### 4.1: Subgradient descent
################################################################################

# Set up a plot
fig, ax = plt.subplots(figsize=(6.5, 3.5))

# Plot the average hinge loss
ax.plot(L_gd, color='blue', alpha=.8)

# Label the axes
ax.set_xlabel('Iteration', fontsize=11)
ax.set_ylabel('Average hinge loss', fontsize=11)

# Get rid of unnecessary whitespace
fig.tight_layout()

# Save and close the figure (the bbox_inches='tight' helps remove even more
# unwanted whitespace)
plt.savefig(fn_lossplot_gd, bbox_inches='tight')
plt.close()

# Make an indicator for features with label 1
lab1 = y[:,0] == 1

# Get minimum and maximum values for the first dimension of X
xmin = np.floor(np.min(X[0,:]))
xmax = np.ceil(np.max(X[0,:]))

# Set up points to plot the classifier line
plotx = np.linspace(start=xmin, stop=xmax, num=2)
ploty = [classline(x1, w_hat_gd) for x1 in plotx]

# Set up another figure
fig, ax = plt.subplots(figsize=(6.5, 3.5))

# Plot both classes, using different colors
ax.scatter(X[0, ~lab1], X[1, ~lab1], color='green', alpha=.8, marker='1',
           label='Gamma rays')
ax.scatter(X[0, lab1], X[1, lab1], color='blue', alpha=.6, label='Neutrons')

# Plot the classifier line
ax.plot(plotx, ploty, color='darkgrey', linewidth=2.5, linestyle='-.',
        label='Classifier line')

# Label the axes
ax.set_xlabel('Total energy', fontsize=11)
ax.set_ylabel('Tail energy', fontsize=11)

# Add a legend
ax.legend(bbox_to_anchor=(.37, 1), loc='upper center', ncol=2, fancybox=False,
          edgecolor='black', framealpha=1)

# Get rid of unnecessary whitespace
fig.tight_layout()

# Save and close the figure (the bbox_inches='tight' helps remove even more
# unwanted whitespace)
plt.savefig(fn_classplot_gd, bbox_inches='tight')
plt.close()

################################################################################
### 4.2: Stochastic subgradient descent
################################################################################

# Set up a plot
fig, ax = plt.subplots(figsize=(6.5, 3.5))

# Make a vector counting the SGD epochs
epochs = np.arange(0, K_sgd, K_sgd / len(L_sgd))

# Plot the average hinge loss
ax.plot(epochs, L_sgd, color='blue', alpha=.8)

# Label the axes
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Average hinge loss', fontsize=11)

# Get rid of unnecessary whitespace
fig.tight_layout()

# Save and close the figure (the bbox_inches='tight' helps remove even more
# unwanted whitespace)
plt.savefig(fn_lossplot_sgd, bbox_inches='tight')
plt.close()

# Get the classifier line
ploty = [classline(x1, w_hat_sgd) for x1 in plotx]

# Set up another figure
fig, ax = plt.subplots(figsize=(6.5, 3.5))

# Plot both classes, using different colors
ax.scatter(X[0, ~lab1], X[1, ~lab1], color='green', alpha=.8, marker='1',
           label='Gamma rays')
ax.scatter(X[0, lab1], X[1, lab1], color='blue', alpha=.6, label='Neutrons')

# Plot the classifier line
ax.plot(plotx, ploty, color='darkgrey', linewidth=2.5, linestyle='-.',
        label='Classifier line')

# Label the axes
ax.set_xlabel('Total energy', fontsize=11)
ax.set_ylabel('Tail energy', fontsize=11)

# Add a legend
ax.legend(bbox_to_anchor=(.37, 1), loc='upper center', ncol=2, fancybox=False,
          edgecolor='black', framealpha=1)

# Get rid of unnecessary whitespace
fig.tight_layout()

# Save and close the figure (the bbox_inches='tight' helps remove even more
# unwanted whitespace)
plt.savefig(fn_classplot_sgd, bbox_inches='tight')
plt.close()
