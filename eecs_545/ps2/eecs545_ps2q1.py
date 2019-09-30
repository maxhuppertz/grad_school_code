################################################################################
### EECS 545, problem set 2 question 1
### Implement multiclass kNN
################################################################################

################################################################################
### 1: Load packages, set directories and files, set graph options
################################################################################

# Import necessary packages
import matplotlib.pyplot as plt
import numpy as np
import os  # Only needed to set main directory and make directories
import pandas as pd
from inspect import getsourcefile  # Only needed to set main directory
from scipy import stats

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

# Set training data set (has to exist in mdir)
fn_dset_tr = 'iris.test.csv'

# Set test data set (has to exist in mdir)
fn_dset_te = 'iris.train.csv'

# Set figures directory (doesn't have to exist)
fdir = 'figures'

# Set file name for 0-1 loss graph
fn_graph = 'kNN_loss.pdf'

# Set graph options
plt.rc('font', **{'family': 'serif', 'serif': ['lmodern']})
plt.rc('text', usetex=True)

################################################################################
### 2: Define functions
################################################################################


# Define kNN classifier
def knn(y_tr, X_tr, y_te, X_te, k):
    """ Implements a multiclass k nearest neighbor algorithm

    Inputs
    y_tr: Training data labels, as a column vector
    X_tr: Training data locations, which should be m by n, where m is the
          number of features, and n the number of instances
    y_te: Test data labels, as a column vector
    X_te: Test data locations, should be m by n
    k: Number of nearest neighbors to use

    Outputs:
    y_mode: Predicted labels for the test data, as a column vector
    loss: 0-1 loss in the test data, scalar
    """
    # Get the number of instances n and number of features m
    m, n_tr = X_tr.shape
    n_te = X_te.shape[1]

    # Get the vectors of squared feature norms
    d_tr = (X_tr * X_tr).transpose() @ np.ones(shape=(m,1))
    d_te = (X_te * X_te).transpose() @ np.ones(shape=(m,1))

    # Make an n by 1 vector of ones (the .astype(int) is not immediately
    # necessary, but will later become useful once I have to make index arrays)
    onevec_tr = np.ones(shape=(n_tr,1)).astype(int)
    onevec_te = np.ones(shape=(n_te,1)).astype(int)

    # Calculate the 'inner product' of the features matrix
    G = X_tr.transpose() @ X_te

    # Calculate the Euclidean square distance matrix
    D = onevec_tr @ d_te.transpose() - 2 * G + d_tr @ onevec_te.transpose()

    # Get the indices of the sorted distances for each instance. It turns out
    # that np.argsort() is kind of atrocious, in that this returns an array
    # where the ij element is the row index of the i+1-th nearest neighbor for
    # instance j. But since there are no column indices in the array, I cannot
    # use it to index other arrays just by itself.
    rowidxs = np.argsort(D, axis=0)

    # Make a matrix of labels by repeating the transposed label vector n times
    Y = y_tr @ onevec_te.transpose()

    # Make a matrix of column indices. This just repeats the column index of
    # each instance n times.
    colidxs = onevec_tr @ np.array(np.arange(n_te), ndmin=2)

    # Use row and column indices together to get the sorted matrix
    Y_allnn = Y.values[rowidxs, colidxs]

    # Get the mode, which is the predicted outcome for this classifier. (I don't
    # love this implementation, because stats.mode() will pick the lowest valued
    # mode in case of a tie. Random tie breaking would be better, but the only
    # easy way I could do that without using loops would rely on pandas. Since
    # we weren't supposed to use that for the actual implementation of the
    # classifier, I instead used stats.mode().)
    y_mode, _ = stats.mode(Y_allnn[0:k,:], axis=0)

    # Take the transpose, to ensure that this is the same shape as the input
    # vector y, assuming that the input was passed as a column vector
    y_mode =  y_mode.transpose()

    # Calculate 0-1 loss in the test data
    loss = (np.sum(y_mode != y_te) / n_te)

    # Return the predicted values and the loss
    return [y_mode, loss]

################################################################################
### 3: Load data
################################################################################

# Load the training and test data
tr_data = pd.read_csv('iris.train.csv')
te_data = pd.read_csv('iris.test.csv')

# Specify predictors
v_X = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# Specify labels
v_y = ['species']

# Get training label and feature data
y_tr = tr_data[v_y]
X_tr = tr_data[v_X]

# To be consistent with course conventions, I want to X to be m by n, rather
# than n by m (which it currently is), so transpose it
X_tr = X_tr.transpose()

# Get test label and feature data
y_te = te_data[v_y]
X_te = te_data[v_X]

# Take the transpose of X_te as well
X_te = X_te.transpose()

################################################################################
### 4: Plot 0-1 loss across different values of k
################################################################################

# Create the figures directory if it doesn't exist
if not os.path.isdir(mdir+'/'+fdir):
    os.mkdir(mdir+'/'+fdir)

# Change to the figures directory
os.chdir(mdir+'/'+fdir)

# Set up subplots
fig, ax = plt.subplots(figsize=(6.5,3.5))

# Set up a vector of possible values for k
K = np.arange(1, 50)

# Set up a vector of the corresponding 0-1 loss
L = []

# Calculate 0-1 loss for all values of K
L = [knn(y_tr, X_tr, y_te, X_te, k)[1] for k in K]

# Plot the 0-1 loss against the number of nearest neighbors
ax.plot(K, L)

# Set axis labels
ax.set_xlabel(r'Number of nearest neighbors $k$', fontsize=11)
ax.set_ylabel(r'Average 0-1 loss', fontsize=11)

# Add some more space after the horizontal axis label
ax.yaxis.labelpad = 10

# Get rid of unnecessary whitespace
fig.tight_layout()

# Save and close the figure (the bbox_inches='tight' helps remove even more
# unwanted whitespace)
plt.savefig(fn_graph, bbox_inches='tight')

# Close the plot
plt.close()

# Print a short message to let me know the program has finished
print('Done')
