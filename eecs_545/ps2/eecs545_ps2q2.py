################################################################################
### EECS 545, problem set 1 question 2
### Implement regularized least squares
################################################################################

################################################################################
### 1: Load packages, set directories and files, set graph options
################################################################################

# Import necessary packages
import numpy as np
import os  # Only needed to set main directory
import scipy.io as sio
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
fn_dset = 'bodyfat_data.mat'

################################################################################
### 2: Define functions
################################################################################


# Define regularized least squares estimator
def regls(y_tr, X_tr, l=10, y_te=None, X_te=None, demean=True, sdscale=True):
    """ Implements regularized least squares (ridge regression)

    Inputs
    y_tr: Training data responses, n_tr by 1 vector
    X_tr: Training data features, d by n_tr matrix
    l: Regularized least squares penalty, scalar
    y_te: Test data responses, n_te by 1 vector. If provided, the function will
          calculate the mean squared error on the test data. (Requires that X_te
          is also provided.)
    X_te: Test data features, d by n_te matrix. If provided, the function will
          calculate predicted values based on these features.
    demean: Boolean. If True, the function de-means any training and test
            features, using the mean of the training data.
    sdscale: Boolean. If True, the function scales any training and test
             features by the inverse of the standard deviation of the training
             features. (This happens after de-meaning, if demean is also True.)

    Outputs
    w_hat: Estimated coefficients, d by 1 vector
    y_hat: Predicted responses, n_te by 1 vector (if X_te was provided)
    mse: Mean squared error, scalar (if y_te and X_te were provided)

    Notes
    The intercept is not penalized
    """
    # Get number of features d and number of instances in the training data n_tr
    d, n_tr = X_tr.shape

    # Demean the training features, if desired
    if demean:
        ones_tr = np.ones(shape=(n_tr,1))
        mu_X_tr = (X_tr @ ones_tr) / n_tr  # It's useful to have this for later
        X_tr = X_tr - mu_X_tr @ ones_tr.transpose()

    # Scale the training features by the inverse of their standard deviation, if
    # desired
    if sdscale:
        sigma_X_tr = np.diag(1 / np.std(X_tr, axis=1))
        X_tr = sigma_X_tr @ X_tr

    # Add intercept to the training features
    X_tr = np.concatenate((np.ones(shape=(1,n_tr)), X_tr), axis=0)

    # Set up a modified identity matrix with the first element set to zero,
    # starting with just an identity matrix
    I_check = np.identity(d+1)

    # Set its first element to zero
    I_check[0,0] = 0

    # Get the estimates, solving a system of linear equations instead of
    # calculating the inverse (since I'm not trying to get standard errors)
    w_hat = np.linalg.solve(X_tr @ X_tr.transpose() + l * I_check, X_tr @ y_tr)

    # Check whether test data features were provided
    if X_te is not None:
        # If so, get number of instances in the test data
        n_te = X_te.shape[1]

        # Demean test data if desired, using training data feature means
        if demean:
            ones_te = np.ones(shape=(n_te,1))
            X_te = X_te - mu_X_tr @ ones_te.transpose()

        # Scale test data if desired, using inverse training data standard
        # deviations
        if sdscale:
            X_te = sigma_X_tr @ X_te

        # Add intercept to the test features
        X_te = np.concatenate((np.ones(shape=(1,n_te)), X_te), axis=0)

        # Get predicted responses
        y_hat = (X_te.transpose() @ w_hat)

        # Check whether test data responses were provided
        if y_te is not None:
            # If so, calculate MSE
            mse = (y_te - y_hat).transpose() @ (y_te - y_hat) / n_te

            # Return the estimated coefficients, predictions, and MSE
            return w_hat, y_hat, mse
        else:
            # Return the estimated coefficients and predictions
            return w_hat, y_hat
    else:
        # Return the estimated coefficients
        return w_hat

################################################################################
### 3: Load data
################################################################################

# Load the data set
data = sio.loadmat(fn_dset)

# Get responses and features
y = data['y']
X = data['X']

# Following the course convention, make sure X is d by n
X = X.transpose()

# Set number of training data instances (uses the first ntrain instances)
ntrain = 150

# Split data into training and test sample
X_tr = X[:, 0:ntrain]
y_tr = y[0:ntrain, :]
X_te = X[:, ntrain:]
y_te = y[ntrain:, :]

# Get the estimated coefficients
w_hat, _, mse = (
    regls(y_tr, X_tr, y_te=y_te, X_te=X_te)
    )

# Set up a novel instance (the transpose ensures it's a column vector)
x_new = np.array([100, 100], ndmin=2).transpose()

# Get the predicted response, and convert it back into non-standardized units of
# the response
_, y_hat = regls(y_tr, X_tr, X_te=x_new)

# Now, do the whole thing again, but without standardizing the features
w_hat_nst, _, mse_nst = (
    regls(y_tr, X_tr, y_te=y_te, X_te=X_te, demean=False, sdscale=False)
    )
_, y_hat_nst = regls(y_tr, X_tr, X_te=x_new, demean=False, sdscale=False)

# Set number of digits to round output to
nround = 4

# Display the results
print('\nWith standardized features')
print('\nEstimated coefficients (intercept, abdomen, hip)')
print('\n', np.round(w_hat, nround), sep='')
print('\nMSE in the test data:', np.round(mse[0,0], nround))
print('\nPredicted response for instance [', x_new[0,0], ', ', x_new[1,0],
      ']: ', np.round(y_hat[0,0], nround), '% body fat', sep='')
print('\nUsing features as is')
print('\nEstimated coefficients (intercept, abdomen, hip)')
print('\n', np.round(w_hat_nst, nround), sep='')
print('\nMSE in the test data:', np.round(mse_nst[0,0], nround))
print('\nPredicted response for instance [', x_new[0,0], ', ', x_new[1,0],
      ']: ', np.round(y_hat_nst[0,0], nround), '% body fat', sep='')
