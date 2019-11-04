################################################################################
### EECS 545, problem set 4 question 3
### Bayesian Naïve Bayes classifier
################################################################################

################################################################################
### 1: Load packages, set directories and files, set graph options
################################################################################

# Import necessary packages
import numpy as np
import os  # Only needed to set main directory
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

################################################################################
### X: Define functions
################################################################################


# Define Bayesian naïve Bayes classifier
def bnb(X_tr, y_tr, X_te, y_te, alpha=1, beta=1):
    """ Implement Bayesian naïve Bayes classifier

    Inputs
    X_tr: d by n_tr matrix, training features
    y_tr: n_tr vector, training labels
    X_te: d by n_te matrix, test features
    y_te: n_te vector, test labels
    alpha: Scalar, first parameter for Beta(alpha, beta) prior
    beta: Scalar, second parameter for Beta(alpha, beta) prior

    Outputs
    y_hat: n_te vector, predicted labels
    err: Scalar, error rate, <# of test misclassifications> / n_te
    """
    # Get number of features d and training instances n
    d, n = X_tr.shape

    # Get number of test instances n_te
    n_te = X_te.shape[1]

    # Get number of instances labeled 1 in the training data
    n1 = y_tr.sum()

    # Get number of instances labeled 0 in the training data
    n0 = n - n1

    # Calculate probability of class 1, based on prior and training data
    pi = (n1 + alpha) / (n + alpha + beta)

    # Make indicators for class 0 and class 1
    c0 = y_tr == 0
    c1 = ~c0

    # Get the number of times a given word appears in the training documents,
    # for each word in the vocabulary, and divide by the number of documents in
    # that class to get frequencies
    Nx0 = X_tr[:,c0].sum(axis=1) / n1
    Nx1 = X_tr[:,c1].sum(axis=1) / n0

    # Calculate weights theta, based on prior and frequencies
    theta0 = (Nx0 + alpha) / (n0 + alpha + beta)
    theta1 = (Nx1 + alpha) / (n0 + alpha + beta)

    # Calculate intercept of the classifier line (the np.ones() simply sums up)
    w0 = (
        np.log(pi / (1-pi))
        + np.log((1 - theta1) / (1 - theta0)) @ np.ones(shape=(d,1))
        )

    # Calculate classifier line weights for each feature
    w1 = np.log( (theta1 * (1-theta0)) / (theta0 * (1-theta1)) )

    # Combine the two into one vector
    w = np.concatenate([w0, w1])

    # Augment test features by adding an intercept
    X_te = np.concatenate([np.ones(shape=(1, n_te)), X_te], axis=0)

    # Calculate classifier
    y_hat = ((w @ X_te) >= 0).astype(int)

    # Calculate error rate
    err = (y_te != y_hat).sum() / n_te

    # Return the predicted labels and error rate
    return y_hat, err

################################################################################
### 2: Evaluate Bayesian naïve Bayes classifier
################################################################################

# Load the training data, enforcing the convention that X is d by n
train_features = np.load('spam_train_features.npy').T
train_labels = np.load('spam_train_labels.npy')

# Load the test data, enforcing the convention that X is d by n
test_features = np.load('spam_test_features.npy').T
test_labels = np.load('spam_test_labels.npy')

# Calculate naïve Bayes classifier, get the error rate
_, err = bnb(X_tr=train_features, y_tr=train_labels,
             X_te=test_features, y_te=test_labels)

# Display the result
print()
print('Error rate: {:1.5f}'.format(err))
