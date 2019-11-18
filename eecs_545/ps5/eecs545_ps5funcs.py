################################################################################
### EECS 545, problem set 5 functions
################################################################################

################################################################################
### 1: Load packages
################################################################################

import numpy as np

################################################################################
### 2: Define functions
################################################################################

################################################################################
### 2.1: Problem 1
################################################################################


# Define Bayesian naïve Bayes classifier
def bnb(X_tr, y_tr, X_te, y_te, alpha=1, beta=1):
    """ Implements Bayesian naïve Bayes classifier

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
    # for each word in the vocabulary
    Nx0 = X_tr[:,c0].sum(axis=1)
    Nx1 = X_tr[:,c1].sum(axis=1)

    # Calculate weights theta, based on prior and frequencies
    theta0 = (Nx0 + alpha) / (n0 + alpha + beta)
    theta1 = (Nx1 + alpha) / (n1 + alpha + beta)

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
### 2.1: Problem 3
################################################################################

# Define a function to do PCA
def pca(X, K=[1]):
    """ Performs PCA

    Inputs
    X: d by n matrix, features
    K: list, set of principal components to evaluate

    Outputs
    R2: list, i-th element is the fraction of total variance explained by the
        first K[i] principal components
    L: length n array, ordered eigenvalues (largest to smallest)
    U: d by d matrix, associated eigenvectors
    """
    # Get the number of features d and instances n
    d, n = X.shape

    # Make a length n vector of ones
    ones = np.ones(shape=(n,1))

    # Demand the features
    mu = X @ ones / n
    Xc = X - mu @ ones.T

    # Get the gram matrix of the features
    G = Xc @ Xc.T

    # Get eigenvalues l and eigenvectors U (it's more efficient to use
    # np.linalg.eigh() instead of np.linalg.eig(), since the former takes
    # advantage of the fact that Xc Xc' is symmetric, and returns ordered
    # eigenvalues)
    L, U = np.linalg.eigh(G)

    # Unfortunately, the eigenvalues and -vectors are sorted in ascending order
    # (of the eigenvalues), which is kind of annoying, so flip that
    L = np.flip(L)
    U = np.flip(U, axis=1)

    # Calculate total variance of the demeaned features
    V = np.diag(G).sum()

    # Set up an array of the ratios of explained variance to total variance for
    # different numbers of principal components (akin to an R-squared in OLS,
    # hence the name)
    R2 = np.zeros(len(K))

    # Get the explained variance for each number of principal components
    Ve = [sum(L[0:k]) for k in K]

    # Get the fraction of total variance explained (kind of like an R squared in
    # OLS, hence the name)
    R2 = Ve / V

    # Return the results
    return R2, L, U
