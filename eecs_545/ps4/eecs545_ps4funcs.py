################################################################################
### EECS 545, problem set 4 functions
################################################################################

################################################################################
### 1: Load packages
################################################################################

import numpy as np

################################################################################
### 2.4: Problem 4
################################################################################


# Multivariate Gaussian density
def multivariate_gaussian(x, mu, Sigma):
    """ Calculates multivariate Gaussian density

    Inputs
    x: d by 1 vector or list, point at which density is calculated
    mu: d by 1 vector, mean
    Sigma: d by d matrix, covariance matrix

    Outputs
    px: Scalar, density at x
    """
    # Check whether the provided point is a list
    if type(x) == list:
        # If so, convert it to an array
        x = np.array(x, ndmin=2).T

    # Get length of x
    if x.shape == ():  # That is, if x is a scalar
        d = 1
    else:
        d = x.shape[0]

    # Calculate density
    px = (
        np.exp(-(1/2) * (x - mu).T @ np.linalg.inv(Sigma) @ (x-mu))
        / ((2*np.pi)**(d/2) * np.sqrt(np.linalg.det(Sigma)))
        )

    # Return density (as a scalar)
    return px[0,0]


# Parameters of multivariate Gaussian marginal distribution
def marginal_distribution(I, mu, Sigma):
    """ Calculates parameters of multivariate Gaussian marginal density

    Inputs
    I: List, indices of the m elements for which to calculate marginal
    mu: d by 1 vector, mean of the unconditional distribution
    Sigma: d by d matrix, covariance matrix of the unconditional distribution

    Outputs
    mu_marg: m by 1 vector, marginal mean
    Sigma_marg: m by m matrix, marginal covariance matrix
    """
    # Get marginal mean vector
    mu_marg = mu[I,:]

    # Convert indices from a list to a vector
    idx = np.array(I, ndmin=2).T

    # Get marginal covariance matrix
    Sigma_marg = Sigma[idx, idx.T]

    # Return the result
    return mu_marg, Sigma_marg


# Parameters of multivariate Gaussian conditional distribution
def conditional_distribution(I, U, mu, Sigma):
    """ Calculate parameters of multivariate Gaussian conditional density

    Inputs
    I: List, indices of the m elements which are not conditioned on
    U: k by 1 vector, conditioning values for the other k elements
    mu: d by 1 vector, mean of the unconditional distribution
    Sigma: d by d matrix, covariance matrix of the unconditional distribution

    Outputs
    mu_cond: m by 1 vector, conditional mean
    Sigma_cond: m by m matrix, conditional covariance matrix
    """
    # Get the marginal mean and covariance for unconditioned elements V
    mu_V, Sigma_V = marginal_distribution(I, mu, Sigma)

    # Get the marginal mean and covariance for conditioning elements U
    notI = [c for c in np.arange(Sigma.shape[0]) if c not in I]  # Index
    mu_U, Sigma_U = marginal_distribution(notI, mu, Sigma)

    # Convert both indices into an array
    idx_V = np.array(I, ndmin=2).T
    idx_U = np.array(notI, ndmin=2).T

    # Get the cross covariance terms
    Sigma_UV = Sigma[idx_V, idx_U.T]

    # Get the conditional mean and variance
    mu_cond = mu_V + Sigma_UV @ np.linalg.inv(Sigma_U) @ (U - mu_U)
    Sigma_cond = Sigma_V - Sigma_UV @ np.linalg.inv(Sigma_U) @ Sigma_UV.T

    # Return the results
    return mu_cond, Sigma_cond


################################################################################
### 2.5: Problem 5
################################################################################


# Define a function to calculate a Gaussian kernel
def gaussian_kernel(x1, x2, sigma2=1):
    """ Calculates a Gaussian kernel

    Inputs
    x1: d by 1 vector, first instance
    x2: d by 1 vector, second instance
    sigma2: scalar, kernel parameter

    Output
    k: scalar, kernel value
    """
    # Calculate the kernel as exp{-(||x1 - x2||^2) / (2 * sigma2)}
    k = np.exp( -((x1 - x2).T @ (x1 - x2)) / (2 * sigma2) )

    # Return the result
    return k


# Define a function which returns a covariance matrix for a given kernel
# function
def kernel_cov(X, kernel=gaussian_kernel, args=[]):
    """ Calculates covariance matrix for a kernel function

    Inputs
    X: d by n matrix, feature instances
    kernel: Function, kernel function
    args: List, additional arguments to be passed to kernel

    Output
    Sigma: n by n matrix, covariance matrix for the provied kernel function
    """
    # Get the number of instances n
    n = X.shape[1]

    # Set up the covariance matrix (which has to be n by n)
    Sigma = np.identity(n)

    # Go through all instances
    for i in range(n):
        # For each instance, go through all instances with equal or higher
        # indices (this avoids calculating elements twice, but relies on the
        # kernel being symmetric)
        for j in range(i, n):
            # Get instances i and j
            xi = X[:, i:i+1]
            xj = X[:, j:j+1]

            # Fill in the corresponding elements in the covariance matrix
            Sigma[i,j] = Sigma[j,i] = kernel(xi, xj, *args)

    # Return the result
    return Sigma

################################################################################
### 2.6: Problem 6
################################################################################


# Define a function to return the y coordinate of the classifier threshold line
# for a given x coordinate
def classline(x1, w):
    """ Calculates two dimensional classifier line

    Inputs
    x1: scalar, first coordinate (horizontal axis) for the classifier line
    w: 3 by 1 vector, weights

    Outputs
    x2: scalar, second coordinate (vertical axis) for the classifier line
    """
    x2 = -(w[0,0] + w[1,0] * x1) / w[2,0]
    return x2


# Define a function which returns LDA or QDA classification boundaries
def discriminant_bound(X0, X1, pi=.5, common_cov=True, nbound=100):
    """ Calculates boundaries for the LDA and QDA classifiers

    Inputs
    X0: d by n0 vector, features for class 0
    X1: d by n1 vector, features for class 1
    pi: Scalar, prior probability of class 1
    common_cov: Boolean, if True, uses LDA
    nbound: scalar, number of points for which to calculate the boundary line

    Outputs if common_cov is True:
    bound_X1: List, first coordinates of the classifier line
    bound_X1: List, second coordinates of the classifier line

    Outputs if common_cov is False:
    bound_X1: nbound by nbound matrix, first coordinates of a set of points
    bound_X2: nbound by nbound matrix, second coordinates of a set of points
    Y: nbound by nbound matrix, classifier value at those points
    """
    # Get the means for each class
    mu0 = np.array(np.mean(X0, axis=1), ndmin=2).T
    mu1 = np.array(np.mean(X1, axis=1), ndmin=2).T

    # Combine the two classes into one data set
    X = np.concatenate([X0, X1], axis=1)

    # Get the minimum and maximum value along the first dimension of the data
    xmin = np.amin(X[0,:])
    xmax = np.amax(X[0,:])

    # Set up a list of values X1 for which the other coordinate of the boundary
    # line X2 will be calculated
    bound_X1 = np.linspace(start=xmin, stop=xmax, num=nbound)

    # Check whether to use LDA
    if common_cov:
        # If so, get the inverse covariance for the combined data
        SigmaI = np.linalg.inv(np.cov(X))

        # Calculate the feature weight
        w1 = SigmaI @ (mu1 - mu0)

        # Calculate the intercept
        w0 = (
            np.log(pi/(1-pi)) - .5*(mu1.T @ SigmaI @ mu1 - mu0.T @ SigmaI @ mu0)
            )

        # Combine the two into a vector
        w = np.concatenate([w0, w1])

        # Calculate the X2 coordinates of the classifier line
        bound_X2 = [classline(x, w) for x in bound_X1]

        # Return the result
        return bound_X1, bound_X2
    else:
        # Otherwise, also get the min and max values for the other feature
        # dimension
        ymin = np.amin(X[1,:])
        ymax = np.amax(X[1,:])

        # Set up a list of coordinates for X1. (For QDA, solving for the
        # boundary is tedious, so I use a brute force approach in which I simply
        # calculate the classifier value at a bunch of points, and the use
        # matplotlib.pyplot's contour() plot to graph the boundary.)
        bound_X2 = np.linspace(start=ymin, stop=ymax, num=nbound)

        # Get the covariance for each class
        Sigma0 = np.cov(X0)
        Sigma1 = np.cov(X1)

        # Get the determinants of those
        det0 = np.linalg.det(Sigma0)
        det1 = np.linalg.det(Sigma1)

        # Get the inverse covariance for each class
        SigmaI0 = np.linalg.inv(Sigma0)
        SigmaI1 = np.linalg.inv(Sigma1)

        # Combine the two lists into a grid
        bound_X1, bound_X2 = np.meshgrid(bound_X1, bound_X2)

        # Set up a matrix for the resulting classifier values
        Y = np.zeros(shape=(nbound,nbound))

        # Go through all points in the grid
        for i in range(nbound):
            for j in range(nbound):
                # Combine the two current points into a vector
                x = np.array([bound_X1[i,j], bound_X2[i,j]], ndmin=2).T

                # Save its classifier value
                Y[i,j] = (
                    (1/2) * (x - mu0).T @ SigmaI0 @ (x - mu0)
                    - (1/2) * (x - mu1).T @ SigmaI1 @ (x - mu1)
                    + (1/2) * np.log(det0 / det1)
                    - np.log(pi / (1-pi))
                    )

        # Return the grid of points and the classifier values
        return bound_X1, bound_X2, Y
