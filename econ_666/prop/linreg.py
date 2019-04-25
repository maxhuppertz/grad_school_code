################################################################################
### Part 1: Setup
################################################################################

# Import necessary packages
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from numpy.linalg import solve
from scipy.stats import norm

################################################################################
### Part 2: Auxiliary functions
################################################################################

# This function takes an input and converts it to a 'long' array; that is, this
# creates a two-dimensional output (vector or matrix), and the first dimension
# will be longer than the second dimension
def larry(x):
    # Inputs
    # x: [n, k] array-like, has to be convertible to a Numpy array
    #
    # Outputs
    # X: [max(n,k), min(n,k)] array

    # Convert input into a two-dimensional Numpy array
    X = np.array(x, ndmin=2)

    # Get the shape of the array
    n, k = X.shape

    # Check whether the second dimension is larger than the first
    if k > n:
        # If so, take the transpose
        X = X.transpose()

    # Return the array
    return X

################################################################################
### Part 3: Regression models
################################################################################

# This function just runs a standard linear regression of y on X
def ols(y, X, get_cov=True, cov_est='hc1', get_t=True, get_p=True,
        clustvar=None):
    # Inputs
    # y: [n,1] vector, LHS variables
    # X: [n,k] matrix, RHS variables
    # get_cov: boolean, if true, the function returns an estimate of the
    #          variance/covariance matrix, in addition to the OLS coefficients
    # cov_est: string, specifies which variance/covariance matrix estimator to
    #          use. Currently, must be either hmsd (for the homoskedastic
    #          estimator) or hc1 (for the Eicker-Huber-White HC1 estimator)
    # get_t: boolean, if true, the function returns t-statistics for the simple
    #        null of beta[i] = 0, for each element of the coefficient vector
    #        separately
    # get_p: boolean, if true, calculate the p-values for a two-sided test of
    #        beta[i] = 0, for each element of the coefficient vector separately
    #
    # Outputs:
    # beta_hat: [k,1] vector, coefficient estimates
    # V_hat: [k,k] matrix, estimate of the variance/covariance matrix
    # t: [k,1] vector, t-statistics
    # p: [k,1] vector, p-values

    # If p-values are necessary, then t-statistics will be needed
    if get_p and not get_t:
        get_t = True

    # If t-statistics are necessary, then the covariance has to be estimated
    if get_t and not get_cov:
        get_cov = True

    # Get number of observations n and number of coefficients k
    n, k = X.shape[0], X.shape[1]

    # Calculate OLS coefficients
    XXinv = solve(X.transpose() @ X, np.eye(k))  # Calculate (X'X)^(-1)
    beta_hat = XXinv @ (X.transpose() @ y)

    # Check whether covariance is needed
    if get_cov:
        # Get residuals
        U_hat = y - X @ beta_hat

        # Check which covariance estimator to use
        if cov_est == 'hmsd':
            # For the homoskedastic estimator, just calculate the standard
            # variance
            V_hat = ( 1 / (n - k) ) * XXinv * (U_hat.transpose() @ U_hat)
        elif cov_est == 'hc1':
            # Calculate component of middle part of EHW sandwich,
            # S_i = X_i u_i, which makes it very easy to calculate
            # sum_i X_i X_i' u_i^2 = S'S)
            S = ( U_hat @ np.ones(shape=(1,k)) ) * X

            # Calculate EHW variance/covariance matrix
            V_hat = ( n / (n - k) ) * XXinv @ (S.transpose() @ S) @ XXinv
        elif cov_est == 'cluster':
            # Calculate number of clusters
            J = len(np.unique(clustvar))

            # Same thing as S above, but needs to be a DataFrame, because pandas
            # has the groupby method, which is needed in the next step
            S = pd.DataFrame((U_hat @ np.ones(shape=(1,k))) * X)

            # Sum all covariates within clusters
            S = S.groupby(clustvar[:,0], axis=0).sum().values

            # Calculate cluster-robust variance estimator
            V_hat = (
                ( n / (n - k) ) * ( J / (J - 1) )
                * XXinv @ (S.transpose() @ S) @ XXinv)
        else:
            # Print an error message
            print('Error in ',ols.__name__,'(): The specified covariance '
                'method could not be recognized. Please specify another ',
                'method.',sep='')

            # Exit the program
            return

        # Replace NaNs as zeros (happen if division by zero occurs)
        V_hat[np.isnan(V_hat)] = 0

        # Check whether to get t-statistics
        if get_t:
            # Calculate t-statistics (I like having them as a column vector, but
            # to get that, I have to convert the square root of the diagonal
            # elements of V_hat into a proper column vector first)
            t = beta_hat / larry(np.sqrt(np.diag(V_hat)))

            # Check whether to calculate p-values
            if get_p:
                # Calculate p-values
                p = 2 * (1 - norm.cdf(np.abs(t)))

                # Return coefficients, variance/covariance matrix, t-statistics,
                # and p-values
                return beta_hat, V_hat, t, p
            else:
                # Return coefficients, variance/covariance matrix, and
                # t-statistics
                return beta_hat, V_hat, t
        else:
            # Return coefficients and variance/covariance matrix
            return beta_hat, V_hat
    else:
        # Otherwise, just return coefficients
        return beta_hat

################################################################################
### Part 4: Bootstrap algorithms
################################################################################

################################################################################
### 4.1: Single iterations
################################################################################

# Define one iteration of the Cameron, Gelbach, and Miller (2008) cluster robust
# wild bootstrap with the null imposed
def b_iter_cgm0(y, X, e_hat, beta_hat_R, CV, J, seed):
    # Set random number generator's seed
    np.random.seed(seed)

    # To get Rademacher disturbances, draw Bernoulli random variables
    eta = np.random.binomial(1, .5, size=J)

    # Then, change zeros to -1, and convert to a proper (column) vector
    eta = larry(eta - (eta == 0))

    # Use cluster indices to assign each unit its cluster's disturbance
    eta = eta[CV[:,0],:]

    # Get residuals for this bootstrap iteration
    estar = e_hat * eta

    # Get LHS variable for this bootstrap iteration
    ystar = X @ beta_hat_R + estar

    # Get t-statistic for this iteration
    _, _, tstar = ols(ystar, X, get_cov=True, cov_est='cluster', get_t=True,
                      get_p=False, clustvar=CV)

    # Return the t-statistic for this bootstrap iteration
    return tstar

################################################################################
### 4.2: Running algorithms
################################################################################

# Define a function to bootstrap confidence intervals for OLS
def boot_ols(y, X, alg='cgm0', B=4999, alpha=.05, clustvar=None, imp0=None,
             b0=0, seed=0, par=True):
    # Get number of available cores
    ncores = cpu_count()

    # Check which algorithm to use
    if alg =='cgm0':  # Cameron, Gelbach, and Miller (2008)
        # Get length of coefficient vectors
        k = X.shape[1]

        # Calculate number of clusters
        J = len(np.unique(clustvar))

        # Set cluster variable
        CV = clustvar

        # Get original sample unrestricted coefficient estimate and t-statistic
        beta_hat, _, t_hat = ols(y, X, get_cov=True, cov_est='cluster',
                                 get_t=True, get_p=False, clustvar=CV)

        # Get indicator for unrestricted elements of coefficient vector
        unrest = (imp0 == 0)

        # Get columns of X corresponding to unrestricted elements
        X_R = X[:, unrest[:,0]]

        # Set up restricted vector of coefficient estimates
        beta_hat_R = np.zeros(beta_hat.shape)

        # Replace restricted elements with null hypothesis restrictions
        beta_hat_R[imp0[:,0], :] = b0

        # Replace unrestricted elements with original sample restricted
        # coefficient estimates
        beta_hat_R[unrest[:,0], :] = ols(y, X_R, get_cov=False, get_t=False,
                                        get_p=False)

        # Get residuals
        e_hat = y - X @ beta_hat

        # Check whether to use parallel computing
        if par:
            # Get matrix of bootstrapped t-statistics (for now, this will be a
            # list), using parallel computing
            Tb = Parallel(n_jobs=ncores)(
                delayed(b_iter_cgm0)(y=y, X=X, e_hat=e_hat,
                                     beta_hat_R=beta_hat_R, CV=CV, J=J,
                                     seed=seed+b) for b in range(B))
        else:
            # Otherwise, do it in sequence
            Tb = [b_iter_cgm0(y=y, X=X, e_hat=e_hat, beta_hat_R=beta_hat_R,
                              CV=CV, J=J, seed=seed+b) for b in range(B)]

        # Convert list to actual matrix (this is [k,B])
        Tb = np.concatenate(Tb, axis=1)

        # Set up matrix of confidence intervals
        CI = np.zeros(shape=(k,2))

        # Go through all coefficients
        for i in range(k):
            # Get the bootstrapped t-statistics for the current coefficient
            Tbi = np.sort(Tb[i,:])
            # Get the upper and lower bounds of the alpha level confidence
            # interval
            CI[i,:] = [Tbi[np.int((alpha/2) * (B+1))],
                       Tbi[np.int((1 - alpha/2) * (B+1))]]

        # Return the point estimate, t-statistic, and confidence intervals
        return beta_hat, t_hat, CI
