# Import necessary packages
import numpy as np
from numpy.linalg import solve
from scipy.stats import norm

# This function just runs a standard linear regression of y on X
def ols(y, X, get_cov=True, cov_est='hc1', get_t=True, get_p=True):
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
        cov_est = True

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
            # For the homoskedastic estimator, just calculate the standard variance
            V_hat = ( 1 / (n - k) ) * XXinv * (U_hat.transpose() @ U_hat)
        elif cov_est == 'hc1':
            # Calculate component of middle part of EHW sandwich,
            # S_i = X_i u_i, which makes it very easy to calculate
            # sum_i X_i X_i' u_i^2 = S'S)
            S = ( U_hat @ np.ones(shape=(1,k)) ) * X

            # Calculate EHW variance/covariance matrix
            V_hat = ( n / (n - k) ) * XXinv @ (S.transpose() @ S) @ XXinv
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
            # elements of V_hat into a proper vector first and transpose them,
            # since Numpy loves its row vectors for some reason)
            t = beta_hat / np.array(np.sqrt(np.diag(V_hat)),ndmin=2).transpose()

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
