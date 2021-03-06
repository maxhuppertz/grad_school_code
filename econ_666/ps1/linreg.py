# Import necessary packages
import numpy as np
from numpy.linalg import solve

# This function just runs a standard linear regression of y on X
def ols(y, X, get_cov=True, cov_est='hc1'):
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

        # Return coefficients and EHW variance/covariance matrix
        return beta_hat, V_hat
    else:
        # Otherwise, just return coefficients
        return beta_hat
