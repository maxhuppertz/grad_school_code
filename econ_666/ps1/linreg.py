# This function just runs a standard linear regression of y on X
def ols(y, X, get_cov=True, cov_est='homoskedastic'):
    # Import necessary packages
    import numpy as np
    from numpy.linalg import solve

    # Get number of observations n and number of coefficients k
    n, k = X.shape[0], X.shape[1]

    # Calculate OLS coefficients
    XXinv = solve(X.transpose() @ X, np.eye(k))  # (X'X)^(-1)
    beta_hat = XXinv @ (X.transpose() @ y)

    # Check whether covariance is needed
    if get_cov:
        # Get residuals
        U_hat = y - X @ beta_hat

        # Check which covariance estimator to use
        if cov_est == 'homoskedastic':
            # For the homoskedastic estimator, just calculate the standard variance
            V_hat = ( n / (n - k) ) * XXinv * (U_hat.transpose() @ U_hat)
        elif cov_est == 'hc1':
            # Calculate component of middle part of EHW sandwich (S_i = X_i u_i, meaning that it's easy to calculate
            # sum_i X_i X_i' u_i^2 = S'S)
            S = ( U_hat @ np.ones(shape=(1,k)) ) * X

            # Calculate EHW variance/covariance matrix
            V_hat = ( n / (n - k) ) * XXinv @ (S.transpose() @ S) @ XXinv

        # Return coefficients and EHW variance/covariance matrix
        return beta_hat, V_hat
    else:
        # Otherwise, just return coefficients
        return beta_hat
