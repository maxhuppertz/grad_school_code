########################################################################################################################
### Econ 678, PS3Q4: Create your own adventure (TM)
### Generates some data, then compares standard inference and various bootstrap procedures
########################################################################################################################

# Import necessary packages
import numpy as np
from numpy.linalg import inv
from scipy.stats import norm
from statsmodels.regression.linear_model import OLS as OLS_builtin

# Define standard OLS regression, with Eicker-Huber-White (EHW) variance/covariance estimator
def OLS(y, X, get_cov=True):
    # Get number of observations n and number of coefficients k, using X.shape[1] = k, X.shape[0] = n
    n, k = X.shape[0], X.shape[1]

    # Calculate OLS coefficients
    beta_hat = inv(X.transpose() @ X) @ (X.transpose() @ y)

    # Check whether covariance is needed
    if get_cov:
        # Get residuals
        U_hat = y - X @ beta_hat

        # Calculate component of middle part of EHW sandwich
        S = X * ( U_hat @ np.ones(shape=(1,k)) )

        # Calculate EHW variance/covariance matrix
        V_hat = ( n / (n - k) ) * inv(X.transpose() @ X) @ (S.transpose() @ S) @ inv(X.transpose() @ X)

        # Return coefficients and EHW variance/covariance matrix
        return beta_hat, V_hat
    else:
        # Otherwise, just return coefficients
        return beta_hat

# Define pairs bootstrap
def pairs_bootstrap(y, X, beta_hat, B=1000):
    # Get sample size and size of the coefficient vector
    n, k = X.shape[0], X.shape[1]

    # Set up vector of bootstrap t statistics
    T = np.zeros(shape=(B,k))

    # Go through all bootstrap iterations
    for b in range(B):
        # Draw indices for bootstrap sample
        I = np.random.randint(low=0, high=n, size=n)

        # Get bootstrap data
        y_star, X_star = y[I], X[I,:]

        # Estimate model
        beta_hat_star, V_hat_star = OLS(y_star, X_star)

        # Calculate t statistic
        T[b,:] = np.sqrt(n) * (beta_hat_star[:,0] - beta_hat[:,0]) #/ np.sqrt(np.diag(V_hat_star))

    # Return the matrix of bootstrap t statistics
    return T

# Set seed
np.random.seed(678)

# Specify sample sizes
N = [50]

# Specify how often you want to run the experiment for each sample size
E = 100

# Specify the number of bootstrap simulations per experiment
B = 400

# Set up components of beta vector
beta_0 = 1
beta_1 = 0
beta_2 = 1

# Combine to (column) vector
beta = np.array([beta_0, beta_1, beta_2], ndmin=2).transpose()

# Set test level
alpha = .05

# Go through all sample sizes
for n in N:
    # Set up rejection counters
    reject_OLS = 0
    reject_PB = 0

    # Go through all iterations of the experiment
    for e in range(E):
        # Generate components of X (as column vectors)
        X_1 = np.random.normal(loc=0, scale=1, size=(n, 1))
        X_2 = np.random.normal(loc=0, scale=1, size=(n, 1))

        # Stack components (plus intercept)
        X = np.concatenate((np.ones(shape=(n, 1)), X_1, X_2), axis=1)

        # Generate additional component of the error term
        V = np.random.chisquare(df=5, size=(n, 1)) - 5

        # Generate y
        y = X @ beta + V * X_1**2

        # Perform standard inference (using EHW standard errors)
        beta_hat_OLS, V_hat_OLS = OLS(y[0:n], X[0:n,:])
        #test = OLS_builtin(y, X).fit(cov_type='HC1')

        # Get standard asymptotic confidence interval
        CI_OLS = [
            beta_hat_OLS[1] - norm.ppf(1 - alpha/2) * np.sqrt(V_hat_OLS[1,1]),
            beta_hat_OLS[1] + norm.ppf(1 - alpha/2) * np.sqrt(V_hat_OLS[1,1])
            ]

        # Check whether the standard test rejects
        if not CI_OLS[0] <= 0 <= CI_OLS[1]:
            reject_OLS += 1

        # Do the pairs bootstrap
        T_PB = pairs_bootstrap(y, X, beta_hat_OLS, B=B)

        # Get sorted vector of t statistics for beta_1
        Q_PB = np.sort(T_PB[:,1])

        # Get pairs bootstrap confidence interval
        CI_PB = [
            beta_hat_OLS[1] - Q_PB[np.int(np.ceil((1 - alpha/2) * B))] * np.sqrt(V_hat_OLS[1,1]),
            beta_hat_OLS[1] + Q_PB[np.int(np.floor(alpha/2 * B))] * np.sqrt(V_hat_OLS[1,1])
            ]

        # Check whether the pairs bootstrap test rejects
        if not CI_PB[0] <= 0 <= CI_PB[1]:
            reject_PB += 1

    # Print results for the current sample size
    print('Sample size:', n)
    print('Rejection rate for standard OLS:', reject_OLS / E)
    print('Rejection rate for pairs bootstrap:', reject_PB / E)
