########################################################################################################################
### Econ 678, PS3Q4: Create your own adventure (TM)
### Generates some data, then compares standard inference and various bootstrap procedures
########################################################################################################################

# Import necessary packages
import numpy as np
import warnings
from numpy.linalg import pinv
from scipy.stats import norm

# Define standard OLS regression, with Eicker-Huber-White (EHW) variance/covariance estimator
def OLS(y, X, get_cov=True):
    # Get number of observations n and number of coefficients k
    n, k = X.shape[0], X.shape[1]

    # Calculate OLS coefficients
    beta_hat = pinv(X.transpose() @ X) @ (X.transpose() @ y)

    # Check whether covariance is needed
    if get_cov:
        # Get residuals
        U_hat = y - X @ beta_hat

        # Calculate component of middle part of EHW sandwich (S_i = X_i u_i, meaning that it's easy to calculate
        # sum_i X_i X_i' u_i^2 = S'S)
        S = X * ( U_hat @ np.ones(shape=(1,k)) )

        # Calculate EHW variance/covariance matrix
        V_hat = ( n / (n - k) ) * pinv(X.transpose() @ X) @ (S.transpose() @ S) @ pinv(X.transpose() @ X)

        # Return coefficients and EHW variance/covariance matrix
        return beta_hat, V_hat
    else:
        # Otherwise, just return coefficients
        return beta_hat

# Define pairs bootstrap
def pairs_bootstrap(y, X, beta_hat, estimator=OLS, B=1000):
    # Get number of observations n and number of coefficients k
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
        beta_hat_star, V_hat_star = estimator(y_star, X_star)

        # Calculate t statistic
        T[b,:] = (beta_hat_star[:,0] - beta_hat[:,0]) / np.sqrt(np.diag(V_hat_star))

    # Return the matrix of bootstrap t statistics
    return T

# Define wild bootstrap (uses a Rademacher distribution for disturbances)
def wild_bootstrap(y, X, beta_hat, U_hat, estimator=OLS, B=1000):
    # Get number of observations n and number of coefficients k
    n, k = X.shape[0], X.shape[1]

    # Set up vector of bootstrap t statistics
    T = np.zeros(shape=(B,k))

    # Go through all bootstrap iterations
    for b in range(B):
        # Draw perturbations from a Rademacher distribution
        I = np.random.uniform(low=0, high=1, size=(n,1))
        eta = np.ones(shape=(n,1)) * ( (-1)**(I < .5) )  # Isn't Pyton cool sometimes?

        # Generate bootstrap data
        y_star = X @ beta_hat + U_hat * eta

        # Estimate model
        beta_hat_star, V_hat_star = estimator(y_star, X)

        # Calculate t statistic
        T[b,:] = (beta_hat_star[:,0] - beta_hat[:,0]) / np.sqrt(np.diag(V_hat_star))

    # Return the matrix of bootstrap t statistics
    return T

# Set seed
np.random.seed(678)

# Specify sample sizes
N = [10, 25, 50]

# Specify how often you want to run the experiment for each sample size
E = 1000

# Specify the number of bootstrap iterations per experiment
B = 299

# Set up components of beta vector
beta_0 = 1
beta_1 = 0
beta_2 = 1

# Combine to (column) vector
beta = np.array([beta_0, beta_1, beta_2], ndmin=2).transpose()

# Set test level
alpha = .05

# Display number of experiments and number of bootstrap iterations
print(E, 'experiments,', B, 'bootstrap iterations')

# Go through all sample sizes
for n in N:
    # Set up rejection counters
    reject_OLS = 0
    reject_PB = 0
    reject_WB_WIN = 0
    reject_WB_NULL = 0

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
        beta_hat_OLS, V_hat_OLS = OLS(y, X, get_cov=True)

        # Get t statistic for beta_1
        t_OLS = beta_hat_OLS[1,0] / np.sqrt(V_hat_OLS[1,1])

        # Check whether standard asymptotic test rejects
        if not norm.ppf(alpha/2) <= t_OLS <= norm.ppf(1 - alpha/2):
            reject_OLS += 1

        # Do the pairs bootstrap
        T_PB = pairs_bootstrap(y, X, beta_hat_OLS, B=B)

        # Get sorted vector of bootstrap t statistics for beta_1
        Q_PB = np.sort(T_PB[:,1])

        # Check whether the pairs bootstrap test rejects
        if not Q_PB[np.int((alpha/2) * (B+1))] <= t_OLS <= Q_PB[np.int((1 - alpha/2) * (B+1))]:
            reject_PB += 1

        # Calculate OLS residuals (the wild bootstrap needs these)
        U_hat = y - X @ beta_hat_OLS

        # Do the wild bootstrap without imposing the null (WIN)
        T_WB_WIN = wild_bootstrap(y, X, beta_hat_OLS, U_hat=U_hat, B=B)

        # Get sorted vector of bootstrap t statistics for beta_1
        Q_WB_WIN = np.sort(T_WB_WIN[:,1])

        # Check whether the wild bootstrap test rejects
        if not Q_WB_WIN[np.int((alpha/2) * (B+1))] <= t_OLS <= Q_WB_WIN[np.int((1 - alpha/2) * (B+1))]:
            reject_WB_WIN += 1

        # Estimate OLS under the null
        beta_hat_OLS_NULL = OLS(y, X[:,[0,2]], get_cov=False)

        # Add beta_1 = 0 back into beta_hat_OLS_NULL (obj=1 specifies the index before which values=0 is inserted)
        beta_hat_OLS_NULL = np.insert(beta_hat_OLS_NULL, obj=1, values=0, axis=0)

        # Get residuals under the null
        U_hat_NULL = y - X @ beta_hat_OLS_NULL

        # Do the wild bootstrap imposing the null
        T_WB_NULL = wild_bootstrap(y, X, beta_hat_OLS_NULL, U_hat_NULL, B=B)

        # Get sorted vector of bootstrap t statistics for beta_1
        Q_WB_NULL = np.sort(T_WB_NULL[:,1])

        # Check whether the wild bootstrap test rejects
        if not Q_WB_NULL[np.int((alpha/2) * (B+1))] <= t_OLS <= Q_WB_NULL[np.int((1 - alpha/2) * (B+1))]:
            reject_WB_NULL += 1

    # Print results for the current sample size
    print('Sample size:', n)
    print('Rejection rate for standard OLS:', reject_OLS / E)
    print('Rejection rate for pairs bootstrap:', reject_PB / E)
    print('Rejection rate for wild bootstrap (without imposing the null):', reject_WB_WIN / E)
    print('Rejection rate for wild bootstrap (imposing the null):', reject_WB_NULL / E)
