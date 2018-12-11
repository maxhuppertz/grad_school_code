########################################################################################################################
### Econ 678, PS3Q4: Create your own adventure (TM)
### Runs a Monte Carlo experiment that compares standard inference and three bootstrap procedures
########################################################################################################################

# Import necessary packages
import multiprocessing as mp
import numpy as np
import time
import warnings
from joblib import Parallel, delayed
from numpy.linalg import pinv
from scipy.stats import norm

########################################################################################################################
### Part 1: Define functions
########################################################################################################################

# Set up a function which does standard OLS regression, with Eicker-Huber-White (EHW) variance/covariance estimator
# (HC1, i.e. it has the n / (n - k) correction)
def OLS(y, X, get_cov=True):
    # Get number of observations n and number of coefficients k
    n, k = X.shape[0], X.shape[1]

    # Calculate OLS coefficients (pinv() uses the normal matrix inverse if X'X is invertible, and the Moore-Penrose
    # pseudo-inverse otherwise; invertibility of X'X can be an issue with the pairs bootstrap if sample sizes are small,
    # which is why this is helpful)
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

# Set up a function which does the bootstraps (I originally defined different functions, but really that just means I
# ran many many more for-loops, which was extremely inefficient)
def bootstrap(y, X, beta_hat, U_hat, beta_hat_null, U_hat_null, B=999):
    # Get number of observations n and number of coefficients k
    n, k = X.shape[0], X.shape[1]

    # Set up vector of bootstrap t statistics (output of this function)
    T = np.zeros(shape=(B,k*3))

    # Go through all bootstrap iterations
    for b in range(B):
        # First, do the pairs bootstrap
        # Draw indices for bootstrap sample
        I = np.random.randint(low=0, high=n, size=n)

        # Estimate model on bootstrap data
        beta_hat_star, V_hat_star = OLS(y[I], X[I,:])

        # Calculate t statistic (remember Python's zero indexing: T[b,:3] allocates to column elements 0, 1, and 2)
        T[b,:3] = (beta_hat_star[:,0] - beta_hat[:,0]) / np.sqrt(np.diag(V_hat_star))

        # Second, do the wild bootstrap without imposing the null
        # Draw perturbations from a Rademacher distribution
        I = np.random.binomial(n=1, p=.5, size=(n,1))
        eta = np.ones(shape=(n,1)) - 2*(I == 0)

        # Generate bootstrap data
        y_star = X @ beta_hat + U_hat * eta

        # Estimate model
        beta_hat_star, V_hat_star = OLS(y_star, X)

        # Calculate t statistic
        T[b,3:6] = (beta_hat_star[:,0] - beta_hat[:,0]) / np.sqrt(np.diag(V_hat_star))

        # Third, do the wild bootstrap without imposing the null
        # Draw perturbations from a Rademacher distribution
        I = np.random.binomial(n=1, p=.5, size=(n,1))
        eta = np.ones(shape=(n,1)) - 2*(I == 0)

        # Generate bootstrap data
        y_star = X @ beta_hat_null + U_hat_null * eta

        # Estimate model
        beta_hat_star, V_hat_star = OLS(y_star, X)

        # Calculate t statistic
        T[b,6:] = (beta_hat_star[:,0] - beta_hat_null[:,0]) / np.sqrt(np.diag(V_hat_star))

    # Return the matrix of bootstrap t statistics
    return T

# Define a function to run E experiments for a given sample size n, using B bootstrap iterations and testing at level
# alpha for standard OLS and all bootstraps
def run_experiments(n, beta, B=999, E=1000, alpha=.05):
    # Set seed
    np.random.seed(n)

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
        y = X @ beta + V * (X_1**2)

        # Perform standard inference (using EHW standard errors)
        beta_hat_OLS, V_hat_OLS = OLS(y, X, get_cov=True)

        # Get t statistic for beta_1
        t_OLS = beta_hat_OLS[1] / np.sqrt(V_hat_OLS[1,1])

        # Check whether standard asymptotic test rejects
        if not norm.ppf(alpha/2) <= t_OLS <= norm.ppf(1 - alpha/2):
            reject_OLS += 1

        # Calculate OLS residuals (the wild bootstrap needs these)
        U_hat_OLS = y - X @ beta_hat_OLS

        # Estimate OLS under the null (the wild bootstrap with the null imposed needs this)
        beta_hat_OLS_NULL = OLS(y, X[:,[0,2]], get_cov=False)

        # Add beta_1 = 0 back into beta_hat_OLS_NULL (obj=1 specifies the index before which values=0 is inserted)
        beta_hat_OLS_NULL = np.insert(beta_hat_OLS_NULL, obj=1, values=0, axis=0)

        # Get residuals under the null (also for the wild bootstrap with the null imposed)
        U_hat_OLS_NULL = y - X @ beta_hat_OLS_NULL

        # Do the bootstraps
        T = bootstrap(y, X, beta_hat_OLS, U_hat_OLS, beta_hat_OLS_NULL, U_hat_OLS_NULL, B=B)

        # Get sorted vector of pairs bootstrap t statistics for beta_1
        Q_PB = np.sort(T[:,1])

        # Check whether the pairs bootstrap test rejects
        if not Q_PB[np.int((alpha/2) * (B+1))] <= t_OLS <= Q_PB[np.int((1 - alpha/2) * (B+1))]:
            reject_PB += 1

        # Get sorted vector of wild boostrap without imposing the null (WIN) t statistics for beta_1
        Q_WB_WIN = np.sort(T[:,4])

        # Check whether the wild bootstrap test rejects
        if not Q_WB_WIN[np.int((alpha/2) * (B+1))] <= t_OLS <= Q_WB_WIN[np.int((1 - alpha/2) * (B+1))]:
            reject_WB_WIN += 1

        # Get sorted vector of wild bootstrap with the null imposed t statistics for beta_1
        Q_WB_NULL = np.sort(T[:,7])

        # Check whether the wild bootstrap test rejects
        if not Q_WB_NULL[np.int((alpha/2) * (B+1))] <= t_OLS <= Q_WB_NULL[np.int((1 - alpha/2) * (B+1))]:
            reject_WB_NULL += 1

    # Print results for the current sample size
    print('Sample size: ', n,
        '\nRejection rate for standard OLS: ', reject_OLS / E,
        '\nRejection rate for pairs bootstrap: ', reject_PB / E,
        '\nRejection rate for wild bootstrap (without imposing the null): ', reject_WB_WIN / E,
        '\nRejection rate for wild bootstrap (imposing the null): ', reject_WB_NULL / E,
        sep='')

########################################################################################################################
### Part 2: Set up & run experiments
########################################################################################################################

# Specify sample sizes
N = [30, 100, 1000]

# Specify how often you want to run the experiment for each sample size
E = 1000

# Specify the number of bootstrap iterations per experiment
B = 4999

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

# Run experiments in parallel, using all but one of the available cores
Parallel(n_jobs=mp.cpu_count() - 1)(delayed(run_experiments)(n, beta, B=B, E=E, alpha=alpha) for n in N)
