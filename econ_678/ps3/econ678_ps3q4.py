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
def OLS(y, X):
    # Get number of observations n and number of coefficients k, using X.shape[1] = k, X.shape[0] = n
    n, k = X.shape[0], X.shape[1]

    # Calculate OLS coefficients
    beta_hat = inv(X.transpose() @ X) @ (X.transpose() @ y)

    # Get residuals
    U_hat = y - X @ beta_hat

    # Calculate component of middle part of EHW sandwich
    S = X * ( U_hat @ np.ones(shape=(1,k)) )

    # Calculate EHW variance/covariance matrix
    V_hat = ( n / (n - k) ) * inv(X.transpose() @ X) @ (S.transpose() @ S) @ inv(X.transpose() @ X)

    # Return coefficients and EHW variance/covariance matrix
    return beta_hat, V_hat

# Set seed
np.random.seed(678)

# Specify sample sizes
N = [50] #N = [10, 25, 50]

# Specify how often you want to run the experiment
E = 1 #E = 1000

# Specify the number of bootstrap simulations per experiment
B = 1 #B = 299

# Set up components of beta vector
beta_0 = 1
beta_1 = 0
beta_2 = 1

# Combine to (column) vector
beta = np.array([beta_0, beta_1, beta_2], ndmin=2).transpose()

# Set test level
alpha = .05

# Set up rejection counters
reject_OLS = 0

# Go through all iterations of the experiment
for e in range(E):
    # Generate components of X (as column vectors)
    X_1 = np.random.normal(loc=0, scale=1, size=(max(N), 1))
    X_2 = np.random.normal(loc=0, scale=1, size=X_1.shape)

    # Stack components (plus intercept)
    X = np.concatenate((np.ones(X_1.shape), X_1, X_2), axis=1)

    # Generate additional component for the error term
    V = np.random.chisquare(df=5, size=X_1.shape) - 5

    # Generate y
    y = X @ beta + V * X_1**2

    # Perform standard inference (using EHW standard errors)
    beta_hat_OLS, V_hat_OLS = OLS(y, X)

    CI = [beta_hat_OLS[1] - norm.ppf(1 - alpha/2) * np.sqrt(V_hat_OLS[1,1]),
        beta_hat_OLS[1] + norm.ppf(1 - alpha/2) * np.sqrt(V_hat_OLS[1,1])]

    test = OLS_builtin(y, X).fit(cov_type='HC1')

    if not CI[0] <= 0 <= CI[1]:
        reject_OLS += 1

print(reject_OLS/E)
