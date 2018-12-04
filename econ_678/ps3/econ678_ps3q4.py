########################################################################################################################
### Econ 678, PS3Q4: Create your own adventure (TM)
### Generates some data, then compares standard inference and various bootstrap procedures
########################################################################################################################

# Import necessary packages
import numpy as np

# Set seed
np.random.seed(678)

# Specify sample sizes
N = [10] #N = [10, 25, 50]

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

# Go through all iterations of the experiment
for e in range(E):
    # Generate components of X (as column vectors)
    X_1 = np.random.normal(loc=0, scale=1, size=(max(N), 1))
    X2 = np.random.normal(loc=0, scale=1, size=X_1.shape)

    # Stack components (plus intercept)
    X = np.concatenate((np.ones(X_1.shape), X_1, X2), axis=1)

    # Generate additional component for the error term
    V = np.random.chisquare(df=5, size=X_1.shape) - 5

    # Generate y
    y = X @ beta + V * X_1**2

    # Perform standard inference (using HAC standard errors)
    beta_hat_OLS = (X.transpose() @ X)**(-1) @ (X.transpose() @ y)
