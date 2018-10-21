import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import laplace

# Set true theta parameter
theta = 5

# Define the random variable based on that
f_Xi = laplace(loc=theta, scale=1)

# Set sample size (n) and number of bootstrap iterations (B)
n = 50
B = 100000

# Compute Cramér-Rao lower bound (CRLB)
CRLB = 1/n

# Define theta_hat (which uses all bootstrap samples as an input and calculates B theta_hats)
def theta_hats(X): return np.median(X, axis=0)

# Calculate variance of theta_hat across bootstrap samples
var_theta_hats = theta_hats(f_Xi.rvs(size=(n, B))).var()

# Print the CRLB, variance of the theta_hats, and percentage deviation
print('CRLB = ' + str(CRLB) + ', Var(theta_hat_MM) = ' + str(var_theta_hats) +
      r', % deviation: ' + str(np.abs(var_theta_hats - CRLB)/CRLB * 100))
