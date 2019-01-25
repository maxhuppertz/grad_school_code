import numpy as np
from os import chdir, path
from linreg import ols
from scipy.stats.stats import pearsonr

# Specify name for main directory (just uses the file's directory)
mdir = path.dirname(path.abspath(__file__)).replace('\\', '/')

# Change directory
chdir(mdir)

# Specify pairs of correlations
corrs = [[0,0], [.1,.1], [.6,.1], [.1,.6]]

# Specify number of tuples
T = 100

for corr in corrs:
    # Set up covariance matrix
    C = np.eye(len(corr)+1)

    for i, c in enumerate(corr):
        C[0,i+1] = C[i+1,0] = c

    D = np.random.multivariate_normal(np.zeros(len(corr)+1), C, size=T)
