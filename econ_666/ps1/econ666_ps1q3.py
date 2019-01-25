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

# Set sample sizes
sampsi = [10, 25, 100]

# Set treatment probabilities
tprobs = [.3, .5]

# Specify number of partitions for X
nparts = 3

# Specify number of tuples
T = 100

# Make an intercept
beta0 = np.ones(shape=(T,1))

for corr in corrs:
    # Set up covariance matrix
    C = np.eye(len(corr)+1)

    # Fill in off-diagonal elements
    for i, c in enumerate(corr):
        C[0,i+1] = C[i+1,0] = c

    # Get data
    D = np.random.multivariate_normal(np.zeros(len(corr)+1), C, size=T)

    # Split them up into X, y, and tau
    X = D[:,0]
    y = D[:,1]
    tau = D[:,2:]

    # Get the partition of X. First, X.argsort() gets the ranks in the
    # distribution of X. Then, nparts/T converts it into fractions of the
    # length of X. Taking the ceil() makes sure that the groups are between 1
    # and nparts. The +1 is necessary because of Python's zero indexing, which
    # leads to the lowest rank being zero, and ceil(0) = 0 when it should be
    # equal to 1.
    P = np.ceil((X.argsort()+1)*nparts/T)

    # Set up a set of dummies for each section of the partition
    DP = np.array([P==i+1 for i in range(nparts-1)], ndmin=2).transpose()

    # Go through all sample sizes
    for N in sampsi:
        # Go through all treatment probabilities
        for p in tprobs:
            # Draw random variables as basis for treatment indicator
            W = np.random.normal(size=T)

            # Go through all groups in the partition
            for i in range(nparts):
                # Get the treatment indicator for the current group. Get the
                # rank within group from .argsort(), add +1 to get ranks
                # starting at 1, divide by the number of people in the group,
                # and assign everyone at or below the treatment probability
                # to treatment.
                W[P==i+1] = ((W[P==i+1].argsort()+1) / sum(P==i+1)) <= p
