################################################################################
### Econ 666, PS1Q3: Simulation exercise
################################################################################

# Import necessary packages
import multiprocessing as mp
import numpy as np
from os import chdir, path
from linreg import ols
from joblib import Parallel, delayed

################################################################################
### Part 1: Define necessary functions
################################################################################

# Define how to run the simulation for a given correlation pair
def run_simulation(corr, T, sampsi, tprobs, nparts, nsimul, nrdmax):
    # Set up covariance matrix
    C = np.eye(len(corr)+1)

    # Fill in off-diagonal elements
    for i, c in enumerate(corr):
        C[0,i+1] = C[i+1,0] = c

    # Get data
    D = np.random.multivariate_normal(np.zeros(len(corr)+1), C, size=T)

    # Split them up into X, Y0, and tau. I'd like these to be Numpy arrays, i.e.
    # (for all practical purposes) vectors. By default, np.array() likes to
    # create row vectors, which I find unintuitive. The tranpose() changes these
    # into column vectors.
    X = np.array(D[:,0], ndmin=2).transpose()
    Y0 = np.array(D[:,1], ndmin=2).transpose()
    tau = np.array(D[:,len(corr)], ndmin=2).transpose()

    # Get the partition of X. First, X[:,0].argsort() gets the ranks in the
    # distribution of X. Then, nparts/T converts it into fractions of the
    # length of X. Taking the ceil() makes sure that the groups are between 1
    # and nparts. The +1 is necessary because of Python's zero indexing, which
    # leads to the lowest rank being zero, and ceil(0) = 0 when it should be
    # equal to 1.
    P = np.ceil((X[:,0].argsort()+1)*nparts/T)

    # Set up a set of dummies for each section of the partition. Since P is a
    # list, each of the checks creates a list of ones and zeros which indicate
    # whether an element of P is equal to the current i. When np.array() is
    # applied to this list of lists, it stacks them as rows of a matrix. This
    # creates an nparts - 1 by T matrix of indicator dummies. The transpose
    # converts it into a more conventional format
    D = np.array([P==i+1 for i in range(nparts-1)], ndmin=2).transpose()

    # Make a vector to store the estimated treatment effects tau_hat

    # Go through all sample sizes
    for N in sampsi:
        # Draw a random sample of units
        I = np.random.normal(size=T).argsort() + 1 <= N

        # Make an intercept for this sample size
        beta0 = np.ones(shape=(N,1))

        # Go through all treatment probabilities
        for p in tprobs:
            # Draw random variables as basis for treatment indicator
            W = np.random.normal(size=(N,1))

            # Go through all groups in the partition
            for i in range(nparts):
                # Get the treatment indicator for the current group. Get the
                # rank within group from .argsort(), add +1 to get ranks
                # starting at 1, divide by the number of people in the group,
                # and assign everyone at or below the treatment probability
                # to treatment.
                W[P[I]==i+1,0] = (
                    ((W[P[I]==i+1,0].argsort() + 1) / sum(P[I]==i+1))
                    <= p
                    )

            # Generate observed outcome for the simulation regressions
            Yobs = Y0[I,:] + tau[I,:] * W

            # Generate RHS data sets for the simulation regressions
            # The first data set is just an intercept and a treatment dummy
            Z1 = np.concatenate((beta0,W),axis=1)

            # The first data set also includes the partition dummies
            Z2 = np.concatenate((beta0,W,D[I,:]),axis=1)

            # The third data set also includes an interaction between the
            # treatment dummy and the partition dummies
            Z3 = np.concatenate(
                (beta0,W,D[I,:],(W @ np.ones(shape=(1,nparts-1))) * D[I,:]),
                axis=1)

            # Go through all simulations for the current set of parameters
            for i in range(nsimul):
                beta_hat_simp, S_hat_simp = ols(Yobs,Z1)
                beta_hat_dumm, S_hat_dumm = ols(Yobs,Z2)
                beta_hat_satu, S_hat_satu = ols(Yobs,Z1)

################################################################################
### Part 2: Run simulations
################################################################################

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

# Specify number of simulations to run
nsimul = 100

# Specify maximum number of repetitions for randomization distribution
nrdmax = 10000

# Specify number of tuples
T = 100

# Run simulations for all correlation pairs
for corr in corrs:
    run_simulation(corr, T=T, sampsi=sampsi, tprobs=tprobs, nparts=nparts,
        nsimul=nsimul, nrdmax=nrdmax)

# Run simluations on all available cores in parallel
#Parallel(n_jobs=mp.cpu_count())(delayed(run_simulation)
#    (corr, T=T, sampsi=sampsi, tprobs=tprobs, nparts=nparts, nsimul=nsimul,
#    nrdmax=nrdmax, beta0=beta0)
#    for corr in corrs)
