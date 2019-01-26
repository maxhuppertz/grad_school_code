################################################################################
### Econ 666, PS1Q3: Simulation exercise
################################################################################

# Import necessary packages
import multiprocessing as mp
import numpy as np
from os import chdir, path
from itertools import combinations, product
from linreg import ols
from joblib import Parallel, delayed
from scipy.misc import factorial as fac
from sympy.utilities.iterables import cartes, permutations, variations

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
        # Draw random variables as the basis for a random sample of units
        I = np.random.normal(size=T)

        # Go through all groups in the partition
        for i in range(nparts):
            # Get the number of people in the group
            ngroup = sum(P==i+1)

            # Figure out how many people to sample in this group (at least 2,
            # otherwise the treatment assignment by group will fail)
            nsamp = max(np.floor(ngroup * N/T),2)

            # Get the treatment indicator for the current group. Get the
            # rank within group from .argsort(), add +1 to get ranks
            # starting at 1, divide by the number of people in the
            # group, and assign everyone at or below the treatment
            # probability to treatment.
            I[P==i+1] = (I[P==i+1].argsort()  + 1) <= nsamp

        # The above mechanism could assign too few or too many units to
        # the sample. Calculate that discrepancy, as an integer.
        discrepancy = np.int(N - sum(I))

        # Check whether the discrepancy is positive
        if discrepancy >= 0:
            # If so, iterate over all 'missing' units
            for i in range(discrepancy):
                # Make a temporary vector containing all units not in the sample
                temp = I[I==0]

                # Pick a random integer index in that vector, and assign that
                # unit to the sample
                temp[np.random.randint(0,len(temp))] = 1

                # Replace the sample assignment vector with the temporary one,
                # which means one more unit has now been assigned to treatment
                # at random.
                I[I==0] = temp
        else:
            # If too many units were assigned, then the parameters for this
            # problem are badly set. Just print an error message.
            print(
            'Error: Between the number of tuples, the number of groups in ',
            'the partition, and the sample sizes, it is impossible to ',
            'assign at least two units from each group to the sample. ',
            'Please adjust the parameters. (This occured at N = ', N, '.)',
            sep=''
            )

        # Annoyingly, the data type of I will now be float. To be used as an
        # index, it has to be boolean or integer. I find it easiest to convert
        # it to boolean by just check where it isn't zero.
        I = (I != 0)

        # Make an intercept for this sample size
        beta0 = np.ones(shape=(N,1))

        # Go through all treatment probabilities
        for p in tprobs:
            # I'll need to know how many draws of treatment vectors would be
            # needed to get the exact randomization distribution for this
            # treatment probabilty and sample size. For now, just set that up
            # as 1.
            nrdexact = 1

            tau_hats = np.zeros(n_simul,)

            # Go through all simulations for the current set of parameters
            for s in range(nsimul):
                # Draw random variables as basis for treatment indicator
                W = np.random.normal(size=(N,1))

                # Go through all groups in the partition
                for i in range(nparts):
                    # Get number of people in the group n
                    ngroup = sum(P[I]==i+1)

                    # Get number of treated units k
                    ntreat = max(np.floor(p*ngroup),1)

                    # Get the treatment indicator for the current group. Get the
                    # rank within group from .argsort(), add +1 to get ranks
                    # starting at 1.
                    W[P[I]==i+1,0] = W[P[I]==i+1,0].argsort() + 1 <= ntreat

                    # Check whether this is the first group and the first
                    # simulation. If so, do the calculations required for the
                    # number of draws in the randomization distribution. It's
                    # convenient to do this now, since it saves some loops later
                    # on.
                    if s==0 and i == 0:
                        # If so, calculate n choose k for this group, and save
                        # the result
                        nrdexact = (
                            fac(ngroup) / (fac(ngroup-ntreat)*fac(ntreat))
                            )
                    elif s==0:
                        # If it's the first sumlation but not the first group,
                        # get n choose k, and multiply it by the number of
                        # possible assignments of all other groups calculated
                        # so far
                        nrdexact = (
                            nrdexact *
                            (fac(ngroup) / (fac(ngroup-ntreat)*fac(ntreat)))
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

                # Estimate the regression models
                beta_hat_simp, S_hat_simp = ols(Yobs,Z1)
                beta_hat_dumm, S_hat_dumm = ols(Yobs,Z2)
                beta_hat_satu, S_hat_satu = ols(Yobs,Z1)

            # Make sure this is an integer
            nrdexact = np.int(nrdexact)

            # Check whether the number of iterations required to get the exact
            # randomization distribution exceeds the maximum allowable number
            # of iterations
            if nrdexact <= nrdmax:
                # If so, set up an empty list
                A = []

                # Go through all groups in the partition
                for i in range(nparts):
                    # Get number of people in the group n
                    ngroup = sum(P[I]==i+1)

                    # Get number of treated units k
                    ntreat = np.int(max(np.floor(p*sum(P[I]==i+1)),1))
                    # If not, get all assignment vectors for this group,
                    # and store the Cartesian product of all assignment
                    # vectors for this group and all assignments calculated
                    # so far
                    #A = product(A,
                    #    (combinations(range(ngroup),ntreat)))
                    A.append(combinations(range(ngroup),ntreat))

                # Get the Cartesian product of the assignment vector of all
                # lists. Note that the asterisk matters, because that unpacks
                # A, which is a list of lists, before getting the product.
                # (Otherwise, this will just return the same three lists, since
                # A itself has size one: It is a single list of lists. So
                # without unpacking it first, product() gets the Cartesian
                # product of A with itself, which is just A.)
                A = product(*A)

                # Go through all possible assignment vectors
                for a in list(A):
                    # Set up treatment assignment as a vector of zeros
                    W = np.zeros(shape=(N,1))

                    # Go through all groups in the partition
                    for i in range(nparts):
                        # Get the assignment vector for that group
                        temp = W[P[I]==i+1]

                        # Replace is as one as appropriate
                        temp[a[i],0] = 1

                        # Replace the assignment vector
                        W[P[I]==i+1] = temp

                    # Generate observed outcome for this assignment
                    Yobs = Y0[I,:] + tau[I,:] * W

                    # Put together the RHS variables
                    Z1 = np.concatenate((beta0,W),axis=1)

                    # Run the regression
                    beta_hat_simp = ols(Yobs,Z1,get_cov=False)
            else:
                # If getting the exact randomization distribution is too
                # computationally intensive, go through the maximum number of
                # allowable iterations
                for s in range(nrdmax):
                    # Here, the treatment assignment is just as for the
                    # simulations above
                    # Draw random variables as basis for treatment indicator
                    W = np.random.normal(size=(N,1))

                    # Go through all groups in the partition
                    for i in range(nparts):
                        # Get number of people in the group n
                        ngroup = sum(P[I]==i+1)

                        # Get number of treated units k
                        ntreat = max(np.floor(p*ngroup),1)

                        # Get the treatment indicator for the current group.
                        # Get the rank within group from .argsort(), add +1 to
                        # get ranks starting at 1.
                        W[P[I]==i+1,0] = W[P[I]==i+1,0].argsort() + 1 <= ntreat

                    # Generate observed outcome for this assignment
                    Yobs = Y0[I,:] + tau[I,:] * W

                    # Put together the RHS variables
                    Z1 = np.concatenate((beta0,W),axis=1)

                    # Run the regression
                    beta_hat_simp = ols(Yobs,Z1,get_cov=False)

################################################################################
### Part 2: Run simulations
################################################################################

# Set seed
np.random.seed(666)

# Specify name for main directory (just uses the file's directory)
mdir = path.dirname(path.abspath(__file__)).replace('\\', '/')

# Change directory
chdir(mdir)

# Specify pairs of correlations
corrs = [[0,0], [.1,.1], [.6,.1], [.1,.6]]

# Specify number of tuples
T = 100

# Set sample sizes
sampsi = [10, 25, 100]

# Set treatment probabilities
tprobs = [.3, .5]

# Specify number of partitions for X
nparts = 3

# Specify number of simulations to run
nsimul = 100

# Tell the program how many estimations will be run for each simulation. (For
# example, if there is a simply regression of Y on a treament dummy, another
# regression of Y on a treatment dummy and partition indicators, and then a
# third regression for the saturated model, that's three estimations.)
nest = 3

# Specify maximum number of repetitions for randomization distribution
nrdmax = 10000

# Check how many cores are available
ncores = mp.cpu_count()

# Run simluations on all but one of the available cores in parallel
Parallel(n_jobs=ncores-1)(delayed(run_simulation)
    (corr, T=T, sampsi=sampsi, tprobs=tprobs, nparts=nparts, nsimul=nsimul,
    nrdmax=nrdmax) for corr in corrs)
