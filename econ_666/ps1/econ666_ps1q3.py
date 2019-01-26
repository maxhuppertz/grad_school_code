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
            # Get the treatment indicator for the current group. Get the
            # rank within group from .argsort(), add +1 to get ranks
            # starting at 1, divide by the number of people in the
            # group, and assign everyone at or below the treatment
            # probability to treatment.
            I[P==i+1] = (
                ((I[P==i+1].argsort()  + 1) / sum(P==i+1))
                <= N/T
                )

        # The above mechanism could assign too few or too many units to
        # the sample. This loop iterates over the number of such units, no
        # matter whether there are too many or too few.
        for i, excess in enumerate(range(np.int(np.abs(N - sum(I))))):
            if N > sum(I):
                # If there are too few units assigned, randomly pick a unit
                # and add it to the sample. First, make a temporary vector
                # containing all units not in the sample
                temp = I[I==0]

                # Then, pick a random integer index in that vector, and assign
                # that unit to the sample
                temp[np.random.randint(0,len(temp))] = 1

                # Replace the sample assignment vector with the temporary one,
                # which means one more unit has now been assigned to treatment
                # at random.
                I[I==0] = temp
            else:
                # If there are too many units assigned, remove one unit at
                # random, but be sure to do it group by group. (The first
                # iteration deletes one unit from the first group at random,
                # the second iteration from the second group, the third from the
                # third group, and then back to the first, etc., although there
                # really shouldn't be that many excess assignments.)
                temp = I[(I==0) and (P[I==0]==i+1-np.floor(i/nparts))]
                temp[np.random.randint(0,len(temp))] = 1
                I[(I==0) and (P[I==0]==i+1-np.floor(i/nparts))] = 0

        # Annoyingly, the data type of I will now be float. To be used as an
        # index, it has to be boolean or integer. I find it easiest to convert
        # it to boolean by just check where it isn't zero.
        I = (I != 0)

        # Make an intercept for this sample size
        beta0 = np.ones(shape=(N,1))

        # Go through all treatment probabilities
        for p in tprobs:
            # Go through all simulations for the current set of parameters
            for s in range(nsimul):
                # Draw random variables as basis for treatment indicator
                W = np.random.normal(size=(N,1))

                # Go through all groups in the partition
                for i in range(nparts):
                    # Get the treatment indicator for the current group. Get the
                    # rank within group from .argsort(), add +1 to get ranks
                    # starting at 1, divide by the number of people in the
                    # group, and assign everyone at or below the treatment
                    # probability to treatment.
                    W[P[I]==i+1,0] = (
                        ((W[P[I]==i+1,0].argsort() + 1) / sum(P[I]==i+1))
                        <= p
                        )

                    # For very small sample sizes and small treatment
                    # probabilities, this can result in no units being assigned
                    # to treatment. In that case, assign at least one.
                    if sum(W[P[I]==i+1,0]) == 0:
                        temp = W[P[I]==i+1,0]
                        temp[np.random.randint(0,len(temp))] = 1
                        W[P[I]==i+1,0] = temp

                    # Obviously, it will very rarely happen that the number
                    # of units assigned to treatment from this group divided
                    # by the number of units in the group will be exactly
                    # equal to p. This statement checks whether p is larger or
                    # smaller than the ratio of assigned units in this group,
                    # which is sum(W[P[I]==i+1,0]), to the number of units in
                    # the group, which is sum(P[I]==i+1).
                    if ( (p > sum(W[P[I]==i+1,0])/sum(P[I]==i+1))
                    and (sum(W[P[I]==i+1,0]) < sum(P[I]==i+1)-1) ):
                        # If p is larger, and if assigning one more unit to
                        # treatment wouldn't mean the whole group is assigned,
                        # I arrive here.
                        # Get the treatment assignment vector for the group, put
                        # it in a temporary object
                        temp1 = W[P[I]==i+1,0]

                        # Make another temporary object, which gets only units
                        # which weren't assigned to treatment
                        temp2 = temp1[W[P[I]==i+1,0] == 0]

                        # Pick one of those units at random using randint().
                        # Replace the treatment indicator with a Bernoulli
                        # trial (Binomial distribution with only 1 trial,which
                        # is the first argument in binomial()) that succeeds
                        # with a probability which is equal to the distance
                        # between p and the assigned number of units in the
                        # group, divided by the size of the group. In
                        # expectation, the number of units assigned will now be
                        # exactly p.
                        temp2[np.random.randint(0,len(temp2))] = (
                            np.random.binomial(
                                1,p*sum(P[I]==i+1)-sum(W[P[I]==i+1,0]))
                            )

                        # Replace the temporary version of the group's treatment
                        # indicator with the new version including the
                        # potentially changed assignment
                        temp1[W[P[I]==i+1,0] == 0] = temp2

                        # Replace the actual treatment vector for the group
                        W[P[I]==i+1,0] = temp1
                    elif ( (p < sum(W[P[I]==i+1,0])/sum(P[I]==i+1))
                    and (sum(W[P[I]==i+1,0]) > 1) ):
                        # If p is lower, and removing one unit doesn't assign
                        # everyone to the control group, I will end up here.
                        # Again, get the treatment assignement vector for this
                        # group.
                        temp1 = W[P[I]==i+1,0]

                        # Get only units assigned to the treatment group
                        temp2 = temp1[W[P[I]==i+1,0] == 1]

                        # Pick a random unit and replace their treatment status
                        # as zero using another Bernoulli trial. The probability
                        # of success is equal to the distance between the number
                        # of units assigned to treatment in the group divided
                        # by group size and p, divided by 1 over the group
                        # size.
                        temp2[np.random.randint(0,len(temp2))] = (
                            np.random.binomial(
                                1,sum(W[P[I]==i+1,0])-p*sum(P[I]==i+1))
                            )

                        # Replace the temporary version of the treatment vector
                        # with the new one
                        temp1[W[P[I]==i+1,0] == 1] = temp2

                        # Replace the actual treatment assignment vector for the
                        # group
                        W[P[I]==i+1,0] = temp1

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

            # Calculate how many draws would be needed to get the exact
            # randomization distribution, given how many units are assigned to
            # treatment for the given sample size.
            nrdexact = 1

            # GO through all groups in the partition
            for i in range(nparts):
                # Get number of people in the group n
                ngroup = sum(P[I]==i+1)

                # Get number of treated units k
                ntreat = max(np.floor(p*sum(P[I]==i+1)),1)

                # Check whether this is the first group
                if i+1 == 1:
                    # If this is the first group, calculate n choose k for this
                    # group, and save the result
                    nrdexact = fac(ngroup) / (fac(ngroup-ntreat)*fac(ntreat))
                else:
                    # If this isn't the first group, calculate n choose k for
                    # this group, multiply it by the number of possible
                    # assignments of all other groups calculated so far
                    nrdexact = (
                        nrdexact *
                        (fac(ngroup) / (fac(ngroup-ntreat)*fac(ntreat)))
                        )

            # Make sure this is an integer
            nrdexact = np.int(nrdexact)

            # Check whether the number of iterations required to get the exact
            # randomization distribution exceeds the maximum allowable number
            # of iterations
            A = []
            if nrdexact <= nrdmax:
                # Go through all groups in the partition
                for i in range(nparts):
                    # Get number of people in the group n
                    ngroup = sum(P[I]==i+1)

                    # Get number of treated units k
                    ntreat = np.int(max(np.floor(p*sum(P[I]==i+1)),1))

                    # Check whether this is the first group
                    if i+1 == 1 and 0==1:
                        # If so, get all assignment vectors for this group
                        # and store them. (The combinations() function doesn't
                        # produce whole vectors. Rather, it produces indices.)
                        A = combinations(range(ngroup),ntreat)
                    else:
                        # If not, get all assignment vectors for this group,
                        # and store the Cartesian product of all assignment
                        # vectors for this group and all assignments calculated
                        # so far
                        #A = product(A,
                        #    (combinations(range(ngroup),ntreat)))
                        A.append(combinations(range(ngroup),ntreat))
                A = product(*A)

                for a in list(A):
                    W = np.zeros(shape=(N,1))

                    for i in range(nparts):
                        temp = W[P[I]==i+1]
                        temp[a[i],0] = 1
                        W[P[I]==i+1] = temp

                    Z1 = np.concatenate((beta0,W),axis=1)
                    beta_hat_simp, S_hat_simp = ols(Yobs,Z1)
            else:
                for i in range(nrdmax):
                    pass

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
