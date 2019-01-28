################################################################################
### Econ 666, PS1Q3: Simulation exercise
################################################################################

# Import necessary packages and functions
import numpy as np
import pandas as pd
from itertools import combinations, product
from joblib import Parallel, delayed
from linreg import ols
from multiprocessing import cpu_count
from os import chdir, mkdir, path
from scipy.special import binom as binomial

################################################################################
### Part 1: Define necessary functions
################################################################################

# Define how to run the simulation for a given correlation pair
def run_simulation(corr, means, var_X, T, sampsis, tprobs, nparts, nsimul,
    nrdmax, dfdef=1, locdef=0, scaledef=1, cov_est = 'hc1', postau=1, nmod=3,
    cnum=0, prec=4, sups=True, mlw=100, getresults=False, tex=True,
    fnamepref='results_'):
    # Inputs
    # corr: 2-element tuple, specified correlation between X and Y0, and X and
    #       tau
    # means: 3-element vector, specified means for X, Y0, and tau
    # T: scalar, number of tuples in the simulated data
    # sampsis: vector, different sizes for random samples to draw
    # tprobs: vector, different treatment probabilities for each sample size
    # nparts: scalar, number of partitions on X
    # nsimul: scalar, number of simulations to run
    # nrdmax: scalar, maximum number of iterations to use for randomization
    #         distributions
    # dfdef: scalar, default degrees of freedom for chi2 distribution of Y0 if
    #        corr(X,Y0) = 0
    # locdef: scalar, default location parameter for Gumbel distribution of tau
    #         if corr(X,tau) = 0
    # scaledef: scalar, default scale parameter for Gumbel distribution of tau
    #         if corr(X,tau) = 0
    # cov_est: string, specifies the covariance estimator to use for the OLS
    #          estimation
    # postau: integer, position of the estimate of tau (the coefficient on the
    #         treatment dummy) in all models to be estimated
    # nmod: integer, number of models to be estimated
    # prec: integer, precision for floating point number printing in results
    # sups: boolean, if true, number which are too small to be printed using the
    #       selected printing precision will be printed as zero
    # mlw: integer, maximum line width for printing results
    # getresults: boolean, if true, the function returns the results as a pandas
    #             DataFrame (usually unnecessary, since it also prints them and
    #             can provide tex tables, see below)
    # tex: boolean, if true, saves results as tex tables
    # fnamepref: string, prefix for file names for tex tables (only matters if
    #            tex is true)
    #
    # Outputs
    # results: DataFrame, contains the results
    #
    # Note
    # To generate the three variables I need, I start with X as a normally
    # distributed random variable. Then, I generate the other two variables
    # based on that. Let Z denote any of them. I want to achieve
    #
    # Corr(X,Z) = Cov(X,Z) / sqrt(Var(X) Var(Z)) = gamma                     (1)
    #
    # for some gamma. I can generate
    #
    # Z = alpha + X + Z_eps                                                  (2)
    #
    # where Z_eps is an error term, if you will. From standard linear regression
    # this implies Cov(X,V) / Var(X) = 1. Also, taking the variance of (2), I
    # have Var(Z) = Var(X) + Var(Z_eps). Plugging both of these into (1),
    #
    # Var(Z_eps) = Var(X) * (gamma^(-2) - 1)
    #
    # and since I get to choose Var(Z_eps), I can thereby generate random
    # variables with arbitrary correlation structure. I can then use alpha to
    # adjust the mean of the generated variable.

    # Set seed (since this will be run in parallel, it's actually important to
    # set the seed within the function, rather than outside)
    np.random.seed(666+cnum)

    # Generate X as a normally distributed random variable
    X = np.random.normal(means[0], np.sqrt(var_X), size=(T,1))

    # Let Y0_eps have a chi2 distribution
    if corr[0] != 0:
        # Calculate the necessary variance
        var_Y0 = var_X*(corr[0]**(-2)-1)

        # Calculate the degrees of freedom implied by this variance (this comes
        # from the fact that for a chi2(k) random variable, its variance is
        # equal to 2k)
        df_Y0 = .5*var_Y0

        # Calculate Y0, where I need to make sure to correct for the mean of
        # the error term (the mean of a chi2(k) is simply k)
        Y0 = means[1] - df_Y0 + X + np.random.chisquare(df_Y0,size=(T,1))
    else:
        # In the case without correlation between X and Y0, just make sure to
        # get the mean right, and choose a chi2(1) error term
        Y0 = means[1] - dfdef + np.random.chisquare(dfdef,size=(T,1))

    # Let tau_eps have a Gumbel distribution
    if corr[1] != 0:
        # Calculate the necessary variance
        var_tau = var_X*(corr[1]**(-2)-1)

        # Calculate the implied scale for the Gumbel distribution (a
        # Gumbel(0,b) random variable has variance b^2 (pi^2/6))
        beta_tau = np.sqrt( (6/(np.pi**2)) * var_tau )

        # Calculate tau, correcting for the fact that a Gumbel(0,b) random
        # variable has mean gb, where g is the Euler-Mascheroni constant)
        tau = (means[2] - np.euler_gamma*beta_tau + X +
            np.random.gumbel(0,beta_tau,size=(T,1)))
    else:
        # In the case of no correlation between X and tau, just make sure to get
        # the mean right, and use a Gumbel(0,1) error term
        tau = ( means[2] - np.euler_gamma*scaledef +
        np.random.gumbel(locdef,scaledef,size=(T,1)) )

    # Get the partition of X. First, X[:,0].argsort() gets the ranks in the
    # distribution of X. Then, nparts/T converts it into fractions of the
    # length of X. Taking the ceil() makes sure that the groups are between 1
    # and nparts. The +1 is necessary because of Python's zero indexing, which
    # leads to the lowest rank being zero, and ceil(0) = 0 when it should be
    # equal to 1.
    P = np.ceil((X[:,0].argsort()+1)*nparts/T)

    # Set up a set of dummies for each but one group in the partition. Since P
    # is a list, each of the checks creates a list of ones and zeros which
    # indicate whether an element of P is equal to the current i. When
    # np.array() is applied to this list of lists, it stacks them as rows of a
    # matrix. This creates an nparts - 1 by T matrix of indicator dummies. The
    # transpose converts it into a more conventional format. The last group in
    # the partition is omitted.
    D = np.array([P==i+1 for i in range(nparts-1)], ndmin=2).transpose()

    # Make a vector to store the mean treatment effect estimates and mean
    # standard errors. This needs one row for each sample size and each
    # treatment probability, two columns for each estimation, two columns for
    # the true tau and its standard deviations, and an extra two columns for
    # the sample size and treatment probability. (That makes it easiert to
    # print the results later.)
    tau_hats_avg = np.zeros(shape=(len(sampsis)*len(tprobs),4+nest*2))

    # Go through all sample sizes
    for nsampsi, N in enumerate(sampsis):
        # Record sample size indicator in the mean estimate array
        tau_hats_avg[nsampsi*2:nsampsi*2+2,0] = N

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
        for nprob, p in enumerate(tprobs):
            # Record treatment probability in the mean estimate array
            tau_hats_avg[nsampsi*2+nprob,1] = p

            # I'll need to know how many draws of treatment vectors would be
            # needed to get the exact randomization distribution for this
            # treatment probabilty and sample size. For now, just set that up
            # as 1.
            nrdexact = 1

            # Set up an empty array to store the estimated tau_hat and its
            # standard error for each of the three models for each of the
            # simulations. (Each row is a given simulation, and each two
            # columns are for a given tau_hat and its standard error.)
            tau_hats = np.zeros(shape=(nsimul,nest*2))

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
                        # the result. (I originally did this by hand using
                        # factorials, but using this has the nice side effect
                        # of being evaluated as np.inf (positive infinity) in
                        # case this runs into overflow issues, whereas my code
                        # would result in a NaN, which I would then manually
                        # have to change into an Inf.)
                        nrdexact = binomial(ngroup,ntreat)
                    elif s==0:
                        # If it's the first sumlation but not the first group,
                        # get n choose k, and multiply it by the number of
                        # possible assignments of all other groups calculated
                        # so far
                        nrdexact = nrdexact * binomial(ngroup,ntreat)

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

                # Estimate the first two regression models and store the
                # estimates in the tau_hats array, in row s
                for i, Z in enumerate([Z1, Z2]):
                    # Estimate the model
                    beta_hat, S_hat = ols(Yobs,Z,cov_est=cov_est)

                    # Store the estimates. The row index is easy. For the column
                    # index, it's important to remember Python's zero indexing,
                    # and how it assigns elements to indices. This maps counter
                    # i to index [j,k] as
                    #
                    # 0 -> [0,2], 1 -> [2,4], ...
                    #
                    # and for any given index [j,k], Python will try to assign
                    # contents to the elements j,j+1,...,k-1, but not to k
                    # itself. Therefore, this gets me the right indices for a
                    # two element assignment.
                    tau_hats[s,2*i:2*i+2] = (
                        beta_hat[postau,0], np.sqrt(S_hat[postau,postau])
                        )

                # For the saturated model, I need to get the average treatment
                # effect. First, estimate the model.
                beta_hat, S_hat = ols(Yobs,Z3,cov_est=cov_est)

                # Set up a vector of linear constraints on tau
                L = np.zeros(shape=(beta_hat.shape))

                # Replace the element corresponding to the base effect as one,
                # since every group in the partition has this as part of their
                # estimated effect
                L[postau,0] = 1

                # Go through all groups in the partition for which there are
                # dummies in D
                for i in range(nparts-1):
                    # Get the number of treated units
                    ntreat = sum(W[:,0])

                    # Get the number of treated units in this group
                    ntreatgroup = sum((P[I]==i+1) * (W[:,0]==1))

                    # Replace the corresponding element of L with the
                    # probability of being in this group, conditional on being
                    # a treated unit. The position of that element is equal to
                    # the length of beta_hat minus the number of groups in the
                    # partition minus one plus the number of the group under
                    # consideration. That is
                    #
                    # beta_hat.shape[0]-(nparts-1)+i
                    # = beta_hat.shape[0]-nparts+i+1
                    #
                    # remembering that due to Python's zero indexing, the number
                    # of the group is i+1, not i.
                    L[beta_hat.shape[0]-nparts+i+1,0] = ntreatgroup/ntreat

                # Calculate the average treatment effect for the saturated
                # model
                tau_hat_avg_satu = L.transpose() @ beta_hat

                # Calculate the estimated variance
                S_hat_satu = L.transpose() @ S_hat @ L

                # Store the estimate and its standard error
                tau_hats[s,2*(nmod-1):] = (
                    tau_hat_avg_satu, np.sqrt(S_hat_satu)
                    )

            # Store the average estimates and standard errors for all three
            # models, for the current sample size and treatment probability
            tau_hats_avg[nsampsi*2+nprob,4:] = np.mean(tau_hats, axis=0)

            # Set up an array to store the randomization distribution of tau_hat
            # (or the maximum number of simulation draws used to approximate it,
            # if getting the exact distribution is not feasible.)
            tau_true = np.zeros(shape=(np.int(np.minimum(nrdexact, nrdmax)),1))

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

                    # Get all assignment vectors for this group, and add them
                    # to the list
                    A.append(combinations(range(ngroup),ntreat))

                # Get the Cartesian product of the assignment vectors for all
                # groups. Note that the asterisk matters, because that unpacks
                # A, which is a list of lists, before getting the product.
                # (Otherwise, this will just return the same three lists, since
                # A itself has size one: It is a single list of lists. So
                # without unpacking it first, product() gets the Cartesian
                # product of A with itself, which is just A.)
                A = product(*A)

                # Go through all possible assignment vectors
                for s, a in enumerate(list(A)):

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

                    # Store the result
                    tau_true[s,0] = beta_hat_simp[1,0]
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

                    # Store the result
                    tau_true[s,0] = beta_hat_simp[1,0]

            # Store the expected value of tau
            tau_hats_avg[nsampsi*2+nprob,2] = np.mean(tau_true, axis=0)

            # Store the standard deviation of tau
            tau_hats_avg[nsampsi*2+nprob,3] = np.std(tau_true, axis=0)

    # Set display options (has to be done within each function if this runs in
    # parallel)
    pd.set_option('display.max_columns', tau_hats_avg.shape[1])
    pd.set_option('display.width', mlw)
    pd.set_option('display.precision', prec)

    # Make a header line for the results, starting with the basic parameters
    firstline = ['N', 'p', 'tau', 'SD']

    # Use Python's amazing list comprehension to make a list that goes,
    # [tau_hat 1, SE 1, tau_hat 2, SE 2, ...]
    firstline.extend(
        x for i in range(nest) for x in ['tau_hat '+str(i+1), 'SE '+str(i+1)])

    # Put the results in a pandas DataFrame
    results = pd.DataFrame(data=tau_hats_avg, columns=firstline)

    # Make sure sample sizes are stored as integers
    results['N'] = results['N'].astype(int)

    # Print the results
    print('Correlations: corr(X,Y0) = ', corr[0], ', corr(X,tau) = ', corr[1],
        '\n', results, '\n', sep='')

    # Check whether to export to latex
    if tex:
        # Save the results as a tex table
        results.to_latex(fnamepref+str(cnum)+'.tex', index=False)

    # If desired, return results DataFrame
    if getresults:
        return results

################################################################################
### Part 2: Run simulations
################################################################################

# Specify name for main directory (just uses the file's directory)
mdir = path.dirname(path.abspath(__file__)).replace('\\', '/')

# Set figures/tables directory (doesn't need to exist)
fdir = '/figures'

# Create it if it doesn't exist
if not path.isdir(mdir+fdir):
    mkdir(mdir+fdir)

# Change directory to figures
chdir(mdir+fdir)

# Specify pairs of correlations, in the order [corr(X,Y0), corr(X,tau)]
corrs = [[.0,.0], [.1,.1], [.6,.1], [.1,.6]]

# Specify means for the three variables, in the order
# [mean_X, mean_Y0, mean_tau]
means = [0, 0, .2]

# Specify variance of X
var_X = 1

# Specify number of tuples
T = 100

# Set sample sizes
sampsis = [10, 25, 100]

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

# Specify how many cores to use for parallel processing
ncores = cpu_count()

# Run simluations on all but one of the available cores in parallel
Parallel(n_jobs=ncores)(delayed(run_simulation)
    (corr=corr, means=means, var_X=var_X,T=T, sampsis=sampsis, tprobs=tprobs,
    nparts=nparts, nsimul=nsimul, nrdmax=nrdmax, cnum=cnum) for cnum, corr in
    enumerate(corrs))
