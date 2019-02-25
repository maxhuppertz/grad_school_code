################################################################################
### Econ 666, PS2Q1: Multiple testing
### Recreates some results from Ashraf, Field, and Lee (2014)
### Then uses multiple testing corrections to adjust p-values
################################################################################

# Import necessary packages and functions
import io
import numpy as np
import pandas as pd
import re
import requests
from inspect import getsourcefile
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from os import chdir, mkdir, path
from shutil import copyfile, rmtree
from zipfile import ZipFile

# Specify name for main directory (just uses the file's directory)
# I used to use path.abspath(__file__), but apparently, it may be a better idea
# to use getsourcefile() instead of __file__ to make sure this runs on
# different OSs. I just give it an object, and it checks which file defined it.
# But since the object I give it is an inline function lambda, which was
# created in this file, it points to this file
mdir = path.dirname(path.abspath(getsourcefile(lambda:0))).replace('\\', '/')

# Change to main directory
chdir(mdir)

# Import custom packages (have to be in the main directory)
from linreg import ols

################################################################################
### Part 1: Define necessary functions
################################################################################

# Define a function to do the Bonferroni correction
def bonferroni(p):
    # Inputs
    # p: [M,k] matrix, p-values for the original hypotheses
    #
    # Outputs
    # p_bc: [M,k] matrix, Bonferroni-adjusted p-values

    # Get number of members in the family M, and number of parameters of
    # interest k
    M, k = p.shape

    # Calculate Bonferroni corrected p-values
    p_bc = np.minimum(p * (M*k), 1)

    # Return them
    return p_bc

# Define a function to get Holm-Bonferroni adjusted p-values
def holm_bonferroni(p, alpha=.05, order='F'):
    # Inputs
    # p: [M,k] matrix, p-values for the original hypotheses
    # alpha: scalar, level of test used
    # order: string, one of 'C', 'F', or 'A', specificies how to flatten and
    #        reshape the input p-values (does not really matter)
    #
    # Outputs
    # p_hb: [M,k] matrix, Holm-Bonferroni adjusted p-values

    # Get original dimensions of p-values, which might be provided as a matrix
    M, k = p.shape

    # Flatten the array of p-values, in case a matrix is provided. The order
    # argument is important only to ensure that this is put back into place the
    # same way later. Which order is chosen does not matter.
    p = p.flatten(order=order)

    # Get indices of sorted p-values
    p_sorted_index = p.argsort()

    # Sort the p-values, make them into a proper (column) vector
    p = np.array(p[p_sorted_index], ndmin=2).transpose()

    # Set up array of adjusted p-values
    p_hb = p * np.array([M*k-s for s in range(M*k)], ndmin=2).transpose()

    # Go through all p-values but the first and enforce monotonicity
    for i in range(len(p_hb[1:])):
        # Replaces the current adjusted p-value with the preceding one if that
        # is larger, and enforces p <= 1
        p_hb[i+1] = np.minimum(np.maximum(p_hb[i], p_hb[i+1]), 1)

    # Now, put the p-values back in the original order. Set up an array of zeros
    # of the same length as p_hb. (This will also be a column vector.)
    p_hbo = np.zeros(shape=p_hb.shape)

    # Put the adjusted p-values back in the same order as the original. To do
    # that, go through the sorting indices. The first index is the original
    # position of the smallest p-value, so put that back where it came from. The
    # second index is the original position of the second smallest p-value, so
    # put that where it came from. And so on.
    for sorti, origi in enumerate(p_sorted_index): p_hbo[origi] = p_hb[sorti]

    # Put the ordered adjusted p-values back in the same shape as the input
    # p-values
    p_hbo = np.reshape(p_hbo, newshape=(M,k), order=order)

    # Return the adjusted p-values
    return p_hbo

# Define a function to do one interation of the free step down resampling
# algorithm (that is, one treatment reassignment plus calculating the
# corresponding p-values)
def permute_p(Y, Isamp, ntreat, balvars, prank, X=None, Z=None, seed=1,
    Breg=10, breg_icept=True, cov_est='hmsd', order='F', shape=None):
    # Inputs
    # Y: [N,M] matrix, data for each of the M outcomes in the family
    # Isamp: [N,1] vector, estimation sample to use. (If treatment assignment is
    #        at a higher level than the estimation, e.g. because the estimatiion
    #        is for a follow-up sample, this function will reassign treatment at
    #        the higher level, but use only the subsample that survives to
    #        follow-up for any estimations.)
    # ntreat: scalar, number of treated units
    # balvars: [N,B] matrix, data for each of the B balancing variables
    # prank: [M,1] list-like (one dimensional), original order of the p-values,
    #        from smallest to largest
    # X: [N,D] matrix, data for covariates to include when estimating
    #    regressions, but without saving their p-values
    # Z: [N,E] matrix, data for covariates of interest, will be included in the
    #    estimations, and their p-values will be recorded
    # seed: scalar, random number generator's seed
    # Breg: scalar, number of balancing regressions to use
    # breg_icept: boolean, if true, balancing regressions will include an
    #             intercept
    # cov_est: string, covariance estimator to use (see ols() in linreg.py for
    #          available options)
    # order: string, one of 'C', 'F', or 'A', specificies how to flatten and
    #        reshape the input p-values (does not really matter)
    # shape: tuple, shape into which to reshape the p-values before outputting
    #
    # Outputs
    # p_star: vector or matrix, shape depends on whether Z is included, and
    #         whether shape was specified,  permutation p-values for one
    #         iteration of the free step-down randomization

    # Set random number generator's seed
    np.random.seed(seed)

    # Get total sample size N and number of outcome variables M
    N, M = Y.shape

    # Figure out the index of the parameter of interest
    if X is None and Z is None:
        # If no other RHS variables are being used, it's just the first element
        # of the estimates vector
        cidx = [0]
    elif X is not None and Z is None:
        # If RHS variables are being inserted before it, figure out how many,
        # and use the next index
        cidx = [X.shape[1]]
    else:
        # If RHS variables of interest are included after it, include their
        # indices as well
        if X is none:
            # If X wasn't specified, just make a [0,0] array for it
            X = np.empty(shape=(0,0))

        # Get the coefficient indices
        cidx = [X.shape[1]] + [z + X.shape[1] + 1 for z in range(Z.shape[1])]

    # Get number of tests
    T = M * len(cidx)

    # Set up vector of treatment assignments for this iteration
    W = np.zeros(shape=(N,1))

    # Set up a place to save the smallest maximum t-statistic recorded so far
    tmax = np.inf

    # Go through all balancing regressions
    for b in range(Breg):
        # Get new treatment assignment, by drawing randomly from a standard
        # normal distribution, getting the rank (adjusting by +1 to account for
        # Python's zero indexing), and assigning everyone with a rank equal to
        # or below the number of treated units to treatment
        Wb = np.random.normal(size=(N,1))
        Wb = np.array((Wb[:,0].argsort() + 1 <= ntreat), ndmin=2).transpose()

        # Set up vector of t-statistics for this treatment assignment
        t = np.zeros(shape=(balvars.shape[1],1))

        # Go through all balancing variables
        for i in range(balvars.shape[1]):
            # Get indices of non-missing observations for that variable
            I = ~np.isnan(balvars[:,i])

            # Make an array of balancing variables
            Xbv = np.array(balvars[I,i], ndmin=2).transpose()

            # Check whether an intercept needs to be added
            if breg_icept:
                # If so, combine the balancing variable with an intercept
                Xbv = (
                    np.concatenate((np.ones(shape=(np.sum(I),1)), Xbv),
                        axis=1)
                    )

            # Run OLS of treatment assignemnt on balancing variable, get the
            # t-statistic
            _, _, tb, _ = ols(Wb[I,:], Xbv, cov_est=cov_est)

            # Allocate correct t-statistic to the vector of t-statistics
            if breg_icept:
                # If there is an intercept, use the second one
                t[i,0] = tb[1,0]
            else:
                # Otherwise, there is only one to use
                t[i,0] = tb[0,0]

        # Check whether the largest absolute value of any t-statistic across all
        # balancing regressions is less than the maximum recorded so far. Note
        # that np.minimum() and np.maximum() take exactly two arrays as inputs,
        # and compute the element-wise min or max. Things get funky once they
        # compare inputs of two different shapes. To get the min or max of the
        # elements of a single array, I have to use np.amax() or np.amin()
        # instead.
        if np.amax(np.abs(t[:,0])) <= tmax:
            # If so, save the new minmax t-statistic
            tmax = np.amax(np.abs(t[:,0]))

            # Save the treatment assignment
            W = Wb

    # Set up a vector of p-values, one for each outcome variable and coefficient
    # of interest
    pstar = np.zeros(shape=(T,1))

    # Go through all members in the family
    for i in range(T):
        # Make an index of where both the member and all parts of X are not NaN,
        # and only get units in the estimation sample, i.e. where Isamp == 1
        I = (~np.isnan(Y[:,i]) & ~np.isnan(X.sum(axis=1)) & Isamp)

        # Get the number of effective observations
        n = I.sum()

        # Get outcome variable, i.e. the current member of the family, for those
        # observations with non-missing data for both LHS and RHS variables
        y = np.array(Y[I,i], ndmin=2).transpose()

        # Make a matrix of RHS variables. If X was specified, add it to the
        # treatment assignment
        if X is not None:
            Xstar = np.concatenate((X[I], W[I]), axis=1)
        else:
            Xstar = W[I]

        # If Z was specified, add it in after the treatment assignment
        if Z is not None:
            Xstar = np.concatenate((Xstar, Z[I]), axis=1)

        # Run OLS
        _, _, _, p = ols(y, Xstar, cov_est=cov_est)

        # Save only p-values of interest
        pstar[i,:] = p[cidx]
    # Reorder p-values in the original order (lowest to highest)
    pstar_reord = pstar[prank]

    # Got through all p-values
    for i in range(T):
        # Replace them as the minimum across all originally larger p-values
        pstar_reord[i,:] = np.amin(pstar_reord[i:,:])

    # Put p-values back into the order of hypotheses being tested
    for sorti, origi in enumerate(prank): pstar[origi] = pstar_reord[sorti]

    # Reshape the result if desired
    if shape is not None:
        pstar = pstar.reshape(order=order)

    # Return the adjusted p-values
    return pstar

# Define a function to run the Benjamini-Hochberg FDR control for a set of
# p-values
def benjamini_hochberg(p, tol=10**(-6), order='F'):
    # Save the shape of the input array of p-values
    [M,k] = p.shape

    # Flatten the p-values
    p = p.flatten(order=order)

    # Get indices of sorted p-values
    p_sorted_index = p.argsort()

    # Sort the p-values
    p = p[p_sorted_index]

    # Get number of tests T
    T = M*k

    # Make vectors of upper and lower boundaries for p-values
    qlo = np.zeros(shape=(T,1))
    qhi = np.ones(shape=(T,1))

    # Go through all tests (the index is increasing, but I will work from the
    # largest p-value down to the smallest)
    for i in range(T):
        # Set up a convergence indicator
        converged = False

        # Check whether this is any p-value but the largest
        if i != 0:
            # If it is not, replace the upper bound as the q-value from the last
            # iteration, i.e. from the last p-value, since the current test
            # would automatically be rejected if the preceding one was rejected.
            # (Which implies that the q-value for the current test has to be
            # at least weakly lower than that for the preceding one.)
            qhi[M-i-1,0] = qhi[M-i,0]

        # Iterate until convergence is achieved
        while not converged:
            # Calculate the midpoint between the lower and upper bound on the
            # current p-value, starting with the highest and working my way down
            # as i increases
            mp = (qlo[M-i-1,0] + qhi[M-i-1,0]) / 2

            # Check whether I would reject at the midpoint, using the Benjamini-
            # Hochberg criterion
            rej_mp = (p[M-i-1] < (mp * (M-i)) / M)

            # Check whether I rejected
            if rej_mp:
                # If so, make the midpoint the new upper bound, since the value
                # I'm looking for, which is the lowest q at which I would
                # reject, must be below it
                qhi[M-i-1,0] = mp
            else:
                # Otherwise, make the midpoint the new lower bound, for the same
                # reason, but from below
                qlo[M-i-1,0] = mp

            # Check whether the bounds are within tolerance
            if np.abs(qlo[M-i-1,0] - qhi[M-i-1,0]) <= tol:
                # If so, set the convergence flag to one
                converged = True

    # Calculate q-values
    q_unord = (qhi + qlo) / 2
    q_ord = np.zeros(shape=q_unord.shape)

    # Put the q-values back in the same order as the input p-values
    for sorti, origi in enumerate(p_sorted_index): q_ord[origi] = q_unord[sorti]

    # Put the ordered q-values back in the same shape as the input p-values
    q_ord = np.reshape(q_ord, newshape=(M,k), order=order)

    # Return the q-values
    return q_ord

################################################################################
### Part 2.1: Set directories
################################################################################

# Set data directory (doesn't need to exist, if you choose to download the data)
ddir = '/data'

# Set a flag which will be turned to true if the data directory is generated,
# to make sure data are downloaded
download_enforce = False

# Set figures/tables directory (doesn't need to exist)
fdir = '/figures'

# Create the data directory if it doesn't exist
for subdir in [fdir, ddir]:
    if not path.isdir(mdir+subdir):
        mkdir(mdir+subdir)

        # If this is the data directory, ensure download later on
        if subdir == ddir:
            download_enfore = True

################################################################################
### Part 2.2: Download/load data
################################################################################

# Change directory to data
chdir(mdir+ddir)

# Specify name of data file
data_file = 'fertility_regressions.dta'

# Specify whether to download the data, or use a local copy instead
download_data = True

# This gets overridden if the data directory didn't exist before
if download_enforce:
    download_data = True

# Check whether to download the data
if download_data:
    # Specify URL for data zip file containing data file
    web_zip_url = 'https://www.aeaweb.org/aer/data/10407/20101434_data.zip'

    # Access file on server using requests.get(), which just connects to the raw
    # file and makes it possible to access it
    with requests.get(web_zip_url, stream=True) as web_zip_raw:
        # Use io.BytesIO() to convert the raw web file into a proper file saved
        # in memory, and use ZipFile() to make it into a zip file object
        with ZipFile(io.BytesIO(web_zip_raw.content)) as web_zip:
            # Go through all files in the zip file
            for file in web_zip.namelist():
                # Find the data file. The problem is that it will show up as
                # <path>/data_file, so this checks whether it can find such a
                # <path>/<file> combination which ends with /data_file
                if file.endswith('/'+data_file):
                    # Once it's found it, unpack it using extract(), and copy
                    # the result into the data directory
                    copyfile(web_zip.extract(file), mdir+ddir+'/'+data_file)

                    # But now, the data directory also contains
                    # <path>/data_file, which I don't need. Of course, the
                    # <path> part really consists of
                    # <path1>/<path2>/.../data_file. This regular expression
                    # takes that string, splits it at the first </>, and keeps
                    # the first part, i.e. <path1>.
                    zipdir = re.split('/', file, maxsplit=1)[0]

                    # Delete that folder
                    rmtree(zipdir)

# Load data into memory
id = 'respondentid'  # Column to use as ID
data = pd.read_stata(data_file, index_col=id, convert_categoricals=False)

################################################################################
### Part 3: Data processing
################################################################################

# Specify indicator for ITT sample (people who were assigned to treatment)
v_itt = 'ittsample4'

# Specify indicator for being in the ITT sample and having follow up data
v_itt_follow = 'ittsample4_follow'

# Specify variable denoting couple treatment
v_couple_treatment = 'Icouples'

################################################################################
### Part 3.1: Responder status
################################################################################

# AFL count people as responders if they believe their partner wants more
# children than they do, and if they don't want to have a child over the next
# two years. Some preliminary variables need to be created to get responder
# status.

# Generate an indicator for whether the woman believes her partner wants a
# higher minimum number of children than she does (pandas knows that the column
# names I'm giving it refer to columns)
v_minkids_self = 'e8minnumber'
v_minkids_husb = 'e19minnumber_hus'
v_husb_more_minkids = 'husb_more_kids'
data[v_husb_more_minkids] = data[v_minkids_husb] > data[v_minkids_self]

# Replace it as NaN if one of the two components is missing (here, I need to
# select both columns and rows; .loc[<rows>, <columns>] is very useful for
# getting rows using a boolean vector, and columns using column names or
# something like <:> to select all columns)
data.loc[np.isnan(data[v_minkids_husb] + data[v_minkids_self]),
    v_husb_more_minkids] = np.nan

# Generate an indicator for whether the woman believes her husband wants a
# higher ideal number of children than she wants
v_idkids_self = 'e1_ideal'
v_idkids_husb = 'e12_hus_ideal'
v_husb_more_idkids = 'husb_more_idkids'
data[v_husb_more_idkids] = data[v_idkids_husb] > data[v_idkids_self]

# Replace it as NaN if the ideal number of kids for the husband is missing
v_idkids_husb_miss = 'd_e12_hus_ideal'
data.loc[data[v_idkids_husb_miss] == 1, v_husb_more_idkids] = np.nan

# Generate an indicator for whether the woman believes her partner wants a
# higher maximum number of children than she does
v_maxkids_self = 'e7maxnumber'
v_maxkids_husb = 'e18maxnumber_hus'
v_husb_more_maxkids = 'husb_more_maxkids'
data[v_husb_more_maxkids] = data[v_maxkids_husb] > data[v_maxkids_self]

# Replace it as NaN if either of the components are missing
data.loc[np.isnan(data[v_maxkids_husb] + data[v_maxkids_self]),
    v_husb_more_maxkids] = np.nan

# Generate an indicator for whether the couple currently have fewer children
# than the husband would ideally want to have
v_num_kids = 'currentnumchildren'
v_how_many_more = 'e17morekids_hus'
v_husb_wants_kids = 'husb_wants_kids'

# A note on the variable created in the next step: The original STATA code is
#
# gen h_wantsmore_ideal_m = (((e12_hus_ideal-currentnumchildren)>0) | e17morekids>0 )
#
# but that codes observations as 1 if either e12_hus_ideal or currentnumchildren
# are missing, and if e17morekids is missing. (Since in STATA, anything
# involving missing values is infinitely large and counted as True.) That is why
# I added np.isnan(data[v_idkids_husb] + data[v_num_kids]), which replicates the
# e12_hus_ideal / currentnumchildren issue, and np.isnan(data[v_how_many_more]),
# which replicates the e17morekids issue. (Since np.nan always evaluates to
# False in Python, I have to manually add these conditions to replicates the
# STATA assignments.) The problem is that in the next step, where these
# erroneous assignments are converted to missing, they forgot one condition, I
# think. (See below.)
data[v_husb_wants_kids] = (
    ((data[v_idkids_husb] - data[v_num_kids]) > 0) | (data[v_how_many_more] > 0)
    | np.isnan(data[v_idkids_husb] + data[v_num_kids])
    | np.isnan(data[v_how_many_more])
    )

# Replace it as NaN if any of the components are missing
# The original STATA code is
#
# replace h_wantsmore_ideal_m = . if (d_e12_hus_ideal==1 | currentnumchildren==.) & (e17morekids==-9)
#
# which corrects the issue with missing values for e12_hus_ideal or
# currentnumchildren making the variable true. But it doesn't solve the
# same issue for e17morekids, since that is sometimes coded as -9 (which means
# the responder said she they don't know), but in other cases, it's just coded
# as missing. This code will not fix the missing issue.
data.loc[((data[v_idkids_husb_miss] == 1) | np.isnan(data[v_num_kids]))
    & (data[v_how_many_more] == -9),
    v_husb_wants_kids] = np.nan

# Specify variable name for indicator of whether the woman wants kids in the
# next two years
v_kids_nexttwo = 'wantschildin2'

# Generate an indicator for responder status (luckily, Python evaluates
# np.nan == True as False, so this code works for boolean data)
v_responder = 'responder'
data[v_responder] = (
    ((data[v_husb_more_minkids] == True) | (data[v_husb_more_maxkids] == True)
    | (data[v_husb_more_idkids] == True))
    & (data[v_husb_wants_kids] == True) & (data[v_kids_nexttwo] == 0)
    )

# Replace it as missing if some of the components are missing
data.loc[(np.isnan(data[v_husb_more_minkids]) &
    np.isnan(data[v_husb_more_maxkids]) & np.isnan(data[v_husb_more_idkids]))
    | np.isnan(data[v_husb_wants_kids]) | np.isnan(data[v_kids_nexttwo]),
    v_responder] = np.nan

################################################################################
### Part 3.2: Positive effects on well-being
################################################################################

# Generate a dummy measure of life satisfaction
v_satisfaction_detail = 'j11satisfy'
v_satisfied = 'satisfied'
data[v_satisfied] = (
    (data[v_satisfaction_detail] == 4) | (data[v_satisfaction_detail] == 5)
    )

# Replace it as missing if the life satisfaction score is missing
data.loc[np.isnan(data[v_satisfaction_detail]), v_satisfied] = np.nan

# Generate a dummy measure of health
v_health_detail = 'a21health'
v_healthier = 'healthier'
data[v_healthier] = (
    (data[v_health_detail] == 4) | (data[v_health_detail] == 5)
    )

# Replace it as missing if the health score is missing
data.loc[np.isnan(data[v_health_detail]), v_healthier] = np.nan

# Generate a dummy measure of happiness
v_happy_detail = 'a22happy'
v_happier = 'happier'
data[v_happier] = (
    (data[v_happy_detail] == 4) | (data[v_happy_detail] == 5)
    )

# Replace it as missing if the happiness score is missing
data.loc[np.isnan(data[v_happy_detail]), v_happier] = np.nan

################################################################################
### Part 3.3: Negative side effects
################################################################################

# Generate an indicator for being separated
v_marital_status = 'b1marstat'
v_separated = 'separated'
data[v_separated] = (
    (data[v_marital_status] == 2) | (data[v_marital_status] == 3)
    | np.isnan(data[v_marital_status])
    )

# Generate an indicator for the partner being physically violent
v_violence_detail = 'f10violenc'
v_violence = 'violent'
data[v_violence] = (data[v_violence_detail] == 1)

# Replace it as missing if the detailed violence data are less than zero or
# missing
data.loc[((data[v_violence_detail] < 0) | np.isnan(data[v_violence_detail])),
    v_violence] = np.nan

# Generate an indicator for condom usage
v_condom_detail = 'g14mc'
v_condom = 'condom'
data[v_condom] = (data[v_condom_detail] == 1)

# Replace it as missing if the condom usage data are missing
data.loc[np.isnan(data[v_condom_detail]), v_condom] = np.nan

################################################################################
### Part 5: Adjusted p-values
################################################################################

# Specify two (sub-)families of outcomes, one for measures of well-being, the
# other for measures of potential negative side effects
sad_family = [v_separated, v_violence, v_condom]
happy_family = [v_satisfied, v_healthier, v_happier]

# Combine them into one family
big_family = happy_family + sad_family

# Collect all families
neighborhood = [big_family, happy_family, sad_family]

# Make a dictionary of names for later
family_name = {tuple(big_family): 'all', tuple(happy_family): 'wellbeing',
    tuple(sad_family): 'sideff'}

# Make an indicator for being in the ITT sample, which will be useful later,
# since that's where treatment is being assigned
I_itt = (data[v_itt] == 1)

# Make an indicator for being a responder and being in the ITT follow-up sample,
# which is the subgroup I'm interested in here
I_resitt = (data[v_responder] == 1) & (data[v_itt_follow] == 1)

# Get the treatment indicator for the same group (these should be integers)
Xresitt = np.array(data.loc[I_resitt, v_couple_treatment].astype(int).values,
    ndmin=2).transpose()

# Add an intercept
Xresitt = np.concatenate((np.ones(shape=(Xresitt.shape[0],1)), Xresitt), axis=1)

# If I wanted to use covariates, I'd like to be able to easily ignore those
# when calculating p-values etc., so specify index of coefficient(s) of
# interest. Has to be a list!
cidx = [1]

# Get number of parameters of interest
k = len(cidx)

# Get the treatment indicator for the same group
Xitt = np.array(data.loc[I_itt, v_couple_treatment].astype(int).values,
    ndmin=2).transpose()

# Calculate number of treated units in the sample
ntreat = np.sum(Xitt)

# Make an intercept for the ITT sample
beta0 = np.ones(shape=(Xitt.shape[0],1))

# Specify balancing variable for minmax t randomization
balvars = ['a16_3_ageinyrs', 'school', 'step3_numchildren', 'e1_ideal',
    'fertdesdiff2', 'step7_injectables', 'step7_pill']

# Get data for balancing variable in the ITT sample
BV = data.loc[I_itt, balvars].astype(float).values

# Specify how many cores to use for parallel processing
ncores = cpu_count()

# Set number of replications for the randomization distribution
R = 100000

# Set number of balancing regressions
Breg = 100

# Specify some column headers for printing the results later
col_headers = ['N', 'b_hat', 'SE', 'p', 'p_bf', 'p_hbf', 'p_sr', 'q_bh']

# Specify a label for the outcome column separately
lab_outcome = 'outcome'

# Set pandas print options
pd.set_option('display.max_columns', len(big_family)+2)
pd.set_option('display.width', 110)
pd.set_option('display.precision', 3)

# Change directory to figures/tables
chdir(mdir+fdir)

# Print an empty line before printing other output
print()

# Print the number of replications and balancing regressions
print(R, ' replications for RD, ', Breg,
      ' balancing regressions', sep='')
print()

# Go through all families
for f, family in enumerate(neighborhood):
    # Get number of family members
    M = len(family)

    # Get the data for responders in the ITT follow-up sample, making sure these
    # are provided in float format
    Y = data.loc[I_resitt, family].astype(float).values

    # Set up vector of coefficient estimates b, vector of standard errors SE,
    # and vector of unadjusted p-values p (these could be matrices if I cared
    # about more than one treatment arm or something)
    b = np.zeros(shape=(M,k))
    SE = np.zeros(shape=(M,k))
    p_unadj = np.zeros(shape=(M,k))

    # Set up vector of sample sizes
    N = np.zeros(shape=(M,1))

    # Go through all members in the family
    for i in range(M):
        # Make an index of where both the member and all parts of X are not NaN
        I = (~np.isnan(Y[:,i]) & ~np.isnan(Xresitt.sum(axis=1)))

        # Get the number of effective observations
        n = I.sum()

        # Save it for printing later
        N[i,0] = n

        # Get outcome variable, i.e. the current member of the family, for those
        # observations with non-missing data for both LHS and RHS variables
        y = np.array(Y[I,i], ndmin=2).transpose()

        # Run OLS
        bhat, Vhat, _, p = ols(y, Xresitt[I], cov_est='hmsd')

        # Save p-values
        p_unadj[i,:] = p[cidx,0]

        # Save point estimates for coefficients of interest
        b[i,:] = bhat[cidx,0]

        # Save standard errors
        SE[i,:] = np.sqrt(np.diag(Vhat[cidx,cidx]))

    ############################################################################
    ### Part 5.1: Bonferroni, Holm-Bonferroni
    ############################################################################

    # Calculate Bonferroni-adjusted p-values
    p_bonf = bonferroni(p_unadj)

    # Calculate Holm-Bonferroni adjusted p-values
    p_holmbonf = holm_bonferroni(p_unadj)

    ############################################################################
    ### Part 5.2: Free step down resampling
    ############################################################################

    # Get data for ITT sample
    Y = data.loc[I_itt, family].astype(float).values

    # Get the ranking of unadjusted p-values
    p_unadj_sort_idx = p_unadj[:,0].argsort()

    # Get randomization p-values using all available cores in parallel. Note
    # that the sample index this gets is the indicator for being a responder and
    # in the ITT follow-up sample, but only for those people who are in the
    # original ITT sample
    P = Parallel(n_jobs=ncores)( delayed(permute_p)
        (Y=Y, X=beta0, Isamp=I_resitt[I_itt], ntreat=ntreat,
         prank=p_unadj_sort_idx, balvars=BV, seed=f*R+r, Breg=Breg)
        for r in range(R) )

    # Count how often the randomization values are below the original p-values
    P = np.sum([(p_star <= p_unadj) for p_star in P], axis=0)

    # Divide by the number of iterations
    P = P / R

    # Order the permutation p-values in the same way
    P = P[p_unadj_sort_idx]

    # Go through all but the first p-values
    for i in range(P.shape[0]-1):
        # Replace the current adjusted p-value with the preceding one if that
        # is larger
        P[i+1] = np.maximum(P[i], P[i+1])

    # Set up vector of reordered permutation p-values
    P_sr = np.zeros(shape=P.shape)

    # Put the p-values back in that order
    for sorti, origi in enumerate(p_unadj_sort_idx): P_sr[origi] = P[sorti]

    ############################################################################
    ### Part 6: Benjamini-Hochberg FDR control
    ############################################################################

    # Calculate the Benjamini-Hochberg q-values
    q_bh = benjamini_hochberg(p_unadj)

    ############################################################################
    ### Part 7: Print the results
    ############################################################################

    # Put the results into a data frame, add column labels
    res = pd.DataFrame(
        data=np.concatenate((N, b, SE, p_unadj, p_bonf, p_holmbonf, P_sr, q_bh),
        axis=1), columns=col_headers)

    # Make sure sample sizes are saved as integers
    res[col_headers[0]] = res[col_headers[0]].astype(int)

    # Add outcome labels to the results data frame
    res.insert(loc=0, column=lab_outcome, value=[f.capitalize() for f in family])

    # Use these as the new index
    res.set_index(lab_outcome, inplace=True)

    # Print the results
    print(res, '\n')

    # Save results as Latex table
    res.to_latex('results_'+family_name[tuple(family)]+'.tex', index=True)
