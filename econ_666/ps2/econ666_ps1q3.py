################################################################################
### Econ 666, PS2Q1: Multiple testing
################################################################################

# Import necessary packages and functions
import io
import numpy as np
import pandas as pd
import re
import requests
from joblib import Parallel, delayed
from linreg import ols
from multiprocessing import cpu_count
from os import chdir, mkdir, path
from shutil import copyfile, rmtree
from zipfile import ZipFile

################################################################################
### Part 1: Define necessary functions
################################################################################

# Define a function to do the Bonferroni correction
def bonferroni(p):
    # Get number of members in the family M, and number of parameters of
    # interest k
    M, k = p.shape

    # Calculate Bonferroni corrected p-values
    p_bc = np.minimum(p * (M*k), 1)

    # Return them
    return p_bc

# Define a function to get Holm-Bonferroni adjusted p-values
def holm_bonferroni(p, alpha=.05, order='F'):
    # Get original dimensions of p-values, which might be provided as a matrix
    M, k = p.shape

    # Flatten the array of p-values, in case a matrix is provided. The order
    # argument is important only to ensure that this is put back into place the
    # same way later. Which order is chosen does not matter.
    p = p.flatten(order=order)

    # Get indices of sorted p-values
    p_sorted_index = np.argsort(p)

    # Sort the p-values, make them into a proper (column) vector
    p = np.array(p[p_sorted_index], ndmin=2).transpose()

    # Set up array of adjusted p-values
    p_hb = p * np.array([M*k-s for s in range(M*k)], ndmin=2).transpose()

    # Go through all p-values but the first and enforce monotonicity
    for i, pv in enumerate(p_hb[1:]):
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
def permute_p(Y, Isamp, ntreat, balvars, X=None, Z=None, seed=1,
    Breg=2, breg_icept=True, cov_est='hmsd', order='F', shape=None):
    # Set random number generator's seed
    np.random.seed(seed)

    # Get total sample size N and number of outcome variables M
    N, M = Y.shape

    if X is None and Z is None:
        cidx = [0]
    elif X is not None and Z is None:
        cidx = [X.shape[1]]
    else:
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
            tmax = np.amax(t)

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

        # If add was specified, add it in after the treatment assignment
        if Z is not None:
            Xstar = np.concatenate((Xstar, Z[I]), axis=1)

        # Run OLS
        _, _, _, p = ols(y, Xstar, cov_est='hmsd')

        # Save only p-values of interest
        pstar[i,:] = p[cidx]

    # Reshape the result if desired
    if shape is not None:
        pstar = pstar.reshape(order=order)

    # Return the adjusted p-values
    return pstar

################################################################################
### Part 2.1: Set directories
################################################################################

# Specify name for main directory (just uses the file's directory)
mdir = path.dirname(path.abspath(__file__)).replace('\\', '/')

# Set data directory (doesn't need to exist)
ddir = '/data'

# Create the data directory if it doesn't exist
if not path.isdir(mdir+ddir):
    mkdir(mdir+ddir)

# Change directory to data
chdir(mdir+ddir)

################################################################################
### Part 2.2: Download/load data
################################################################################

# Specify name of data file
data_file = 'fertility_regressions.dta'

# Specify whether to download the data, or use a local copy instead
download_data = False

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
# which corrects the issue with missing values for  e12_hus_ideal or
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

#eststo D: reg satisfied Icouples if ittsample4_follow == 1 & responder_m==1
#eststo E: reg healthier Icouples if ittsample4_follow == 1 & responder_m==1
#eststo F: reg happier Icouples if ittsample4_follow == 1 & responder_m==1

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

#eststo D: reg separated2 Icouples if ittsample4_follow == 1 & responder_m==1
#eststo E: reg violence_follow Icouples if ittsample4_follow == 1 & responder_m==1
#eststo F: reg cur_using_condom Icouples if ittsample4_follow == 1 & responder_m==1

################################################################################
### Part 4: Recreate baseline results
################################################################################

# Specify two (sub-)families of outcomes, one for measures of well-being, the
# other for measures of potential negative side effects
sad_family = [v_separated, v_violence, v_condom]
happy_family = [v_satisfied, v_healthier, v_happier]

# Combine them into one family
family = sad_family + happy_family

# Get number of family members
M = len(family)

# Make an indicator for being a responder and being in the ITT follow-up sample,
# which is the subgroup I'm interested in here
I_resitt = (data[v_responder] == 1) & (data[v_itt_follow] == 1)

# Get the data for responders in the ITT follow-up sample, making sure these are
# provided in float format
Y = data.loc[I_resitt, family].astype(float).values

# Get the treatment indicator for the same group (these should be integers)
X = np.array(data.loc[I_resitt, v_couple_treatment].astype(int).values,
    ndmin=2).transpose()

# Add an intercept
X = np.concatenate((np.ones(shape=(X.shape[0],1)), X), axis=1)

# If I wanted to use covariates, I'd like to be able to easily ignore those
# when calculating p-values etc., so specify index of coefficient(s) of
# interest. Has to be a list!
cidx = [1]

# Get number of parameters of interest
k = len(cidx)

# Set up vector of coefficient estimates b, vector of standard errors SE, and
# vector of unadjusted p-values p (these could be matrices if I cared about more
# than one treatment arm or something)
b = np.zeros(shape=(M,k))
SE = np.zeros(shape=(M,k))
p_unadj = np.zeros(shape=(M,k))

# Set up vector of sample sizes
N = np.zeros(shape=(M,1))

# Go through all members in the family
for i in range(M):
    # Make an index of where both the member and all parts of X are not NaN
    I = (~np.isnan(Y[:,i]) & ~np.isnan(X.sum(axis=1)))

    # Get the number of effective observations
    n = I.sum()

    # Save it for printing later
    N[i,0] = n

    # Get outcome variable, i.e. the current member of the family, for those
    # observations with non-missing data for both LHS and RHS variables
    y = np.array(Y[I,i], ndmin=2).transpose()

    # Run OLS
    bhat, Vhat, _, p = ols(y, X[I], cov_est='hmsd')

    # Save p-values
    p_unadj[i,:] = p[cidx,0]

    # Save point estimates for coefficients of interest
    b[i,:] = bhat[cidx,0]

    # Save standard errors
    SE[i,:] = np.sqrt(np.diag(Vhat[cidx,cidx]))

################################################################################
### Part 5: Adjusted p-values
################################################################################

################################################################################
### Part 5.1: Bonferroni, Holm-Bonferroni
################################################################################

# Calculate Bonferroni-adjusted p-values
p_bonf = bonferroni(p_unadj)

# Calculate Holm-Bonferroni adjusted p-values
p_holmbonf = holm_bonferroni(p_unadj)

################################################################################
### Part 5.2: Free step down resampling
################################################################################

# Make an indicator for being in the ITT sample, which will be useful later,
# since that's where treatment is being assigned
I_itt = (data[v_itt] == 1)

# Get data for ITT sample
Y = data.loc[I_itt, family].astype(float).values

# Get the treatment indicator for the same group
X = np.array(data.loc[I_itt, v_couple_treatment].astype(int).values,
    ndmin=2).transpose()

# Calculate number of treated units in the sample
ntreat = np.sum(X)

# Add an intercept
X = np.concatenate((np.ones(shape=(X.shape[0],1)), X), axis=1)

# Specify balancing variable for minmax t randomization
balvars = ['a16_3_ageinyrs', 'school', 'step3_numchildren', 'e1_ideal',
    'fertdesdiff2', 'step7_injectables', 'step7_pill']

# Get data for balancing variable in the ITT sample
BV = data.loc[I_itt, balvars].astype(float).values

# Specify how many cores to use for parallel processing
ncores = cpu_count()

# Set number of replications for the randomization distribution
R = 2

# Get p-values using all available cores in parallel
P = Parallel(n_jobs=ncores)(delayed(permute_p)(Y=Y, X=X, Isamp=I_resitt[I_itt], ntreat=ntreat, balvars=BV, seed=r) for r in range(R))

################################################################################
### Part 6: Print the results
################################################################################

# Put the results into a data frame, add column labels
res = pd.DataFrame(
    data=np.concatenate((N, b, SE, p_unadj, p_bonf, p_holmbonf), axis=1),
    columns=['N', 'b_hat', 'SE', 'p', 'p_bf', 'p_hbf']
    )

# Make sure sample sizes are saved as integers
res['N'] = res['N'].astype(int)

# Add outcome labels to the results data frame
res.insert(loc=0, column='outcome', value=family)

# Use these as the new index
res.set_index('outcome', inplace=True)

# Set print options for pandas
pd.set_option('display.max_columns', len(family)+1)
pd.set_option('display.width', 110)
pd.set_option('display.precision', 3)

# Print the results
print(res)
