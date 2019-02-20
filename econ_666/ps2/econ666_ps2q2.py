################################################################################
### Econ 666, PS2Q2: Power calculations
################################################################################

# Import matplotlib
import matplotlib as mpl

# Select backend that does not open figures interactively (has to be done before
# pyplot is imported); without this, Python will get confused when it tries to
# close figures, and it will send annoying warnings
mpl.use('Agg')

# Import other necessary packages and functions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from inspect import getsourcefile
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from os import chdir, mkdir, path
from scipy.optimize import fsolve, minimize
from scipy.stats import norm
from scipy.stats import t

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
### Directories, graph options
################################################################################

# Set graph options
#mpl.rcParams["text.latex.preamble"].append(r'\usepackage{amsmath}')
#plt.rc('font', **{'family': 'serif', 'serif': ['lmodern']})
#plt.rc('text', usetex=True)

# Set figures/tables directory (doesn't need to exist)
fdir = '/figures'

# Create the figures directory if it doesn't exist
if not path.isdir(mdir+fdir):
    mkdir(mdir+fdir)

# Change to figures directory
chdir(mdir+fdir)

################################################################################
### Setup
################################################################################

# Set random number generator seed
np.random.seed(666)

# Specify effect size, in units of the control group's standard deviation
ES = .2

# Specify unskilled farmers mean yield
mean_unskilled = 10

# Specify standard deviation of unskilled farmers' yields
sd_unskilled = 5

# Specify lower and upper bound on skilled farmers' yields
yield_lower_skilled = 16
yield_upper_skilled = 28

# Specify level of test
alpha = .05

# Specify desired power
kappa = .8

# Specify probability of treament
p = .5

################################################################################
### 2(b)
################################################################################

# Calculate treated mean
mean_treated = mean_unskilled + ES * sd_unskilled

# Calculate the treatment effect
beta = mean_treated - mean_unskilled

# Define a function to calculate the sample size
def N(MDE, alpha, kappa, p, sigma2):
    # Define the equation defining the sample size
    def N_eq(N):
        # Distribution for the control group
        Fc = t(df=p*N)

        # Distribution for the treatment group
        Ft = t(df=p*N, loc=MDE)

        # Calculate discrepancy
        delta_N = (
            MDE
            - (Fc.ppf(1-alpha/2) + Ft.ppf(1-kappa))
            * np.sqrt( (p * (1-p))**(-1) * sigma2 / N))

        # Return discrepancy
        return delta_N

    # Calculate N
    N = fsolve(N_eq, 200)

    # Return N (note that fsolve provides this as an array, so return only the
    # first (and only) element; make sure it's an integer
    return np.int(np.ceil(N[0]))

# Calculate N, assuming homoskedastic variance and a two-sided test
N_q2b = N(MDE=beta, alpha=alpha, kappa=kappa, p=p, sigma2=sd_unskilled**2)

# Display the result
print('\n2(b) Sample size needed:', N_q2b)

################################################################################
### 2(c)
################################################################################

# Specify the sample size
N_q2c = 200

# Set up distribution for unskilled outcomes
F_unskilled = t(df=p*N_q2c)

# Set up distribution for treated outcomes
F_treated = t(df=p*N_q2c, loc=beta)

# Calculate the power
kappa_q2c = F_treated.cdf(
    beta * np.sqrt(p*(1-p) * N_q2c) / sd_unskilled
    - F_unskilled.ppf(1 - alpha/2)
    )

# Print the result
print('\n2(c) Power:', kappa_q2c)

################################################################################
### 2(d)
################################################################################

# Specify take-up rates
D_control = 0
D_treated = .5

# Calculate sample size
N_q2d = np.int(np.ceil(
    N_q2b * (D_treated - D_control)**(-2)
    ))

# Print the result
print('\n2(d) Sample size needed:', N_q2d)

################################################################################
### 2(e)
################################################################################

# Define a function to calculate the MDE
def MDE(N, alpha, kappa, p, sigma2, F_crit=None):
    # Define the equation defining the MDE
    def MDE_eq(E):
        # Check whether a distribution of test statistics was supplied as a
        # Numpy column vector; see question 2(h) for why that is useful
        if F_crit is None:
            # Distribution for the control group
            Fc = t(df=p*N)

            # Get the critical value from that distribution
            crit_control = Fc.ppf(1-alpha/2)

            # Distribution for the treatment group
            Ft = t(df=p*N, loc=E)

            # Get critical value for the treatment group
            crit_treatment = Ft.ppf(1-kappa)
        else:
            # Get index of element needed for a two-sided test at level alpha
            alpha_idx = np.int((1-alpha/2)*F_crit.shape[0] - 1)

            # Get critical value for the control group from supplied values
            crit_control = F_crit[alpha_idx,0]

            # Get index of element needed for kappa level of power
            kappa_idx = np.int((1-kappa)*F_crit.shape[0] - 1)

            # Get critical values for the treatment group
            crit_treatment = F_crit[kappa_idx,0] + E

        # Calculate discrepancy
        delta_MDE = (
            E
            - (crit_control + crit_treatment)
            * np.sqrt( (p * (1-p))**(-1) * sigma2 / N))

        # Return discrepancy
        return delta_MDE

    # Calculate MDE
    MDE = fsolve(MDE_eq, .2)

    # Return MDE (note that fsolve provides this as an array, so return only the
    # first (and only) element
    return MDE[0]

# Make a vectorized version of the function
vec_MDE = np.vectorize(MDE)

# Specify minimum and maximum number of N over which to plot
Nmin = 500
Nmax = 501

# Calculate number of steps (using only integers)
Nsteps = Nmax - Nmin + 1

# Make a vector of Ns
x = np.linspace(Nmin, Nmax, Nsteps)

# Set up a figure (using the num keyword makes sure that if this script is run a
# a lot of times within the same Python process, this figure gets overwritten,
# instead of starting a bunch of figures and using up lots of memory)
fig, ax = plt.subplots(1, 1, num='fig_mde_n', figsize=(6.5, 6.5*(9/16)))

# Plot the MDE for all Ns
ax.plot(x, vec_MDE(x, alpha=alpha, kappa=kappa, p=p, sigma2=sd_unskilled**2),
    color='blue', linestyle='-')

# Set x axis limits
ax.set_xlim(Nmin, Nmax)

# Label y axis, set label position (getting the dollar sign to show up is a bit
# of a pain)
ax.set_ylabel(r'MDE', fontsize=11, rotation=0)

# Add some more space after the horizontal axis label
ax.yaxis.labelpad = 20

# Label x axis, set label position
ax.set_xlabel('Sample size', fontsize=11)

################################################################################
### 2(f)
################################################################################

# Specify by what factor the baseline survey reduces the variance
res_var_fac = .8

# Calculate residual variance
res_var = res_var_fac*sd_unskilled**2

# Add alternative MDE to the plot
ax.plot(x, vec_MDE(x, alpha=alpha, kappa=kappa, p=p, sigma2=res_var),
    color='blue', linestyle='--')

# Calculate minimum and maximum MDEs (i.e. for the ends of the interval). Note
# that the lower end produces the max, and the upper end produces the end.
MDEmin = MDE(Nmax, alpha=alpha, kappa=kappa, p=p, sigma2=res_var)
MDEmax = MDE(Nmin, alpha=alpha, kappa=kappa, p=p, sigma2=sd_unskilled**2)

# Add a lagend to the figure
ax.legend(['Without baseline survey', 'With baseline survey'])

# Get rid of unnecessary whitespace
fig.tight_layout()

# Save figure
figname_q2e = 'q2e.pdf'  # Name to use when saving figure
plt.savefig(figname_q2e, bbox_inches='tight')

# Close the figure
plt.close('fig_mde_n')

################################################################################
### 2(g)
################################################################################

################################################################################
### 2(h)
################################################################################

# Set sample size
N_q2h = 3000

# Set fraction of large villages
f_lg = .5

# Set treatment probabilities
p_sm = .7  # Small villages
p_lg = .3  # Large villages

# Calculate population in small and large villages
N_sm = (1-f_lg) * N_q2h
N_lg = f_lg * N_q2h

# Generate outcome for the unskilled population
y_unskilled = np.random.normal(size=(N_q2h, 1), scale=sd_unskilled)

# Generate village indicator
# Draw random normals
V_lg = np.random.normal(size=N_q2h)

# Allocate everyone ranked at or below the number of people in large villages
# to a large village; that is, V[i] = 1 <=> i lives in a large village
V_lg = (V_lg.argsort() + 1 <= N_lg)

# Generate treatment indicator
# Draw random normals
W = np.random.normal(size=N_q2h)

# Replace the treatment in small villages as 1 with probability p_sm
W[~V_lg] = (W[~V_lg].argsort() + 1 <= p_sm * N_sm)

# Replace the treatment in large villages as 1 with probability p_lg
W[V_lg] = (W[V_lg].argsort() + 1 <= p_lg * N_lg)

# Make the treatment indicator into a proper vector
W = np.array(W, ndmin=2).transpose()

# Generate observed data
y = y_unskilled + beta * W

# Make an intercept for regressions
cons = np.ones(shape=(N_q2h, 1))

# Specify number of simulations to get the randomization distribution
R = 1000

# Define a function to do one iteration of the randomization distribution
def random_t(N, y, x1, V, N_0, N_1, p_0, p_1, seed):
    # Set random number generator's seed
    np.random.seed(seed)

    # Generate treatment indicator
    # Draw random normals
    W = np.random.normal(size=N)

    # Replace the treatment in small villages as 1 with probability p_sm
    W[~V] = (W[~V].argsort() + 1 <= p_0 * N_0)

    # Replace the treatment in large villages as 1 with probability p_lg
    W[V] = (W[V].argsort() + 1 <= p_1 * N_1)

    # Make the treatment indicator into a proper vector
    W = np.array(W, ndmin=2).transpose()

    # Generate RHS data
    X = np.concatenate((x1, W), axis=1)

    # Estimate regression, get t-statistics
    _, _, t = ols(y, X, cov_est='hmsd', get_p=False)

    # Return t-statistic for treatment dummy
    return t[1,0]

# Get number of available cores
ncores = cpu_count()

# Get randomization distribution
T = Parallel(n_jobs=ncores)( delayed(random_t)
        (N=N_q2h, y=y, x1=cons, V=V_lg, N_0=N_sm, N_1=N_lg, p_0=p_sm, p_1=p_lg,
        seed = r) for r in range(R) )

# Make the randomization t-statistics into a proper (column) vector
T = np.array(T, ndmin=2).transpose()

# Sort the t-statistics (Python sorts from smallest to largest)
T.sort(axis=0)

# Calculate MDE
MDE_q2h = MDE(
    N=N_q2h, alpha=alpha, kappa=kappa, p=p, sigma2=sd_unskilled**2, F_crit=T
    )

# Print the results
print('\n2(h) MDE:', MDE_q2h)

#
