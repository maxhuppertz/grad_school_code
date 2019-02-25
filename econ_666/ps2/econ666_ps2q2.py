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
from scipy.optimize import fsolve, minimize, NonlinearConstraint
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
mpl.rcParams["text.latex.preamble"].append(r'\usepackage{amsmath}')
plt.rc('font', **{'family': 'serif', 'serif': ['lmodern']})
plt.rc('text', usetex=True)

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
        Fc = t(df=p*N-2)

        # Distribution for the treatment group
        Ft = t(df=p*N-2, loc=MDE)

        # Calculate discrepancy
        delta_N = (
            MDE
            - (Fc.ppf(1-alpha/2) + Ft.ppf(kappa))
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
F_unskilled = t(df=N_q2c-2)

# Set up distribution for treated outcomes
F_treated = t(df=N_q2c-2, loc=beta)

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
def MDE(N, alpha, kappa, p, sigma2, F_crit=None, sigma_ovr=None):
    # Define the equation defining the MDE
    def MDE_eq(E):
        # Check whether a distribution of test statistics was supplied as a
        # Numpy column vector; see question 2(h) for why that is useful
        if F_crit is None:
            # Distribution for the control group
            Fc = t(df=N-2)

            # Get the critical value from that distribution
            crit_control = Fc.ppf(1-alpha/2)

            # Distribution for the treatment group
            Ft = t(df=N-2, loc=E)

            # Get the critical value for the treatment group
            crit_treatment = Ft.ppf(kappa)
        else:
            # Get index of element needed for a two-sided test at level alpha
            alpha_idx = np.int((1-alpha/2)*F_crit.shape[0] - 1)

            # Get critical value for the control group from supplied values
            crit_control = F_crit[alpha_idx,0]

            # Get index of element needed for kappa level of power
            kappa_idx = np.int((kappa)*F_crit.shape[0] - 1)

            # Get critical values for the treatment group
            crit_treatment = F_crit[kappa_idx,0] + E

        # Check whether an override for sigma_hat was provided
        if sigma_ovr is None:
            # If not, get sigma_hat using the standard formula
            sigma_hat = np.sqrt( (p * (1-p))**(-1) * sigma2 / N )
        else:
            # If yes, use it
            sigma_hat = sigma_ovr

        # Calculate discrepancy
        delta_MDE = E - (crit_control + crit_treatment) * sigma_hat

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

# Define a function to calculate the vector of sample sizes needed to do three
# tests, a) treatment 1 vs. control, b) treatment 2 vs. control, and c)
# treatment 2 vs. treatment 1, at level alpha (two sided) and with power kappa
# against the targeted (true) vector of MDEs MDE_bar
def N_mult(MDE_bar, alpha, kappa, sigma2, C, W):
    # Get number of groups
    ngroup = len(C)

    # Get number of effects
    neff = len(MDE_bar)

    # Define a function that calculates how much it costs to sample a given
    # vector of sample sizes
    def cost_N(N):
        return sum([N[i] * C[i] for i in range(ngroup)])

    # Define a function that calculates implied treatment probabilities for a
    # given set of sample sizes and gets the resulting MDEs
    def MDE_N(N):
        # Calculate probabilities
        P = [N[1]/(N[0]+N[1]), N[2]/(N[0]+N[2]), N[2]/(N[1]+N[2])]

        # Calculate MDEs
        MDEs = [
            MDE(N[0]+N[1], alpha, kappa[0], P[0], sigma2),  # T1 vs. C
            MDE(N[0]+N[2], alpha, kappa[1], P[1], sigma2),  # T2 vs. C
            MDE(N[1]+N[2], alpha, kappa[2], P[2], sigma2)  # T2 vs. T1
            ]

        # Return the MDEs
        return MDEs

    # Calculate (weighted) targeted MDEs
    MDE_target = [MDE_bar[i]/W[i] for i in range(neff)]

    # Set up nonlinear constraints on the MDEs. These say that each MDE has to
    # be weakly smaller than the corresponding target value, that is, between
    # negative infinity and the target values
    const = NonlinearConstraint(MDE_N, [-np.inf for mde in MDE_bar], MDE_target)

    # Specify bounds to ensure positive sample sizes
    bounds_N = ((0, np.inf), (0, np.inf), (0, np.inf))

    # Calculate the optimal sample sizes by minimizing the total sample size
    # constrained by having to get each MDE at most as large as its target
    Nstar = minimize(cost_N,
                     x0=[300, 300, 300], bounds=bounds_N, constraints=const,
                     tol=10**(-14)).x

    # Make sure these are all rounded up integers
    Nstar = [np.int(np.ceil(n)) for n in Nstar]

    # Return the result
    return Nstar

# Specify treatment effect for second treatment, in units of unskilled SD
ES_2 = .4

# Calculate treatment 2 mean
mean_treated_2 = mean_unskilled + ES_2 * sd_unskilled

# Calculate the treatment effect for treatment 2
beta_2 = mean_treated_2 - mean_unskilled

# Make a vector of treatment effects for a) treatment 1 vs. control, b)
# treatment 2 vs. control, and c) treatment 2 vs. treatment 1
MDE_bar = [beta, beta_2, beta_2 - beta]

# Specify a vector of power levels for each test
kappa_q2g = [kappa, kappa, .75]

# Specify a vector of cost per unit sampled/treated
C = [.1, 1, 5]

# Specify a vector of importance weights for each test. Higher means more
# important. They way these work is that I inflate each MDE by 1/W[i], so
# a weight of one means targe the MDE, a weight of 1/2 means target twice
# the MDE (which allows for a lower sample size), and so on.
W = [1, 1, .8]

# Calculate sample sizes to get these effects as MDEs, if I want to do an alpha
# level (two sided) test and have power kappa
N_q2g = N_mult(MDE_bar=MDE_bar, alpha=alpha, kappa=kappa_q2g,
               sigma2=sd_unskilled**2, C=C, W=W)

# Print the results
print('\n2(g) Sample sizes required - C: ', N_q2g[0], ', T1: ', N_q2g[1],
      ', T2: ', N_q2g[2], sep='')

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

# Calculate population in small and large villages (have to be integers!)
N_sm = np.int((1-f_lg) * N_q2h)
N_lg = np.int(f_lg * N_q2h)

# Generate outcome for the unskilled population
y_unskilled = np.random.normal(size=(N_q2h, 1), scale=sd_unskilled)

# Generate village size indicator
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
R = 10000

# Make a village dummy
D = np.zeros(shape=(y.shape[0],1))

D[V_lg == 1] = 1

# Define a function to do one iteration of the randomization distribution
def random_t(y, x1, V, N_0, N_1, p_0, p_1, seed, D):
    # Set random number generator's seed
    np.random.seed(seed)

    # Calculate sample size
    N = y.shape[0]

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
    X = np.concatenate((x1, W, D), axis=1)

    # Estimate regression, get coefficients and t-statistics
    b, _, t = ols(y, X, cov_est='hmsd', get_p=False)

    # Return coefficient and t-statistic for treatment dummy
    return t[1,0], b[1,0]

# Get number of available cores
ncores = cpu_count()

# Get results
res = Parallel(n_jobs=ncores)(
    delayed(random_t)
    (y=y, x1=cons, V=V_lg, N_0=N_sm, N_1=N_lg, p_0=p_sm, p_1=p_lg, seed = r, D=D)
    for r in range(R))

# Make the results into a Numpy array
res = np.array(res, ndmin=2)

# Get the randomization t-statistics, make them  into a proper (column) vector
T = np.array(res[:,0], ndmin=2).transpose()

# Get the coefficient estimates, make them  into a proper (column) vector
B = np.array(res[:,1], ndmin=2).transpose()

# Calculate their standard deviation
sigma_hat = np.std(B, axis=0)

# Sort the t-statistics (Python sorts from smallest to largest)
T.sort(axis=0)

# Calculate MDE
MDE_q2h = MDE(
    N=N_q2h, alpha=alpha, kappa=kappa, p=p, sigma2=sd_unskilled**2, F_crit=T,
    sigma_ovr=sigma_hat
    )

# Print the results
print('\n2(h) MDE:', MDE_q2h)

################################################################################
### 2(i)
################################################################################

# Set ICC
ICC = 1/3

# Get variance of cluster-level effects
var_cls = (sd_unskilled**2) * ICC

# Get variance of individual effects
var_ind = (sd_unskilled**2) * (1 - ICC)

# Set number of small and large clusters
J_sm = 150
J_lg = 50

# Get total number of clusters
J = J_sm + J_lg

# Generate cluster-level effects
A = np.random.normal(size=(J, 1), scale=np.sqrt(var_cls))

# Generate individual-level effects (I have these be the correct mean, but you
# could just as well add the mean later, or have the village level effects be
# the correct mean)
eps = np.random.normal(size=(N_q2h, 1), loc=mean_unskilled,
                       scale=np.sqrt(var_ind))

# Generate village size indicator
# Draw random normals
V_lg = np.random.normal(size=N_q2h)

# Allocate everyone ranked at or below the number of people in large villages
# to a large village; that is, V[i] = 1 <=> i lives in a large village
V_lg = (V_lg.argsort() + 1 <= N_lg)

D = np.zeros(shape=(y.shape[0],1))

D[V_lg == 1] = 1

# Stack the village level effects for large villages (this uses the first J_lg
# village level effects)
A_lg = np.kron(A[:J_lg], np.ones(shape=(np.int(np.ceil(N_lg/J_lg)), 1)))

# Stack the village level effect for small villages (uses the remaining effects)
A_sm = np.kron(A[J_lg:], np.ones(shape=(np.int(np.ceil(N_sm/J_sm)), 1)))

# Make a stacked vector of village IDs for small villages
Vid_sm = np.array([i for i in range(J_sm)], ndmin=2).transpose()
Vid_sm = np.kron(Vid_sm, np.ones(shape=(np.int(np.ceil(N_sm/J_sm)), 1)))

# Subset it, in case the .ceil() function binds
Vid_sm = Vid_sm[:N_sm]

# Make a stacked vector of village IDs for large villages
Vid_lg = np.array([i for i in range(J_lg)], ndmin=2).transpose()
Vid_lg = np.kron(Vid_lg, np.ones(shape=(np.int(np.ceil(N_lg/J_lg)), 1)))

# Subset it, in case the .ceil() function binds
Vid_lg = Vid_lg[:N_lg]

# Stack the two
Vid = np.concatenate((Vid_sm, Vid_lg+N_sm), axis=0)

# Get unskilled outcomes
y_unskilled = eps

# Add village level effects for large villages (I subset A_lg just in case the
# .ceil() function above was binding)
y_unskilled[V_lg] = y_unskilled[V_lg] + A_lg[:N_lg]

# Add village level effects for small villages
y_unskilled[~V_lg] = y_unskilled[~V_lg] + A_sm[:N_sm]

# Generate treatment indicator
# Draw random normals
W = np.random.normal(size=N_q2h)

# Get the treatment indicators for people in small villages
W_sm = W[~V_lg]

# Get the treatment indicators for people in large villages
W_lg = W[V_lg]

# Go through all small villages
for i in range(J_sm):
    # Count how many people there are in the village
    nv = sum(Vid_sm[:,0] == i)

    # Replace the treatment indicator as 1 with probability p_sm
    W_sm[Vid_sm[:,0] == i] = (W_sm[Vid_sm[:,0] == i].argsort() + 1 <= nv * p_sm)

# Go through all large villages
for i in range(J_lg):
    # Count how many people there are in the village
    nv = sum(Vid_lg[:,0] == i)

    # Replace the treatment indicator as 1 with probability p_lg
    W_lg[Vid_lg[:,0] == i] = (W_lg[Vid_lg[:,0] == i].argsort() + 1 <= nv * p_lg)

# Get the old treatment indicator back
W[~V_lg] = W_sm
W[V_lg] = W_lg

# Make the treatment indicator into a proper vector
W = np.array(W, ndmin=2).transpose()

# Generate observed data
y = y_unskilled + beta * W

# Define a function to do one iteration of the randomization distribution
def random_t_cls(N, y, x1, V_1, Vid, Vid_0, Vid_1, J_0, J_1, N_0, N_1, p_0, p_1,
                 seed, D):
    # Set random number generator's seed
    np.random.seed(seed)

    # Calculate sample size
    N = y.shape[0]

    # Generate treatment indicator
    # Draw random normals
    W = np.random.normal(size=N)

    # Get the treatment indicators for people in small villages
    W_0 = W[~V_1]

    # Get the treatment indicators for people in large villages
    W_1 = W[V_1]

    # Go through all small villages
    for i in range(J_0):
        # Count how many people there are in the village
        nv = sum(Vid_0[:,0] == i)

        # Replace the treatment indicator as 1 with probability p_sm
        W_0[Vid_0[:,0] == i] = (W_0[Vid_0[:,0] == i].argsort() + 1 <= nv * p_0)

    # Go through all large villages
    for i in range(J_1):
        # Count how many people there are in the village
        nv = sum(Vid_1[:,0] == i)

        # Replace the treatment indicator as 1 with probability p_lg
        W_1[Vid_1[:,0] == i] = (W_1[Vid_1[:,0] == i].argsort() + 1 <= nv * p_1)

    # Get the old treatment indicator back
    W[~V_lg] = W_0
    W[V_lg] = W_1

    # Make the treatment indicator into a proper vector
    W = np.array(W, ndmin=2).transpose()

    # Generate RHS data
    X = np.concatenate((x1, W, D), axis=1)

    # Estimate regression, get coefficients and t-statistics
    b, _, t = ols(y, X, cov_est='cluster', clustvar=Vid, get_p=False)

    # Return coefficient and t-statistic for treatment dummy
    return t[1,0], b[1,0]

# Get randomization distribution
res = Parallel(n_jobs=ncores)(
    delayed(random_t_cls)
    (N=N_q2h, y=y, x1=cons, V_1=V_lg, Vid=Vid, Vid_0=Vid_sm, Vid_1=Vid_lg,
     J_0=J_sm, J_1=J_lg, N_0=N_sm, N_1=N_lg, p_0=p_sm, p_1=p_lg, seed = r, D=D)
    for r in range(R))

# Make the results into a Numpy array
res = np.array(res, ndmin=2)

# Get the randomization t-statistics, make them  into a proper (column) vector
T = np.array(res[:,0], ndmin=2).transpose()

# Get the coefficient estimates, make them  into a proper (column) vector
B = np.array(res[:,1], ndmin=2).transpose()

# Calculate their standard deviation
sigma_hat = np.std(B, axis=0)

# Sort the t-statistics (Python sorts from smallest to largest)
T.sort(axis=0)

# Calculate MDE
MDE_q2i = MDE(
    N=N_q2h, alpha=alpha, kappa=kappa, p=p, sigma2=sd_unskilled**2, F_crit=T,
    sigma_ovr=sigma_hat
    )

# Print the results
print('\n2(i) MDE:', MDE_q2i)

################################################################################
### 2(j)
################################################################################

def MDE_vr(J, n, alpha, kappa, p, rho):
    # Define the equation defining the MDE
    def MDE_eq(E):
        # Distribution for the control group
        Fc = t(df=J-2)

        # Get the critical value from that distribution
        crit_control = Fc.ppf(1-alpha/2)

        # Distribution for the treatment group
        Ft = t(df=J-2, loc=E)

        # Get the critical value for the treatment group
        crit_treatment = Ft.ppf(kappa)

        # Get sigma_hat
        sigma_hat = np.sqrt( (p * (1-p) * J)**(-1) * (rho + (1 - rho) / n) )

        # Calculate discrepancy
        delta_MDE = E - (crit_control + crit_treatment) * sigma_hat

        # Return discrepancy
        return delta_MDE

    # Calculate MDE
    MDE = fsolve(MDE_eq, .2)

    # Return MDE (note that fsolve provides this as an array, so return only the
    # first (and only) element
    return MDE[0]

# Make a list of J and n combinations
Jn = [[200, 10], [100, 20]]

# Set up a list of the results
MDE_jn = []

# Print a header for this question
print('\n2(j) MDEs for different sample sizes and clusters')

# Go through all pairs
for i, pair in enumerate(Jn):
    # Get parts of the pair
    J, n = pair

    # Calculate MDE
    MDE_jn.append(MDE_vr(J=J, n=n, alpha=alpha, kappa=kappa, p=p, rho=ICC))

    # Print the result
    print('MDE(', J, ' clusters, ', n, ' farmers): ', MDE_jn[i], sep='')

# Define a function to calculate the sample size for a given MDE (thankfully,
# this has a closed form solution)
def n_vr(J, MDE, alpha, kappa, p, rho):
    # Distribution for the control group
    Fc = t(df=J-2)

    # Get the critical value from that distribution
    crit_control = Fc.ppf(1-alpha/2)

    # Distribution for the treatment group
    Ft = t(df=J-2, loc=MDE)

    # Get the critical value for the treatment group
    crit_treatment = Ft.ppf(kappa)

    # Calculate sample size
    nstar = ((1-rho)
        / ( p * (1-p) * J * (MDE**2) * (crit_treatment+crit_control)**(-2)
        - rho ))

    # If the MDE is not feasible, this will result in a negative value. In that
    # case, set it to NaN
    if nstar < 0:
        nstar = None

    # Return sample size
    return nstar

# Print a header
print('\nSample sizes needed to achieve other MDEs')

# Go through all MDEs
for i, mde in enumerate(MDE_jn):
    # Go through all pairs of J and n combinations
    for j, pair in enumerate(Jn):
        # Unless this is the same combination as above
        if i != j:
            # Get number of clusters in the pair
            J, _ = pair

            # Calculate required sample size to match MDE
            n = n_vr(J=J, MDE=mde, alpha=alpha, kappa=kappa, p=p, rho=ICC)

            # Check whether the MDE is feasible
            if n is not None:
                # Convert the sample size to an integer
                n = np.int(np.ceil(n))

                # Print the sample size
                print('MDE of', mde, 'with', J, 'clusters requires', n,
                      'farmers')
            else:
                # Print that the MDE is not feasible
                print('MDE of', mde, 'is not feasible with', J, 'clusters')
