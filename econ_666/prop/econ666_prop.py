################################################################################
### Econ 666, grant proposal: Power calculations
################################################################################

################################################################################
### Part 1: Setup
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
import time
import warnings
from inspect import getsourcefile
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from os import chdir, mkdir, path
from scipy.optimize import fsolve, minimize, NonlinearConstraint
from scipy.stats import beta, t

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
from linreg import larry, boot_ols

################################################################################
### 1.1: Directories, graph options, display options, seed
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

# Set number of digits to round to
nround = 4

# Set random number generator seed
np.random.seed(666)

################################################################################
### Part 2: Optimal sample sizes
################################################################################

################################################################################
### 2.1: Define necessary functions
################################################################################

# Define a function that calculate the MDE
def MDE(J, n, alpha, kappa, p, rho, sigma2_i):
    # Specify the equation defining the MDE
    def MDE_eq(E):
        # Distribution for the control group
        Fc = t(df=J-2)

        # Get the critical value from that distribution
        crit_control = Fc.ppf(1-alpha/2)

        # Distribution for the treatment group
        Ft = t(df=J-2, loc=E)

        # Get the critical value for the treatment group
        crit_treatment = Ft.ppf(kappa)

        # Calculate implied variance of cluster level adoption effect
        tau2 = (rho / (1 - rho)) * sigma2_i

        # Get sigma_hat
        sigma_hat = np.sqrt( (p * (1-p) * J)**(-1) * (tau2 + sigma2_i / n) )

        # Calculate discrepancy
        delta_MDE = E - (crit_control + crit_treatment) * sigma_hat

        # Return discrepancy
        return delta_MDE

    # Calculate MDE
    MDE = fsolve(MDE_eq, .2)

    # Return MDE (note that fsolve provides this as an array, so return only the
    # first (and only) element
    return MDE[0]

# Define a function to calculate the vector of sample sizes needed to do three
# tests, a) treatment 1 vs. control, b) treatment 2 vs. control, and c)
# treatment 2 vs. treatment 1, at level alpha (two sided) and with power kappa
# against the targeted (true) vector of MDEs MDE_bar
def N_mult(MDE_bar, n, alpha, kappa, rho, sigma2_i, C, W, tol):
    # Get number of treatment groups
    ngroup = len(C)

    # Get number of effects to be calculated
    neff = len(MDE_bar)

    # Define a function that calculates how much it costs to sample a given
    # vector of sample sizes
    def cost_N(J):
        return sum([J[i] * C[i] for i in range(ngroup)])

    # Define a function that calculates implied treatment probabilities for a
    # given set of cluster allocations and gets the resulting MDEs
    def MDE_J(J):
        # Calculate implied treatment probabilities
        P = [J[1]/(J[0]+J[1]), J[2]/(J[0]+J[2]), J[2]/(J[1]+J[2])]

        # Calculate MDEs
        MDEs = [
            MDE(J[0]+J[1], n, alpha, kappa[0], P[0], rho, sigma2_i),  # T1 vs. C
            MDE(J[0]+J[2], n, alpha, kappa[1], P[1], rho, sigma2_i),  # T2 vs. C
            MDE(J[1]+J[2], n, alpha, kappa[2], P[2], rho, sigma2_i)  # T2 vs. T1
            ]

        # Return the MDEs
        return MDEs

    # Calculate (weighted) targeted MDEs
    MDE_target = [MDE_bar[i]/W[i] for i in range(neff)]

    # Set up nonlinear constraints on the MDEs. These say that each MDE has to
    # be weakly smaller than the corresponding target value, that is, between
    # negative infinity and the target values.
    const = NonlinearConstraint(MDE_J, [-np.inf for mde in MDE_bar], MDE_target)

    # Specify bounds to ensure positive sample sizes
    bounds_J = ((0, np.inf), (0, np.inf), (0, np.inf))

    # Calculate the optimal sample sizes by minimizing the total sample size
    # constrained by having to get each MDE at most as large as its target
    Jstar = minimize(cost_N,
                     x0=[30, 30, 30], bounds=bounds_J, constraints=const,
                     tol=tol).x

    # Make sure these are all rounded up integers
    Jstar = [np.int(np.ceil(n)) for n in Jstar]

    # Return the result
    return Jstar

################################################################################
### 2.2: Find optimal (least cost) sample size
################################################################################

# Specify vector of targeted (assumed to be true) MDEs
tau_1 = .05  # 'Low' treatment effect
tau_2 = .1  # 'High' treatment effect
MDE_bar = [tau_1, tau_2, tau_2 - tau_1]

# Specify intracluster correlation (ICC)
rho = .35

# Specify variance of individual level adoption effect
sigma2_i = .1**2

# Specify number of households sampled per village
n = 10

# Specify level of tests to be performed
alpha = .05

# Specify vector of power levels for each test
kappa = [.8, .8, .8]

# Specify a vector of cost per unit sampled/treated. The format is
# [control, 'low' treatment, 'high' treament]
C = [100, 100 + 200 * tau_1, 100 + 500 * tau_2]

# Specify a vector of importance weights for each test. Higher means more
# important. They way these work is that I inflate each MDE by 1/W[i], so
# a weight of one means targe the MDE, a weight of 1/2 means target twice
# the MDE (which allows for a lower sample size), and so on.
W = [1, 1, 1]

# Set a tolerance for the optimization routine
tol = 10**(-14)

# Record the time this started
time_start = time.time()

# Sometimes this raises divide by zero issues, which I want to ignore
with warnings.catch_warnings():
    # This makes sure warnings are ignored
    warnings.simplefilter("ignore")

    # Calculate sample sizes to get these effects as MDEs, if I want to do an
    # alpha level (two sided) test and have power kappa
    Jstar = N_mult(MDE_bar=MDE_bar, n=n, alpha=alpha, kappa=kappa, rho=rho,
                   sigma2_i=sigma2_i, C=C, W=W, tol=tol)

# Record the time this was done
time_end = time.time()

# Calculate how long the simulation took to run
duration = np.around(time_end - time_start, nround)

# Display the results
print()
print('Optimal sample sizes:')
print('Control group:', Jstar[0], 'villages')
print("'Low' treatment:", Jstar[1], 'villages')
print("'High' treatment:", Jstar[2], 'villages')
print('Time elapsed:', duration, 'seconds')

################################################################################
### Part 3: Power simulation
################################################################################

################################################################################
### 3.1: Specify DGP
################################################################################

# Specify number of control villages
V_c = Jstar[0]

# Specify number of villages in the 'low' treatment
V_lo = Jstar[1]

# Specify number of villages in the 'high' treatment
V_hi = Jstar[2]

# Calculate total number of villages
J = V_c + V_lo + V_hi

# Calculate total sample size (number of households across all villages)
N = J * n

# Make a vector of village indicators
I_v = larry([x for x in range(J)])

# Stack this for all individuals
I_v = np.kron(I_v, np.ones(shape=(n,1)))

# Make sure these village indicators are stored as integers
I_v = I_v.astype(int)

# Set up a vector of village level treatment assignments
I_T = np.zeros(shape=(J,1))

# Replace treatment assignment for 'low' treatment
I_T[V_c:V_c+V_lo,:] = 1

# Replace treatment assignment for 'high' treatment
I_T[V_c+V_lo:,:] = 2

# Stack this for all individuals
I_T = I_T[I_v[:,0], :]

# Make sure these assignments are stored as integers
I_T = I_T.astype(int)

# Specify mean adoption rates coming from treatment status
mu_T = larry([0, tau_1, tau_2])

# Stack them to a vector for each individual
mu_T = mu_T[I_T[:,0], :]

# Set up distribution for mean village level adoption rates
alpha_v = 5
beta_v = 100 - alpha_v
F_v = beta(alpha_v, beta_v)

# Get the expected village level adoption rate
E_v = F_v.stats(moments='m')

# Set up distribution for individual level adoption rates
alpha_i = 10
beta_i = 100 - alpha_i
F_i = beta(alpha_i, beta_i)

# Get the expected individual adoption rate
E_i = F_i.stats(moments='m')

# Calculate expected adoption rate in the control group
adr_ctr = E_i + E_v + np.amin(mu_T)

# Display it
print()  # Looks nicer with an empty line preceding it
print('DGP statistics:')
print('Expected individual level effect:', E_i)
print('Expected village level effect:', E_v)
print('Expected adoption rate in the control group:', adr_ctr)

# Calculate implied ICC
ICC = E_v * (1 - E_v) / (E_i * (1 - E_i) + E_v * (1 - E_v))

# Display it
print('Implied ICC:', np.around(ICC, nround))

################################################################################
### 3.2: Run power simulation
################################################################################

# Make a set of treatment dummies (control is ommitted)
D = np.zeros(shape=(N,2))  # Set up as array of zeros

# Go through all treatments
for i in range(D.shape[1]):
    # Add the dummy
    D[:,i] = (I_T[:,0] >= i+1)

# Specify on which coefficients to impose the null
imp0 = larry([0, 1, 1])

# Add an intercept
X = np.concatenate((np.ones(shape=(N,1)), D), axis=1)

# Specify number of bootstrap iterations to use for standard errors
B = 1999

# Specify number of simulations to use for power calculation
S = 100

# Specify whether to run simulations in parallel
parsim = True

# Define one iteration of the power calculation
def power_iter(s, N=N, F_v=F_v, F_i=F_i, mu_T=mu_T, I_v=I_v, imp0=imp0, B=B,
               seed=0, parbs=(not parsim)):
    # Set random number generator seed
    np.random.seed(seed+s)

    # Get village level adoption rates
    mu_v = larry(F_v.rvs(size=J))

    # Stack them to a vector for each individual
    mu_v = mu_v[I_v[:,0], :]

    # Get individual level adoption rates
    mu_i = larry(F_i.rvs(size=N))

    # Draw adoption rates, as the minimum of 1 and a sum of three Bernoulli
    # random variables (individual level, village level, and treatment level)
    y = np.amin(
        [np.random.binomial(1, mu_i, size=(N,1))
         + np.random.binomial(1, mu_v, size=(N,1))
         + np.random.binomial(1, mu_T, size=(N,1)),
         np.ones(shape=(N,1))], axis=0)

    # Use Cameron, Gelbach, and Miller (2008) cluster robust bootstrap with the
    # null imposed to get point estimates, t-statistics, and confidence
    # intervals (do not use parallel computing if the simulation is run in
    # parallel already)
    beta_hat, t_hat, CI = boot_ols(y, X, alg='cgm0', B=B, clustvar=I_v,
                                   imp0=imp0, par=parbs)

    # Get rejection decision, by checking whether the treatment coefficients'
    # t-statistics are outside of the confidence intervals calculated under the
    # null
    r = (t_hat[:,0] < CI[:,0]) | (CI[:,1] < t_hat[:,0])

    # Return it
    return larry(r)

# Record the time this started
time_start = time.time()

# Check whether to run this in parallel
if parsim:
    # If so, get number of available cores
    ncores = cpu_count()

    # Go through all simulations in parallel, save rejection rates
    R = Parallel(n_jobs=ncores)(
        delayed(power_iter)(s, seed=666) for s in range(S))
else:
    # Go through all simulations in sequence, save rejection rates
    R = [power_iter(s) for s in range(S)]

# Record the time this was done
time_end = time.time()

# Calculate how long the simulation took to run
duration = np.around(time_end - time_start, nround)

# Make these into a proper matrix (this will be [e,S], which means I can take
# the mean along the second dimension later)
R = np.concatenate(R, axis=1)

# Get simulated power as rate at which true null was rejected across simulations
kappa_hat = larry(R.mean(axis=1))

# Print resulting simulated power levels
print()
print('Simulated power:')
print('Any treatment vs. control:', kappa_hat[1,0])
print("'High' treatment vs. 'low' treatment:", kappa_hat[2,0])
print('Time elapsed:', duration, 'seconds')
