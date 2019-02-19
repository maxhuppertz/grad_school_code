################################################################################
### Econ 666, PS2Q2: Power calculations
################################################################################

# Import matplotlib
import matplotlib as mpl

# Select backend that does not open figures interactively (has to be done before
# pyplot is imported)
mpl.use('Agg')

# Import other necessary packages and functions
import numpy as np
import pandas as pd
from inspect import getsourcefile
import matplotlib.pyplot as plt
from os import chdir, mkdir, path
from scipy.optimize import fsolve
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
def MDE(N, alpha, kappa, p, sigma2):
    # Define the equation defining the MDE
    def MDE_eq(E):
        # Distribution for the control group
        #Fc = norm()
        Fc = t(df=p*N)

        # Distribution for the treatment group
        Ft = t(df=p*N, loc=E)

        # Calculate discrepancy
        delta_MDE = (
            E
            - (Fc.ppf(1-alpha/2) + Ft.ppf(1-kappa))
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
Nmax = 3000

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
### 2(h)
################################################################################

# Set sample size
N_q2h = 3000

# Set fraction of large villages
f_lg = .5

# Set treatment probabilities
p_sm = .7  # Small villages
p_lg = .3  # Large villages

# Generate outcome for the unskilled population
Y_unskilled = np.random.normal(size=(N_q2h, 1), scale=sd_unskilled)



#
