################################################################################
### Econ 666, grant proposal: Power calculations
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
from inspect import getsourcefile
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from os import chdir, mkdir, path
from scipy.stats import beta

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

# Specify number of control villages
V_c = 30

# Specify number of villages in the 'low' treatment
V_lo = 30

# Specify number of villages in the 'high' treatment
V_hi = 30

# Calculate total number of villages
J = V_c + V_lo + V_hi

# Specify number of households sampled per village
n = 10

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

# Specify mean treatment group adoption rates
mu_T = larry([0, .2, .3])

# Stack them to a vector for each individual
mu_T = mu_T[I_T[:,0], :]

# Set up distribution for mean village level adoption rates
alpha_v = 1
beta_v = 100 - alpha_v
F_v = beta(alpha_v, beta_v)

# Get the expected village level adoption rate
E_v = F_v.stats(moments='m')

# Get village level adoption rates
mu_v = larry(F_v.rvs(size=J))

# Stack them to a vector for each individual
mu_v = mu_v[I_v[:,0], :]

# Set up distribution for individual level adoption rates
alpha_i = 2
beta_i = 100 - alpha_i
F_i = beta(alpha_i, beta_i)

# Get the expected individual adoption rate
E_i = F_i.stats(moments='m')

# Get individual level adoption rates
mu_i = larry(F_i.rvs(size=N))

# Get combined adoption rate
mu = np.amin([mu_v + mu_T + mu_i, np.ones(shape=(N,1))], axis=0)

# Calculate expected adoption rate in the control group
adr_ctr = E_i + E_v + np.amin(mu_T)

# Display it
print()  # Looks nicer with an empty line preceding it
print('Expected adoption rate in the control group: ' + str(adr_ctr))

# Calculate implied ICC
ICC = E_v * (1 - E_v) / (E_i * (1 - E_i) + E_v * (1 - E_v))

# Display it
print()
print('ICC: ' + str(ICC))

# Draw adoption rates, as the minimum of 1 and a sum of three Bernoulli random
# variables (individual level, village level, and treatment level)
y = np.amin(
    [np.random.binomial(1, mu_i, size=(N,1))
     + np.random.binomial(1, mu_v, size=(N,1))
     + np.random.binomial(1, mu_T, size=(N,1)),
     np.ones(shape=(N,1))], axis=0)

# Make a set of treatment dummies (control is ommitted)
D = np.zeros(shape=(N,2))

for i in range(D.shape[1]):
    D[:,i] = (I_T[:,0] >= i+1)

# Specify on which coefficients to impose the null
imp0 = larry([0, 1, 1])

# Add an intercept
Z = np.concatenate((np.ones(shape=(N,1)), D), axis=1)

beta_hat, t_hat, CI = boot_ols(y, Z, clustvar=I_v, B=4999, imp0=imp0)

print(np.concatenate((beta_hat, t_hat, CI), axis=1))
#
