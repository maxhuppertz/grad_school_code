import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from os import mkdir, path
from scipy.stats import chi2, uniform, f, norm


# Define a cd function that gives a warning if the chosen directory doesn't exist (and aborts the program)
# This is overly cautious, since this thing finds its own directory and creates its figures directory in there
# But why not?
def cd(path):
    from os import chdir
    try:
        chdir(path)
    except WindowsError or OSError:
        print('The directory: ', path, '\n', 'does not exist', '\n', 'Aborting program', sep='')
        exit()


# Define a function to iterate over value functions
def V_iter(V_0, wages, beta, T):
    # Set up V matrix to store value functions
    V = np.zeros((n, T+1))
    V[:, 0] = V_0

    # Set up W matrix to store reservation wages
    W = np.zeros(T)

    # Loop over all time periods
    for t in range(1, T+1):
        # Calculate the new value function
        V_t = np.maximum(c + beta * np.dot(f_w, V[:, t-1]), wages / (1 - beta))

        # Store it in the V matrix
        V[:, t] = V_t

        # Calculate the reservation wage
        W[t-1] = wages[np.argmin(np.abs((wages / (1 - beta)) - (c + beta * np.dot(f_w, V[:, t-1]))))]

    # Return the value function and wage matrices
    return V, W

# Set plot options
# plt.rc('text', usetex=True)  # Use LaTeX to compile text, which looks way better but also takes longer
plt.rc('font', size=11, **{'family':'serif',
       'serif':['lmodern', 'Palatino Linotype', 'DejaVu Serif']})  # Font size and type
margin = 1  # Page margins
figratio = 6/9  # Default ratio for plots
col_main = '#0000CD'  # Main color used for plot lines

# Set main directory
mdir = path.dirname(path.abspath(__file__)).replace('\\', '/')

# Set figures directory
fdir = '/figures/'

# Create that directory if it doesn't exist
if not path.isdir(mdir+fdir):
    mkdir(mdir+fdir)

# Set up wages
B = 10**6  # Maximum wage offer
n = 10**4  # Number of wages in the vector
wages = np.linspace(1, B, num=n)  # Vector of n wages, between 0 and B

# Set up probability density over wages
# w_dist = norm()
# w_dist = chi2(df=3)
# w_dist = uniform()
w_dist = f(dfn=500, dfd=10)  # Underlying distribution function
w_pctiles = np.linspace(10**(-10), 1 - 10**(-10), num=n)  # Percentiles for which a probability will be calculated
f_w = w_dist.pdf(w_dist.ppf(w_pctiles))  # value of the pdf for each percentile
f_w = f_w / sum(f_w)  # Discretized pdf (i.e. this sums to one)

# Set up discount factor
beta = .97

# Set up value of unemployment
c = 12000

# Set up value function initial guess
V_0 = np.sort(f.rvs(dfn=2, dfd=80, size=n))*wages

# Specify number of iterations
T = 100

# Get value functions and reservations wages over time
V, W = V_iter(V_0, wages, beta, T)

# Change to figures directory
cd(mdir + fdir)

# Set up value function plot
fig, ax = plt.subplots(figsize=(8.5 - margin, (8.5 - margin) * figratio))

# Plot the initial guess
ax.plot(wages, V_0, color=col_main, linestyle='-.', alpha=0.9, lw=0.9, label='$V_0(w)$')

# For the first period, graph V_t and add a label to the graph
ax.plot(wages, V[:, 1], color=col_main, linestyle='-', alpha=0.4, lw=0.4, label='$V_t(w)$')

# Otherwise, just graph it (unless its's the last iteration, which will get a special treatment below)
ax.plot(wages, V[:, 2:-1], color=col_main, linestyle='-', alpha=0.4, lw=0.4)

# Plot the last iteration with a label
ax.plot(wages, V[:, -1], color=col_main, linestyle='-', alpha=0.9, lw=0.9, label='$V_T(w)$')

# Include a legend and title, set axis limits and labels
fig.suptitle(r'Value function iteration: $\beta = '+str(beta)+', \: c = \: $'+format(c, ',')+
             '$, \: w \sim \mathcal{N}$')
ax.legend()
ax.set_xlabel('$w$')
ax.set_ylabel('$V$', labelpad=20).set_rotation(0)
ax.set_xlim(min(wages), max(wages))

# Do some magic to get the x axis values formatted with a thousands separator
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))

# Save the plot (the bbox_inches argument prevents labels being cut off)
plt.savefig('V_t.pdf', bbox_inches='tight')

# Set up plot of reservation wages
fig, ax = plt.subplots(figsize=(8.5 - margin, (8.5 - margin) * figratio))

# Plot reservation wages
ax.plot(np.linspace(1, T, T), W, color=col_main, linestyle='-', alpha=0.9, lw=0.9)

# Insert a title, axis limits and labels
fig.suptitle(r'$\bar{w}$ over time: $\beta = '+str(beta)+', \: c = \: $'+format(c, ',')+'$, \: w \sim \mathcal{N}$')
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$\bar{w}$', labelpad=20).set_rotation(0)
ax.set_xlim(1, T)
ax.set_ylim(0)

# Do some magic to get the x axis values formatted with a thousands separator
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))

# Save this plot as well
plt.savefig('w_bar.pdf', bbox_inches='tight')

# Set up plot of wage distribution
fig, ax = plt.subplots(figsize=(8.5 - margin, (8.5 - margin) * figratio))

# Plot wage density
ax.plot(wages, f_w, color=col_main, linestyle='-', alpha=0.9, lw=0.9)

# Insert a title, axis limits and labels
fig.suptitle('Wage distribution (pdf)')
ax.set_xlabel('$w$')
ax.set_ylabel('$f_w(w)$', labelpad=25).set_rotation(0)
ax.set_xlim(min(wages), max(wages))
ax.set_ylim(0)

# Do some magic to get the x axis values formatted with a thousands separator
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))

# Save this plot as well
plt.savefig('f_w.pdf', bbox_inches='tight')
