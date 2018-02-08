import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
from numpy.linalg import inv
from os import chdir, mkdir, path
from scipy.stats import uniform, norm
from sklearn.neighbors import KernelDensity

# PS3Q3: OLS III

# Set main directory
mdir = path.dirname(path.abspath(__file__)).replace('\\', '/')

# Set figures directory
fdir = '/figures/'

# Create that directory if it doesn't exist
if not path.isdir(mdir+fdir):
    mkdir(mdir+fdir)

# Change to figures directory
chdir(mdir+fdir)

# Set random generator seed
np.random.seed(seed=8675309)

# Set model parameters
beta = np.array([.4, .9], ndmin=2).transpose()  # True betas
N = [100, 500, 1000]  # Different sample sizes to consider
imax = 10000  # Number of iterations per sample size

Z = np.zeros(shape=(imax, len(N)))

# Go through different sample sizes
for j, n in enumerate(N):
    # Go through all iterations for given sample size (it would be way more efficient to do this with a matrix instead
    # of a loop, but it's not worth it optimizing this I think)
    for i in range(imax):
        # Generate X and U
        X = np.column_stack((np.ones(shape=(n, 1)), uniform().rvs(size=(n, 1)) + 1))
        U = norm(loc=0, scale=2).rvs(size=(n, 1))

        # Generate y
        y = X @ beta + U

        # Calculate OLS estimate of beta_hat
        beta_hat = inv(X.transpose() @ X) @ X.transpose() @ y

        # Calculate Z_n and save it
        Z[i, j] = beta_hat[0, 0] / beta_hat[1, 0]

    # Center and divide by standard deviation
    Z[:, j] = np.sqrt(n) * (Z[:, j] - (beta[0, 0] / beta[1, 0])) / np.std(np.sqrt(n) * Z[:, j])

# Plot the distribution of Z_n for different sample sizes
# Some plot parameters
margin = 1  # Page margins
figratio = 6/9  # Default ratio for plots

# Set up plot of distribution
fig, ax = plt.subplots(figsize=(8.5 - margin, (8.5 - margin) * figratio))

# Grid over which to estimate kernel density (the kernel density estimator seems to like column vectors)
x_grid = np.array(np.linspace(start=-5, stop=5, num=2000), ndmin=2).transpose()

# Bandwidths for kernel density estimation (this is super hacky; it is meant to illustrate how well the approximation
# works, but probably in large part just reflects how kernel density estimation is pretty sensitive to the bandwidth
# selected when running it and so you can show anything using it if you don't think too hard about it... looks nice
# though!)
bandwidths = [.1, .6, .4]

# Line styles for plot, so they alternate (unnecessary to use a cycle, since there are only four lines, but I just came
# across this tool and wanted to play around with it)
ls = cycle([":","-.","--","-"])

# Plot estimate of f(Z_n) for each n
for j, (n, bw) in enumerate(zip(N, bandwidths)):
    # Fit kernel density estimator
    kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(np.array(Z[:, j], ndmin=2).transpose())

    # Retrieve kernel density estimate for the plot
    kde_plot = np.exp(kde.score_samples(x_grid))

    # Plot the estimate
    ax.plot(x_grid, kde_plot, linestyle=next(ls), alpha=0.9, lw=0.9, label='$n = '+str(n)+'$', )

# Plot an N(0, 1) variable for comparison
ax.plot(x_grid, norm.pdf(x_grid), linestyle=next(ls), alpha=0.9, lw=0.9, label='$\mathcal{N}(0,1)$')

# Insert a title, set axis limits, and display a legend
fig.suptitle('Kernel density estimates of $Z_n$')
ax.set_xlabel('$Z_n$')
ax.set_ylabel('$f(Z_n)$', labelpad=25).set_rotation(0)
ax.set_xlim(-3, 3)
ax.set_ylim(-.01, 1)
ax.legend()

# Save the plot
plt.savefig('Z_density.pdf', bbox_inches='tight')

# Display the plot (why not?)
plt.show()
