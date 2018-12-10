# Import necessary modules
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Define PDFs
def fX(x, mu=0, sigma2=1): return norm.pdf((np.log(x) - mu)/np.sqrt(sigma2)) / x
def fY(y, mu=0, sigma2=1): return fX(y, mu, sigma2) * (1 + np.sin(2 * np.pi * np.log(y)))

# Set up values for the horizontal axis
x = np.linspace(10**(-10), 1000, 10000)

# Select parameters mu and sigma^2
mu = 5
sigma2 = 1

# Set up a plot
fig, ax = plt.subplots()

# Plot PDFs
ax.plot(x, fX(x, mu, sigma2))
ax.plot(x, fY(x, mu, sigma2))

plt.show()
