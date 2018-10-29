from matplotlib import pyplot as plt
from os import chdir, mkdir, path, mkdir

# Set graph options
plt.rc('font', **{'family': 'serif', 'serif': ['lmodern']})
plt.rc('text', usetex=True)

# Specify name for main directory (just uses the file's directory)
mdir = path.dirname(path.abspath(__file__)).replace('\\', '/')

# Set figures directory (doesn't need to exist)
fdir = '/figures'

# Create the figures directory if it doesn't exist
if not path.isdir(mdir+fdir):
    mkdir(mdir+fdir)

# Set up model parameters
alpha = .5
w = 1  # Normalization
beta_A = .25
beta_B = .75
L_H = 5
L_F = 45
K_H = 180
K_F = 20

# Calculate equilibrium interest rate
r = (
    (L_H + L_F) / (K_H + K_F)
    * ( alpha * beta_A + (1-alpha) * beta_B ) / ( alpha * (1-beta_A) + (1-alpha) * (1-beta_B) )
    )

# Calculate output prices
p_A = (r / beta_A)**beta_A * (1-beta_A)**(beta_A-1)
p_B = (r / beta_B)**beta_A * (1-beta_B)**(beta_B-1)

# Calculate output levels
Y_A = alpha * ( (L_H + L_F + r*(K_H + K_F)) / p_A )
Y_B = (1-alpha) * ( (L_H + L_F + r*(K_H + K_F)) / p_B )

# Calculate factor allocations
L_A = Y_A * ( r * (1-beta_A) / beta_A )**beta_A
L_B = L_H + L_F - L_A
K_A = Y_A * ( r * (1-beta_A) / beta_A )**(beta_A-1)
K_B = K_H + K_F - K_A

# Display the results
print('r =', r, 'L_A =', L_A, 'L_B =', L_B, 'K_A =', K_A, 'K_B =', K_B)

# Plot the results
# Change to figures directory
chdir(mdir+fdir)

# Set up a plot
fig, ax = plt.subplots(figsize=(15,9))

# Lower contour of the FPE set
V1 = [0, L_A, L_A + L_B]
V2 = [0, K_A, K_A + K_B]

# Upper contour of the FPE set
V3 = [0, L_B, L_A + L_B]
V4 = [0, K_B, K_A + K_B]

# Endowment vector
E = [L_H, K_H]

ax.plot(V1, V2, color='blue')
ax.plot(V3, V4, color='blue')
ax.scatter(E[0], E[1], marker='X', color='red')
ax.annotate('Initial endowment', (E[0] + .5, E[1] + .5))
ax.fill_between(V3, V4, facecolor='none', hatch='\\', edgecolor='b', interpolate=True)
ax.fill_between(V1, V2, facecolor='white', interpolate=True)

# Change axis limits
ax.set_xlim(0, L_A + L_B)
ax.set_ylim(0, K_A + K_B)

# Label the axes
ax.set_xlabel('L')
ax.set_ylabel('K', rotation=0)

# Trim unnecessary whitespace
fig.tight_layout()

# Add some more space after the horizontal axis label
ax.yaxis.labelpad = 10

# Save the plot
plt.savefig('FPE_graph.pdf')
