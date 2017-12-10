import numpy as np
import scipy as sp
from numba import jit
from scipy.stats import uniform


# Define a function to iterate over value functions (unfortunately, some of the Numpy methods in here cause Numba's
# JIT compilation to fail; then again, Numpy is probably faster than the alternatives at those operations, so maybe
# it's not obviously a speed loss? It probably is though)
def v_iter(r, b, u, P, A, Y, v_0, tol=.001, i_max_v=1000, get_g=True):
    # Take the n x 1 Y vector, make an m x n matrix which has one income in each column
    # S = np.tile(Y.transpose(), (A.shape[0], 1))

    # Flatten that matrix to an m*n x 1 vector
    # S = np.array(np.tile(Y.transpose(), (A.shape[0], 1)).flatten(order='F'), ndmin=2).transpose()

    # Repeat that vector m times, to create an m*n x m matrix of incomes
    # The previous two steps and the next stept are now all done in one move, to preserve memory; the old code is still
    # in the comments for ease of understanding
    S = np.tile(np.array(np.tile(Y.transpose(), (A.shape[0], 1)).flatten(order='F'), ndmin=2).transpose(),
                (1, A.shape[0]))

    # Make an m*n x m matrix of assets, for the initial assets people enter a period with; these have to be equal across
    # columns, so it just repeats the array as-is
    B = np.tile(A, (Y.shape[0], A.shape[0]))

    # Make a vector like that for next period's assets, which have to be equal within columns (hence the transpose)
    B_prime = np.tile(A.transpose(), (A.shape[0] * Y.shape[0], 1))

    # Calculate in-period utility (which will be m*n x m also)
    U = u((1 + r) * B + S - B_prime)

    # Set first input value function to v_0
    # Note that everything is set up so the (i, j) element of v refers to income i and asset choice j, so v is n x m
    v_in = v_0

    # Calculate next-period continuation values, also as an m*n x m vector
    # W = np.repeat(P @ v_in, A.shape[0], axis=0)

    # Add the utilities to the continuation values, pick the maximum element for each row
    # This resulty in an m*n x 1 vector of new values for the value function
    # v_out = np.array(np.amax(U + b*W, axis=1), ndmin=2).transpose()

    # Reshape that vector into an n x m matrix, which is the new value function
    # This does all three in one step
    v_out = np.reshape(np.array(np.amax(U + b*np.repeat(P @ v_in, A.shape[0], axis=0),
                       axis=1), ndmin=2).transpose(), v_0.shape)

    # Count the iterations, to be able to stop if this takes too long to converge
    i = 1
    while np.abs(np.amax(v_out - v_in)) > tol and i <= i_max_v:
        v_in = v_out

        # Calculate next-period continuation values, also as an m*n x m vector
        # W = np.repeat(P @ v_in, A.shape[0], axis=0)

        # Get the new value function
        # v_out = np.array(np.amax(U + b*W, axis=1), ndmin=2).transpose()

        # Restack it
        # This does all three in one step
        # When iterating over excess demand later, increased precision might be helpful here
        v_out = np.reshape(np.array(np.amax(U + b*np.repeat(P @ v_in, A.shape[0], axis=0),
                           axis=1), ndmin=2).transpose(), v_0.shape).astype(np.longdouble)

        # If this is close enough, get the optimal policy function
        if np.abs(np.amax(v_out - v_in)) <= tol and get_g:
            # g = np.array(np.argmax(U + b*W, axis=1), ndmin=2).transpose()
            # g = A[g]

            # This does the previous two operations in one step, and restacks the result
            g = np.reshape(A[np.array(np.argmax(U + b*np.repeat(P @ v_in, A.shape[0], axis=0), axis=1),
                           ndmin=2).transpose()], v_0.shape)

        # Increase iteration counter
        i += 1

    # If the maximum number of iterations was reached, print an error message
    if i > i_max_v:
        print('Value function failed to converge after', i_max_v, 'iterations', '\n',
              'Deviation:', abs(np.amax(v_out - v_in)))

    # Return the value function and, if desired, policy function
    if get_g:
        return v_out, g
    else:
        return v_out


# Define a function to find the stationary (a, s) distribution (luckily, this works with JIT)
@jit
def find_L(v, g, A, Y, P, X):
    # Set up the transition matrix across (a, s) (note that Scipy's eigenvector algorithm doesn't accept high precision
    # arrays (float64 / np.longdouble), so there isn't much that can be done here with precision, I think)
    P_X = np.zeros(shape=(X.shape[0], X.shape[0]))

    # Go through all elements of that transition matrix (obviously this is the major bottleneck of the whole script)
    for ij in np.ndindex(P_X.shape):
        # Fill in each element by checking whether the asset value it's mapping to is a' given (a, s), and multiplying
        # that by the probability of ending up in s'
        P_X[ij] = (g[X[ij[0]][0], X[ij[0]][1]] == A[X[ij[1]][1], 0]) * P[X[ij[0]][0], X[ij[1]][0]]

    # Calculate ergodic distribution of transition matrix
    # First get the eigenvectors, then check which one is associated with the unit eigenvalue
    # Checking which eigenvalue is 1 is funky because of floating point issues
    L = sp.linalg.eig(P_X.transpose())[1][:, np.abs(np.linalg.eig(P_X.transpose())[0] - 1) <= 10**(-10)]

    # Divide by the sum of the eigenvalues, since this is a probability distribution
    L /= L.sum(axis=0)

    # Print a warning if the stationary distribution is not unique (usually that means m is too small, I believe)
    # Also stop the program, because this'll really mess things up once we start looking at excess demand and all that
    if L.shape[1] > 1:
        print('Warning: There is more than one stationary distribution across the (a, s) space', '\n',
              'Stopping program')
        exit()

    # Reshape L so that the (i, j) element refers to state i and asset choice j
    L = np.reshape(L, g.shape)

    # Return the stationary distribution across (a, s) tuples
    return L

# Define CRRA utility
def crra(c, g=4, u_inada=-10**10):
    # Set up utility matrix (when iterating over excess demand later, higher precision might be helpful here)
    u = np.zeros_like(c, dtype=np.longdouble)

    # Calculate utlity for positive consumption values
    u[c!=0] = (c[c!=0]**(1 - g) - 1) / (1 - g)

    # To almost preserve Inada conditions, replace utility for 0 and negative consumption with a large negative number
    u[c<=0] = u_inada

    # Return utility matrix
    return u

# Set beta
b = .96

# Set lower and upper limits for r (these better contain the equilibrium interest rate)
r_l = -5
r_h = 5

# Make an initial guess for r
r = (r_h + r_l) / 2

# Create a space of incomes
n = 10  # Number of different values of income
y_1 = .1  # Lowest possible income
y_n = 1  # Highest possible income
Y = np.array(np.linspace(y_1, y_n, num=n), ndmin=2).transpose()  # Column vector of incomes

# Create a space of asset holdings
m = 80  # Number of asset choices
phi = y_1 / (1 - b)  # Borrowing limit; y_1 / (1 - b) is a 'natural' limit in the Ljungqvist & Sargent sense
a_m = 4  # Highest possible asset value
A = np.array(np.linspace(-phi, a_m, num=m).transpose(), ndmin=2).transpose()  # Column vector of asset choices

# Set seed for random variables
np.random.seed(seed=8675309)

# Create a transition matrix with strictly positive transition propabilites
# This is totally random; a more structured setup would probably generate more interesting results?
P = uniform().rvs(size=(n, n))

# This is how the transition probabilites are always positive
while np.count_nonzero(P) - n**2 != 0:
    P = uniform().rvs(size=(n, n))

# Make sure the rows of P sum up to one
P *= np.sum(P, axis=1, keepdims=True)**(-1)

# Make an initial guess for the value function
v_0 = uniform().rvs(size=(n, m)) * 30

# Find the value and policy functions
v, g = v_iter(r=r, b=b, u=crra, P=P, A=A, Y=Y, v_0=v_0)

# Make an n*m x 2 matrix of the indices of all (a, s) combinations in their respective source vectors
# Note that the first column is the index of s, the second column that of a
# So X[i, :] = (j(s_j), l(a_l))
# The reason this doesn't happen in the find_L function is that Numba doesn't currently support list creation, and it
# seemed that having this X thing outside the function was a small price to pay for JIT compilation
X = np.array([ij for ij in np.ndindex(Y.shape[0], A.shape[0])])

# Find the (a, s) distribution lambda
L = find_L(v=v, g=g, A=A, Y=Y, P=P, X=X)

# Calculate excess demand
e = (L * g).sum()

# Set a tolerance level for excess demand
tol_e = .001

# Set up a maximum number of iterations and a counter for those iterations
i_max_e = 100
i = 1

# Throughout this loop there will be two values for the interest rate, that calculated during the preceding iteration
# (r_1), and the new value (r_2)
r_1 = r_2 = r

# If e isn't zero (and it most likely won't be), repeat the process until it is within tolerance, unless the interest
# rate stops converging because it hits its precision maximum, in which case reiterating this loop makes no sense
# since excess demand will not continue to converge
while np.abs(e) > tol_e and (np.longdouble(r_1 - r_2) != 0 or i == 1) and i <= i_max_e:
    # Update interest rate from preceding iteration
    r_1 = r_2

    # Update interest rate bounds in accordance with excess demand
    if e < 0:
        r_l = np.longdouble(r_1)
    else:
        r_h = np.longdouble(r_1)

    # Update interest rate (using higher precision gives you a lot more digits to play with untill excess demand stops
    # getting closer to zero)
    r_2 = np.longdouble((r_h + r_l) / 2)

    # Find the value and policy functions
    v, g = v_iter(r=r_2, b=b, u=crra, P=P, A=A, Y=Y, v_0=v_0)

    # Create the X vector
    X = np.array([ij for ij in np.ndindex(Y.shape[0], A.shape[0])])

    # Find lambda
    L = find_L(v=v, g=g, A=A, Y=Y, P=P, X=X)

    # Calculate excess demand
    e = (L * g).sum()

    # If iterations exceed the maximum, print the remaining excess demand
    if i >= i_max_e:
        print('Excess demand failed to converge after', i_max_e, 'iterations', '\n',
              'Deviation:', e, '\n', 'Interest rate:', r_2)

    # If the interest rate reaches its precision boundary, print its current value
    if np.longdouble(r_1 - r_2) == 0:
        print('Note: Interest rate reached maximum precision at', r_2, '\n', 'Remaining excess demand:', e)

    # Increase iteration counter
    i += 1

# Print the equilibrium interest rate, in case the loop before didn't stop prematurely
if i <= i_max_e and np.longdouble(r_1 - r_2) != 0:
    print('Equilibrium interest rate:', r_2)
