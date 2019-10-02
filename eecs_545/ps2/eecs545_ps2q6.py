################################################################################
### EECS 545, problem set 2 question 6
### Implement logistic regression using Newton-Raphson
################################################################################

################################################################################
### 1: Load packages, set directories and files, set graph options
################################################################################

# Import necessary packages
import matplotlib.pyplot as plt
import numpy as np
import os  # Only needed to set main directory
import scipy.io as sio
import time
from inspect import getsourcefile  # Only needed to set main directory

# Specify name for main directory. (This just uses the file's directory.) I used
# to use os.path.abspath(__file__), but apparently, it may be a better idea to
# use getsourcefile() instead of __file__ to make sure this runs on different
# OSs. The getsourcefile(object) function checks which file defined the object
# it is applied to. But since the object I give it is an inline function lambda,
# which was created in this file, it points to this file. The .replace() just
# ensures compatibility with Windows.
mdir = (
    os.path.dirname(os.path.abspath(getsourcefile(lambda:0))).replace('\\', '/')
    )

# Make sure I'm in the main directory
os.chdir(mdir)

# Set body fat data set (has to exist in mdir)
fn_dset = 'mnist_49_3000.mat'

# Set figures directory (doesn't have to exist)
fdir = 'figures'

fn_plot = 'misclassified_digits.pdf'

# Set graph options
plt.rc('font', **{'family': 'serif', 'serif': ['lmodern']})
plt.rc('text', usetex=True)

################################################################################
### 2: Define functions
################################################################################


# Define penalized negative logit log likelihood
def logit_pnll(theta, y, X, l=10, firstparam_free=False):
    """ Calculates the penalized negative log-likelihood for a logit model

    Inputs
    theta: d by 1 vector, parameter values
    y: n by 1 vector, labels coded as 1 or -1
    X: d by n matrix, features
    l: Scalar, penalty term
    firstparam_free: Boolean, if True, the first parameter (typically the
                     intercept) will not be penalized

    Outputs
    pnll: scalar, penalized negative log-likelihood
    """
    # Get number of features n
    d, n = X.shape

    # Calculate the linear index inside the exponential
    lin_index = -y * X.T @ theta

    # Make a vector of ones
    onevec = np.ones(shape=(1,n))

    # Set up an identity matrix for the penalty term
    I_check = np.identity(d)

    # Check whether the first parameter should be penalized
    if firstparam_free:
        # If not, set the first element of the identity matrix to zero
        I_check[0,0] = 0

    # Calculate the negative log-likelihood
    pnll = (
        onevec @ np.log(1 + np.exp(lin_index))
        + l * theta.T @ I_check @ theta
        )

    # Return the result
    return pnll[0,0]


# Define penalized negative logit log likelihood Jacobian
def logit_jacobian(theta, y, X, l=10, firstparam_free=False):
    """ Calculates the negative Jacobian for a penalized logit model

    Inputs
    theta: d by 1 vector, parameter values
    y: n by 1 vector, labels coded as 1 or -1
    X: d by n matrix, features
    l: Scalar, penalty term
    firstparam_free: Boolean, if True, the first parameter (typically the
                     intercept) will not be penalized

    Outputs
    J: d by 1 vector, Jacobian
    """
    # Get the number of features d
    d = X.shape[0]

    # Calculate the linear index inside the exponential
    lin_index = y * X.T @ theta

    # Get the largest value of the linear index
    A = np.max(lin_index)

    # Calculate the vector of scaling terms. (Subtracting A ensures that
    # numerical overflow is impossible.)
    gamma = y * np.exp(-A) / (np.exp(-A) + np.exp(lin_index - A))

    # Set up an identity matrix for the penalty term
    I_check = np.identity(d)

    # Check whether the first parameter should be penalized
    if firstparam_free:
        # If not, set the first element of the identity matrix to zero
        I_check[0,0] = 0

    # Calculate the Jacobian
    J = -X @ gamma + 2 * l * I_check @ theta

    # Return the result
    return J


# Define penalized negative logit log likelihood Hessian
def logit_hessian(theta, y, X, l=10, firstparam_free=False):
    """ Calculates the negative Hessian for a penalized logit model

    Inputs
    theta: d by 1 vector, parameter values
    y: n by 1 vector, labels coded as 1 or -1
    X: d by n matrix, features
    l: Scalar, penalty term
    firstparam_free: Boolean, if True, the first parameter (typically the
                     intercept) will not be penalized

    Outputs
    H: d by d matrix, Hessian
    """
    # Get the number of features d
    d = X.shape[0]

    # Calculate the linear index inside the exponential
    lin_index = y * X.T @ theta

    # Get the largest value of the linear index
    A = np.max(lin_index)

    # Calculate the diagonal weights matrix. (Subtracting A ensures that
    # numerical overflow is impossible.)
    C = np.diag(
        np.exp(lin_index[:,0] - 2*A)
        / (np.exp(-A) + np.exp(lin_index[:,0] - A))**2
    )

    # Set up an identity matrix for the penalty term
    I_check = np.identity(d)

    # Check whether the first parameter should be penalized
    if firstparam_free:
        # If not, set the first element of the identity matrix to zero
        I_check[0,0] = 0

    # Calculate the Hessian
    H = X @ C @ X.T + 2 * l * I_check

    # Return the result
    return H


# Define a function to implement the Newton-Raphson algorithm
def newton_raphson(theta, args, objfun=logit_pnll, jacobian=logit_jacobian,
                   hessian=logit_hessian, stepsi=1, tol_incr=10**(-16),
                   tol_foc=10**(-14), itmax=100, usedamp=False, itmaxdamp=100,
                   alpha = .5, beta=.9):
    """ Implements the Newton-Raphson algorithm to minimize a function

    Inputs
    theta: d by 1 vector, initial guess for parameters to be estimated
    args: List, additional arguments to be passed to objfun, jacobian, and
          hessian
    objfun: Function, objective function to be minimized
    jacobian: Function, Jacobian of the objective function
    hessian: Function, Hessian of the objective function
    stepsi: Scalar, default step size for the Newton update (multiplier on the
            inverse Hessian times Jacobian term in the update)
    tol_incr: Scalar, tolerance for step size increase termination criterion. If
              the next iteration's parameter vector is at most tol_incr away
              from the current iteration's parameters in the infinity norm, the
              algorithm counts that as convergence.
    tol_foc: Scalar, tolerance for Jacobian size termination criterion. If the
             Jacobian evaluated at the updated parameter value is closer than
             tol_foc in the infinity norm, the algorithm counts that as
             convergence.
    itmax: Scalar, maximum number of iterations
    usedamp: Boolean, indicates whether to use step size dampening. If at any
             iteration, the updated parameter vector would increase objfun
             (rather than decrease it), dampening parameters alpha and beta are
             used to modulate the step size (see below)
    itmaxdamp: Scalar, maximum number of step size dampenings in case of
               overshooting
    alpha: Scalar, step size dampening parameter. If at any iteration, the
           updated parameter vector would increase objfun (rather than decrease
           it), a new update using stepsi=alpha is calculated instead.
    beta: Scalar, iterative step size dampening parameter. If after using
          stepsi=alpha, the update parameter vector would still increase objfun,
          a new update using stepsi=alpha*beta is calculated. If that still
          increases objfun, stepsi=alpha*beta**2 is used, and so forth.

    Outputs
    theta: d by 1 vector, maximum likelihood estimate of the parameters
    stopcrit: list, stopping criterion. The first element is a scalar, and the
              second a string explaining the stopping criterion.
    """
    # Set up an iteration counter
    it = 0

    # Set up a flag for convergence based on increment size
    converged_incr = False

    # Set up a flag for convergence based on the infinity norm of the Jacobian
    converged_foc = False

    # Set up a flag for being unable to prevent overshooting
    overshooting = False

    # Iterate until convergence, or until overshooting cannot be prevented, or
    # the maximum number of iterations has been reached
    while not (converged_incr or converged_foc or overshooting) and it < itmax:
        # Save last iteration's parameter vector
        theta_old = theta

        # Calculate the Jacobian
        J = jacobian(theta_old, *args)

        # Calculate the inverse Hessian
        H_inv = np.linalg.inv(hessian(theta_old, *args))

        # Get this iteration's parameter vector
        theta = theta_old - stepsi * H_inv @ J

        # Set up a counter for the number of dampening iterations. (These may
        # not be necessary at all, see immediately below.)
        dampit = 0

        # Get the initial dampening step size. (Again, this will only be used if
        # dampening is needed to prevent overshooting, see immediately below.)
        t = alpha * stepsi

        # Check for overshooting (i.e. see whether the objective function at the
        # current iteration's parameter value is larger than at the preceding
        # iteration's parameter value). If it occurs, do this until overshooting
        # no longer occurs, or until the maximum number of dampening steps has
        # been reached
        while (usedamp and objfun(theta, *args) > objfun(theta_old, *args)
               and dampit < itmaxdamp):
            # Recalculate theta using the reduced step size
            theta = theta_old - t * H_inv @ J

            # Dampen the step size even further
            t = beta * t

            # Count the dampening iterations
            dampit = dampit + 1

        # Check whether overshooting could not be prevented using the dampening
        # step loop
        if dampit >= itmaxdamp:
            # If so, set theta to the last iteration's value, since this
            # iteration's value would increase the objective function
            theta = theta_old

            # Set the overshooting flag to True
            overshooting = True
        elif np.max(np.abs(jacobian(theta, *args))) <= tol_foc:
            # Otherwise, if the Jacobian is within tolerance, set that flag
            # to True
            converged_foc = True
        elif np.max(np.abs(theta - theta_old)) <= tol_incr and dampit == 0:
            # Or, if the increment is within tolerance, and dampening steps were
            # not used, set that flag to True
            converged_incr = True

        # Increase the iteration counter
        it = it + 1

    # Check whether convergence based on the FOC occured
    if converged_foc:
        # If so, set the stopping criterion to that
        stopcrit = [0, 'Converged (FOC size)']
    elif converged_incr:
        # If instead, increment convergence occured, set the stopping criterion
        # accordingly
        stopcrit = [1, 'Converged (increment size)']
    elif overshooting:
        # If instead, overshooting could not be prevented, set the stopping
        # criterion to that
        stopcrit = [2, 'Could not prevent overshooting']
    elif it >= itmax:
        # Otherwise, set the stopping criterion to reflect the maximum number of
        # iterations has been exceeded
        stopcrit = [3, 'Exceeded maximum number of iterations']

    # Return latest parameter estimates and stopping criterion
    return theta, stopcrit


# Define logit prediction function (logit classifier)
def logit_predict(theta, X, y=None):
    """ Gets predicted values (classifications) for a logit classifier

    Inputs
    theta: d by 1 vector, parameters of the model
    X: d by n matrix, features on which the prediction is based
    y: n by 1 vector, true labels of those features (optional)

    Outputs
    y_hat: n by 1 vector, predicted labels (classification)
    conf: n by 1 vector, confidence of the classifier. Equal to the predicted
          probability of being labeled 1 if an instance is classified as 1, and
          one minus that probability otherwise
    loss: scalar, average 0-1 loss (only returned if y was provided)
    """
    # Get the number of instances n
    n = X.shape[1]

    # Get the probabilities that an instance has label 1
    prob = np.exp(X_te.T @ theta) / (1 + np.exp(X_te.T @ theta))

    # Get the predicted labels, by assigning 1 if that probability is greater
    # than or equal to .5, and assigning -1 otherwise
    y_hat = 2 * (prob >= .5) - 1

    # Get the classifier's confidence, which is the probability of being label 1
    # if an instance is classified as 1, and one minus that probability if it is
    # classified as -1
    conf = prob * (prob >= .5) + (1-prob) * (prob < .5)

    # Check whether labels were provided
    if y is not None:
        # If so, calculate the average 0-1 loss
        loss = np.not_equal(y, y_hat).sum() / n

        # Return the results
        return y_hat, conf, loss
    else:
        # Return the results
        return y_hat, conf

################################################################################
### 3: Load data
################################################################################

# Load the data set
data = sio.loadmat(fn_dset)

# Get responses and features
y = data['y']
X = data['x']

# Get the number of instances n
n = X.shape[1]

# Add an intercept to the features
X = np.concatenate((np.ones(shape=(1,n)), X), axis=0)

d = X.shape[0]

# Following the course convention, make sure y is n by 1
y = y.T

# Set number of training data instances (uses the first ntrain instances)
ntrain = 2000

# Split data into training and test sample
X_tr = X[:, 0:ntrain]
y_tr = y[0:ntrain, :]
X_te = X[:, ntrain:]
y_te = y[ntrain:, :]

# Get the number of test instances
n_te = y_te.shape[0]

################################################################################
### 4: Compute the MLE
################################################################################

# Set up an initial guess for the parameter vector
theta0 = np.zeros(shape=(d,1))

# Set penalty term lambda
l = 10

# Combine training labels, training features, the penalty term, and a flag
# indicating whether the intercept should not be penalized. (That is, if the
# flag is False, the intercept will be penalized.)
args = [y_tr, X_tr, l, False]

# Mark the time
start = time.time()

# Get estimates and predictions (classification)
theta_hat, sc = newton_raphson(theta0, args)
y_hat, conf, loss = logit_predict(theta_hat, X_te, y_te)

# Record the time
dur = np.round(time.time() - start, 4)

# Print the results
print('\nIntercept penalized')
print('Time:', dur, 'seconds')
print('Stopping criterion:', sc)
print('PNLL at theta0:', np.round(logit_pnll(theta0, *args), 4))
print('PNLL at theta_hat', np.round(logit_pnll(theta_hat, *args), 4))
print('Average 0-1 loss:', np.round(loss, 4))

# Make a similar list of parameters, this time for a model which does not
# penalize the intercept
args = [y_tr, X_tr, l, True]

# Mark the time
start = time.time()

# Get estimates and predictions (classification)
theta_hat_np, sc_np = newton_raphson(theta0, args)
y_hat_np, _, loss_np = logit_predict(theta_hat_np, X_te, y_te)

# Record the time
dur = np.round(time.time() - start, 4)

# Print the results
print('\nIntercept not penalized')
print('Time:', dur, 'seconds')
print('Stopping criterion:', sc_np)
print('PNLL at theta0:', np.round(logit_pnll(theta0, *args), 4))
print('PNLL at theta_hat', np.round(logit_pnll(theta_hat_np, *args), 4))
print('Average 0-1 loss:', np.round(loss_np, 4))

################################################################################
### 5: Plot most confidently misclassified images, according to penalized
###    intercept model
################################################################################

# Create the figures directory if it doesn't exist
if not os.path.isdir(mdir+'/'+fdir):
    os.mkdir(mdir+'/'+fdir)

# Change to the figures directory
os.chdir(mdir+'/'+fdir)

# Set up an index vector
idx = np.array(np.arange(n_te), ndmin=2).T

# Combine the index vector with prediction confidence values
confidx = np.concatenate((idx, conf), axis=1)

# Figure out which images were misclassified
mistake = np.not_equal(y_hat, y_te)

# Get the confidence values and indices of all misclassified images
confidx = confidx[mistake[:,0], :]

# Sort them by confidence level (this sorts in ascending order)
confidx = confidx[np.argsort(confidx[:,1]), :]

# Reverse the sort, so the most confidently misclassified images are at the top
confidx = confidx[::-1,:]

# Set up an array of subplots
fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(6.5, 6.5))

# Go through all subplots in the figure
for i, ax in enumerate(fig.axes):
    # Get the index of the i-th most confidently classified image
    imidx = confidx[i,0].astype(int)

    # Get the predicted label for that image
    imhat = y_hat[imidx,0]

    # Get the true label for that image
    imtrue = y_te[imidx,0]

    # Get the confidence value for that image
    imconf = confidx[i,1]

    # Make a plot title, based on those three elements
    plot_ti = (
        '$y_i$: {}, '.format(imtrue)
        + r'$\hat{y}_i$' + ': {}, '.format(imhat)
        + '\n$c_i$: {:2.4f}\%'.format(imconf * 100)
        )

    # Get the image itself
    img = np.reshape(X_te[1:,imidx], (int(np.sqrt(d)), int(np.sqrt(d))))

    # Plot the image
    ax.imshow(img, cmap='Greys')

    # Remove x and y ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Add a plot to the figure
    ax.set_title(plot_ti, fontsize=11)

# Get rid of unnecessary whitespace
fig.tight_layout()

# Save and close the figure (the bbox_inches='tight' helps remove even more
# unwanted whitespace)
plt.savefig(fn_plot, bbox_inches='tight')
plt.close()
