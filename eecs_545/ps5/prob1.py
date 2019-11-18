################################################################################
### EECS 545, problem set 5 question 1
### Bayesian Naïve Bayes classifier
################################################################################

################################################################################
### 1: Load packages, set directories and files, set graph options
################################################################################

# Import necessary packages
import numpy as np
import os  # Only needed to set main directory
from eecs545_ps5funcs import bnb
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

################################################################################
### 2: Evaluate Bayesian naïve Bayes classifier
################################################################################

# Load the training data, enforcing the convention that X is d by n
train_features = np.load('spam_train_features.npy').T
train_labels = np.load('spam_train_labels.npy')

# Load the test data, enforcing the convention that X is d by n
test_features = np.load('spam_test_features.npy').T
test_labels = np.load('spam_test_labels.npy')

# Calculate naïve Bayes classifier, get the error rate
_, err = bnb(X_tr=train_features, y_tr=train_labels,
             X_te=test_features, y_te=test_labels)

# Display the result
print()
print('Error rate: {:1.4f}'.format(err))
