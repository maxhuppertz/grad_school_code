import numpy as np
from os import chdir, mkdir, path, mkdir
from linreg import ols

# Specify name for main directory (just uses the file's directory)
mdir = path.dirname(path.abspath(__file__)).replace('\\', '/')

# Change directory
chdir(mdir)
n = 100000
X = np.random.normal(size=(n,1))
y = .5*X + np.random.normal(size=(n,1)) * 5

[beta_hat,Sigma_hat] = ols(y,np.concatenate((np.ones(shape=(n,1)),X),axis=1))
print(beta_hat.flatten())
print(np.sqrt(np.diag(Sigma_hat)))
print()

[beta_hat,Sigma_hat] = ols(y,np.concatenate((np.ones(shape=(n,1)),X),axis=1),cov_est='hc1')
print(beta_hat.flatten())
print(np.sqrt(np.diag(Sigma_hat)))
print()
