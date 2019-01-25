import numpy as np
from os import chdir, mkdir, path, mkdir
from linreg import ols

# Specify name for main directory (just uses the file's directory)
mdir = path.dirname(path.abspath(__file__)).replace('\\', '/')

# Change directory
chdir(mdir)

D = np.random.normal(size=(1000,1))
Y = .5*D + np.random.normal(size=(1000,1)) * 5

[beta_hat,Sigma_hat] = ols(Y,np.concatenate((np.ones(shape=(1000,1)),D),axis=1))
print(beta_hat.flatten())
print(np.sqrt(np.diag(Sigma_hat)))
print()

[beta_hat,Sigma_hat] = ols(Y,np.concatenate((np.ones(shape=(1000,1)),D),axis=1),cov_est='hc1')
print(beta_hat.flatten())
print(np.sqrt(np.diag(Sigma_hat)))
print()
