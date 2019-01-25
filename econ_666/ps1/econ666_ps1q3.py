import numpy as np
from os import chdir, mkdir, path, mkdir
from myest import ols

# Specify name for main directory (just uses the file's directory)
mdir = path.dirname(path.abspath(__file__)).replace('\\', '/')

# Change directory
chdir(mdir)

X = np.random.normal(size=(1000,1))
y = .5*X + np.random.normal(size=(1000,1))

[beta_hat,Sigma_hat] = ols(y,X,cov_est='hc1')
print(beta_hat, np.sqrt(Sigma_hat))

[beta_hat,Sigma_hat] = ols(y,X,cov_est='hskd')
print(beta_hat, np.sqrt(Sigma_hat))
