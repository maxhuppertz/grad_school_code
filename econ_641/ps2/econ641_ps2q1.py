########################################################################################################################
### ECON 641: PS2, Q1
### Investigates parts of the firm size distribution
########################################################################################################################

# Import necessary packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import pinv
from os import chdir, mkdir, path, mkdir
from requests import get

########################################################################################################################
### Part 1: Gabaix and Ibragimov (2011) estimator
########################################################################################################################

# Set up a function which does standard OLS regression, but reports the Gabaix and Ibragimov (2011) (GI) standard error;
# expects to get data on log rank, minus a scaling factor, and data on log size
def OLS_GI(rank, size, s=.5):
    # Check which rows contains entirely non-NaN values
    use_rows = np.logical_and(np.isfinite(rank), np.isfinite(size))

    # Get number of observations n
    n = rank[use_rows].shape[0]

    # This just makes sure you can feed the function any kind of vector, or list; the try part of this statement will
    # fail if the size data are not a vector (i.e. not two dimensional)
    try:
        # Check whether this is a row vector
        if size.shape[0] < size.shape[1]:
            # If so, transpose it
            X_1 = size[use_rows].transpose
        else:
            # Otherwise, leave it as it is
            X_1 = size[use_rows]
    # If the statement fails...
    except IndexError:
        # ...make it into a vector
        X_1 = np.array(size[use_rows], ndmin=2).transpose()

    # Same thing, but for the rank data; here, I also need to transform the data from y into y - s
    try:
        # Check whether this is a row vector
        if rank.shape[0] < rank.shape[1]:
            # If so, transpose it
            y = rank[use_rows].transpose
        else:
            # Otherwise, leave it as it is
            y = rank[use_rows]
    except IndexError:
        # If the first part fails, make it into a vector
        y = np.array(rank[use_rows], ndmin=2).transpose()

    # Set up X matrix
    X = np.concatenate((np.ones(shape=(n, 1)), X_1), axis=1)

    # Calculate OLS coefficients
    beta_hat = pinv(X.transpose() @ X) @ (X.transpose() @ y)

    # Calculate GI standard errors for the slope coefficient
    V_hat = beta_hat[1,0] / np.sqrt(n/2)

    # Return size coefficient and GI variance/covariance matrix
    return beta_hat[1,0], V_hat

########################################################################################################################
### Part 1: Get data
########################################################################################################################

# Set graph options
plt.rc('font', **{'family': 'serif', 'serif': ['lmodern']})
plt.rc('text', usetex=True)

# Specify name for main directory (just uses the file's directory)
mdir = path.dirname(path.abspath(__file__)).replace('\\', '/')

# Set data directory (doesn't need to exist)
ddir = '/data'

# Set figures directory (same deal)
fdir = '/figures'

# Create the data and figures directories if they don't exist
for dir in [ddir, fdir]:
    if not path.isdir(mdir+dir):
        mkdir(mdir+dir)

# Specify whether to download the data
download_data = False

# Specify name of main data file, plus extension. If you chose to download the data, the program will create this, as
# well as a .pkl version, which is the preferred file format for pandas. If you chose not to download the data, the
# data directory and that .pkl file need to exist!
data_file = 'PanelAnnual_compustat1980_2015'
data_file_ext = '.dta'

# Change directory to data
chdir(mdir+ddir)

# Check whether to download data
if download_data:
    # Specify CompuStat data URL, as well as which data set to download
    compustat_url = 'https://www.dropbox.com/s/rcujpfsm9c4z7r8/'
    compustat_file = 'PanelAnnual_compustat1980_2015.dta?dl=1'

    # Access that spreadshett and save it locally
    web_file = get(compustat_url+compustat_file, stream=True)  # Access file on server
    with open(data_file+data_file_ext, 'wb') as local_file:  # Open local file
        for chunk in web_file.iter_content(chunk_size=128):  # Go through contents on server
            local_file.write(chunk)  # Write to local file

    # Read the downloaded spreadsheet into a DataFrame
    data = pd.read_stata(data_file+data_file_ext)

    # Save the DataFrame locally
    data.to_pickle(data_file+'.pkl')
else:
    # Read in the locally saved DataFrame
    data = pd.read_pickle(data_file+'.pkl')

# Specify the name of the year variable
v_year = 'fyear'

# Generate log sales
# Specify name of sales variable and log sales variable
v_sales = 'sale'
v_log_sales = 'log_'+v_sales

# Check where sales are not zero
non_zero_sales = data[v_sales] > 0

# Generate sales data as NaN
data[v_log_sales] = np.nan

# Where sales are not zero, replace them with the log
data.loc[non_zero_sales, v_log_sales] = np.log(data.loc[non_zero_sales, v_sales])

# Generate firms size rank, by year
v_sales_rank = v_sales + '_rank'
data[v_sales_rank] = data.groupby(v_year)[v_sales].rank() + 1

# Generate log rank, with a scaling factor s, i.e. generate log(rank - s)
v_log_sales_rank = 'log_' + v_sales_rank
s = .5
data[v_log_sales_rank] = np.log(data[v_sales_rank] - s)

# Run OLS of log sales on log size
beta_hat_OLS_GI, V_hat_OLS_GI = OLS_GI(data[v_log_sales_rank], data[v_log_sales])
print(beta_hat_OLS_GI, V_hat_OLS_GI)
