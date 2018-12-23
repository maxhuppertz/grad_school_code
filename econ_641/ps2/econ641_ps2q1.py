########################################################################################################################
### ECON 641: PS2, Q1
### Investigates characteristics of the firm size distribution
########################################################################################################################

# Import necessary packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import pinv
from os import chdir, mkdir, path, mkdir
from requests import get

# pandas_datareader has some issues with pandas sometimes
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import wb

########################################################################################################################
### Part 1: Define Gabaix and Ibragimov (2011) estimator
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
    V_hat = -beta_hat[1,0] / np.sqrt(n/2)

    # Return size coefficient and GI variance/covariance matrix
    return -beta_hat[1,0], V_hat

########################################################################################################################
### Part 2: Get data
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

# Specify which CPI data to get from the World Bank, and a file name for a local copy
cpi_data = 'FP.CPI.TOTL'  # Data to get
cpi_file = 'US_CPI'  # File name for local copy

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

    # Specify the name of the year variable
    v_year = 'fyear'

    # Get CPI data for the years covered in the CompuStat data
    wb_data = wb.download(indicator=cpi_data, country='US',
                          start=int(min(data[v_year])), end=int(max(data[v_year])), errors='ignore')

    # Drop the country level from the index, since it's unnecessary
    wb_data.index = wb_data.index.droplevel('country')

    # Save the DataFrame locally
    wb_data.to_pickle(cpi_file+'.pkl')
else:
    # Read in the locally saved DataFrame
    data = pd.read_pickle(data_file+'.pkl')

    # Specify the name of the year variable
    v_year = 'fyear'

    # Read in the World Bank CPI data
    wb_data = pd.read_pickle(cpi_file+'.pkl')

########################################################################################################################
### Part 2: Adjust for inflation
########################################################################################################################

# Specify name of sales variable
v_sales = 'sale'

# Select a year to which to rescale the CPI date
rescale_year = 2015

# Rescale the CPI data (this also converts it to ratios rather than percentages)
wb_data.loc[:, cpi_data] = wb_data.loc[:, cpi_data] / wb_data.loc[str(rescale_year), cpi_data]

# Go through all years in the data
for year in range(int(min(data[v_year])), int(max(data[v_year])+1)):
    # Adjust for inflation
    data.loc[data[v_year] == year, v_sales] = data.loc[data[v_year] == year, v_sales] * wb_data.loc[str(year), cpi_data]

########################################################################################################################
### Part 3: Estimate rank - size relationship, for different rank cutoffs
########################################################################################################################

# Generate log sales data as NaN
v_log_sales = 'log_'+v_sales
data[v_log_sales] = np.nan

# Where sales are not zero, replace them with the log of sales
data.loc[data[v_sales] > 0, v_log_sales] = np.log(data.loc[data[v_sales] > 0, v_sales])

# Generate firms size rank, by year
v_sales_rank = v_sales + '_rank'
data[v_sales_rank] = data.groupby(v_year)[v_sales].rank(ascending=False)

# Generate log rank, with a scaling factor s, i.e. generate log(rank - s)
v_log_sales_rank = 'log_' + v_sales_rank
s = .5
data[v_log_sales_rank] = np.log(data[v_sales_rank] - s)

# Set minimum and maximum year for the estimation
year_min = -np.inf
year_max = 2015

# Select rank cutoffs for the estimation
rank_cutoffs = [np.inf, 500, 100]

est_results = pd.DataFrame(np.zeros(shape=(len(rank_cutoffs), 3)),
    columns=['Rank cutoff', 'beta_hat', 'SE beta_hat'])

# Go through all cutoffs
for i, c in enumerate(rank_cutoffs):
    # Run the estimation, using only firms which are below the rank cutoffs and only data for selected years
    beta_hat_OLS_GI, V_hat_OLS_GI = OLS_GI(
        data.loc[(data[v_sales_rank] <= c) & (year_min <= data[v_year]) & (data[v_year] <= year_max), v_log_sales_rank],
        data.loc[(data[v_sales_rank] <= c) & (year_min <= data[v_year]) & (data[v_year] <= year_max), v_log_sales])

    # Save cutoff and associated results
    est_results.loc[i, :] = [c, beta_hat_OLS_GI, V_hat_OLS_GI]

# Display estimation results
print(est_results)
