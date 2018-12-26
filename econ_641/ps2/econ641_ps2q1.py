########################################################################################################################
### ECON 641: PS2, Q1
### Investigates characteristics of the firm size distribution
########################################################################################################################

# Import necessary packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from numpy.linalg import pinv
from os import chdir, mkdir, path, mkdir
from requests import get

# pandas_datareader has some issues with pandas sometimes
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import wb

# Some of the logs and divisions will raise warnings, which are obvious and not necessary
warnings.simplefilter("ignore")

########################################################################################################################
### Part 1: Define Gabaix and Ibragimov (2011) estimator and standard OLS
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

# Set up a function which does standard OLS regression, with Eicker-Huber-White (EHW) variance/covariance estimator
# (HC1, i.e. it has the n / (n - k) correction)
def OLS(y_input, X_input, get_cov=True):
    # Check which rows contains entirely non-NaN values
    try:
        use_rows = np.logical_and(np.isfinite(y_input), np.isfinite(X_input.sum(axis=1)))
    except:
        use_rows = np.logical_and(np.isfinite(y_input), np.isfinite(X_input))

    # Get number of observations n
    n = y_input[use_rows].shape[0]

    # This just makes sure you can feed the function any kind of vector, or list; the try part of this statement will
    # fail if the size data are not a vector (i.e. not two dimensional)
    try:
        # Check whether this is a row vector
        if X_input.shape[0] < X_input.shape[1]:
            # If so, transpose it
            X_1 = X_input[use_rows].transpose
        else:
            # Otherwise, leave it as it is
            X_1 = X_input[use_rows]
    # If the statement fails...
    except IndexError:
        # ...make it into a vector
        X_1 = np.array(X_input[use_rows], ndmin=2).transpose()

    # Same thing, but for the rank data; here, I also need to transform the data from y into y - s
    try:
        # Check whether this is a row vector
        if y_input.shape[0] < y_input.shape[1]:
            # If so, transpose it
            y = y_input[use_rows].transpose
        else:
            # Otherwise, leave it as it is
            y = y_input[use_rows]
    except IndexError:
        # If the first part fails, make it into a vector
        y = np.array(y_input[use_rows], ndmin=2).transpose()

    # Set up X matrix
    X = np.concatenate((np.ones(shape=(n, 1)), X_1), axis=1)

    # Get number of coefficients k
    k = X.shape[1]

    # Calculate OLS coefficients (pinv() uses the normal matrix inverse if X'X is invertible, and the Moore-Penrose
    # pseudo-inverse otherwise; invertibility of X'X can be an issue with the pairs bootstrap if sample sizes are small,
    # which is why this is helpful)
    beta_hat = pinv(X.transpose() @ X) @ (X.transpose() @ y)

    # Check whether covariance is needed
    if get_cov:
        # Get residuals
        U_hat = y - X @ beta_hat

        # Calculate component of middle part of EHW sandwich (S_i = X_i u_i, meaning that it's easy to calculate
        # sum_i X_i X_i' u_i^2 = S'S)
        S = X * ( U_hat @ np.ones(shape=(1,k)) )

        # Calculate EHW variance/covariance matrix
        V_hat = ( n / (n - k) ) * pinv(X.transpose() @ X) @ (S.transpose() @ S) @ pinv(X.transpose() @ X)

        # Return coefficients and EHW variance/covariance matrix
        return beta_hat, V_hat
    else:
        # Otherwise, just return coefficients
        return beta_hat

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
### Part 3: Adjust for inflation
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
### Part 4: Estimate log rank - log sales relationship, for different rank cutoffs
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
year_min = 2015
year_max = 2015

# Specify firm name variable
v_name = 'conm'

# Check how many firms there are in the data for the years under consideration
n_firms = len(data.loc[(year_min <= data[v_year]) & (data[v_year] <= year_max), v_name].unique())

# Specify percentile cutoffs for the estimation
perc_cutoffs = [np.inf, .5, .3, .2, .1]

# Make a list of the respective ranks in the firm size distribution
rank_cutoffs = [np.floor(p * n_firms) for p in perc_cutoffs]

# Set up a DataFrame for the estimation results
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
print('Sales: Log size - log rank estimation, full sample')
print(est_results)

# Add sectors to the data set
v_sic = 'sic'  # Variable containing SIC codes
v_sector = 'sector'  # Variable containing sectors, i.e. one digit SIC codes
data[v_sector] = np.floor(data[v_sic] / 1000)

# Generate firms size rank, by year and sector
v_sales_rank_sector = v_sales_rank + '_sector'
data[v_sales_rank_sector] = data.groupby([v_year, v_sector])[v_sales].rank(ascending=False)

# Go through all sectors
for sector in sorted(data[v_sector].unique()):
    # Check how many firms there are in the data for the years under consideration
    n_firms = len(data.loc[(year_min <= data[v_year]) & (data[v_year] <= year_max) &
        (data[v_sector] == sector), v_name].unique())

    # Make a list of the respective ranks in the firm size distribution
    rank_cutoffs = [np.floor(p * n_firms) for p in perc_cutoffs]

    # Set up a DataFrame for the estimation results
    est_results = pd.DataFrame(np.zeros(shape=(len(rank_cutoffs), 3)),
        columns=['Rank cutoff', 'beta_hat', 'SE beta_hat'])

    # Go through all cutoffs
    for i, c in enumerate(rank_cutoffs):
        # Run the estimation, using only firms which are below the rank cutoffs and only data for selected years
        beta_hat_OLS_GI, V_hat_OLS_GI = OLS_GI(
            data.loc[(data[v_sales_rank_sector] <= c) & (year_min <= data[v_year]) & (data[v_year] <= year_max) &
            (data[v_sector] == sector), v_log_sales_rank],
            data.loc[(data[v_sales_rank_sector] <= c) & (year_min <= data[v_year]) & (data[v_year] <= year_max) &
            (data[v_sector] == sector), v_log_sales])

        # Save cutoff and associated results
        est_results.loc[i, :] = [c, beta_hat_OLS_GI, V_hat_OLS_GI]

    # Display estimation results
    print('\n')
    print('Sales: Log size - log rank estimation, SIC', int(sector))
    print(est_results)

########################################################################################################################
### Part 5: Estimate log rank - log employment relationship, for different rank cutoffs
########################################################################################################################

# Generate log employment data as NaN
v_emp = 'emp'
v_log_emp = 'log_' + v_emp
data[v_log_emp] = np.nan

# Where employment is not zero, replace it with the log of employment
data.loc[data[v_emp] > 0, v_log_emp] = np.log(data.loc[data[v_emp] > 0, v_emp])

# Generate firms employment rank, by year
v_emp_rank = v_emp + '_rank'
data[v_emp_rank] = data.groupby(v_year)[v_emp].rank(ascending=False)

# Generate log rank, with a scaling factor s, i.e. generate log(rank - s)
v_log_emp_rank = 'log_' + v_emp_rank
s = .5
data[v_log_emp_rank] = np.log(data[v_emp_rank] - s)

# Check how many firms there are in the data for the years under consideration
n_firms = len(data.loc[(year_min <= data[v_year]) & (data[v_year] <= year_max), v_name].unique())

# Make a list of the respective ranks in the firm size distribution
rank_cutoffs = [np.floor(p * n_firms) for p in perc_cutoffs]

# Set up a DataFrame for the estimation results
est_results = pd.DataFrame(np.zeros(shape=(len(rank_cutoffs), 3)),
    columns=['Rank cutoff', 'beta_hat', 'SE beta_hat'])

# Go through all cutoffs
for i, c in enumerate(rank_cutoffs):
    # Run the estimation, using only firms which are below the rank cutoffs and only data for selected years
    beta_hat_OLS_GI, V_hat_OLS_GI = OLS_GI(
        data.loc[(data[v_emp_rank] <= c) & (year_min <= data[v_year]) & (data[v_year] <= year_max), v_log_emp_rank],
        data.loc[(data[v_emp_rank] <= c) & (year_min <= data[v_year]) & (data[v_year] <= year_max), v_log_emp])

    # Save cutoff and associated results
    est_results.loc[i, :] = [c, beta_hat_OLS_GI, V_hat_OLS_GI]

# Display estimation results
print('\n')
print('Employment: Log size - log rank estimation')
print(est_results)

# Generate firms size rank, by year and sector
v_emp_rank_sector = v_emp_rank + '_sector'
data[v_emp_rank_sector] = data.groupby([v_year, v_sector])[v_emp].rank(ascending=False)

# Go through all sectors
for sector in sorted(data[v_sector].unique()):
    # Check how many firms there are in the data for the years under consideration
    n_firms = len(data.loc[(year_min <= data[v_year]) & (data[v_year] <= year_max) &
        (data[v_sector] == sector), v_name].unique())

    # Make a list of the respective ranks in the firm size distribution
    rank_cutoffs = [np.floor(p * n_firms) for p in perc_cutoffs]

    # Set up a DataFrame for the estimation results
    est_results = pd.DataFrame(np.zeros(shape=(len(rank_cutoffs), 3)),
        columns=['Rank cutoff', 'beta_hat', 'SE beta_hat'])

    # Go through all cutoffs
    for i, c in enumerate(rank_cutoffs):
        # Run the estimation, using only firms which are below the rank cutoffs and only data for selected years
        beta_hat_OLS_GI, V_hat_OLS_GI = OLS_GI(
            data.loc[(data[v_emp_rank_sector] <= c) & (year_min <= data[v_year]) & (data[v_year] <= year_max) &
            (data[v_sector] == sector), v_log_emp_rank],
            data.loc[(data[v_emp_rank_sector] <= c) & (year_min <= data[v_year]) & (data[v_year] <= year_max) &
            (data[v_sector] == sector), v_log_emp])

        # Save cutoff and associated results
        est_results.loc[i, :] = [c, beta_hat_OLS_GI, V_hat_OLS_GI]

    # Display estimation results
    print('\n')
    print('Employment: Log size - log rank estimation, SIC', int(sector))
    print(est_results)

########################################################################################################################
### Part 6: Size-volatility relationship
########################################################################################################################

# Sort data by firm, and by year within firm
data = data.sort_values([v_name, v_year])

# Calculate sales growth rates
# Specify sales growth variable
v_sales_growth = v_sales + '_growth'

# Get the difference in sales from year to year
data[v_sales_growth] = data.groupby(v_name)[v_sales].diff()  # .pct_change() has some weird issues with groupby()

# Divide by the previous year's sales
data.loc[1:, v_sales_growth] = (
    data.loc[1:, v_sales_growth].values / data.loc[0:data[v_sales_growth].shape[0] - 2, v_sales].values
    )

# Calculate the standard deviation of sales growth, within firms, and use it to start a collapsed data set
# Specify the log standard deviation of sales growth variable
v_log_sales_growth_sd = 'log_' + v_sales_growth + '_sd'

# Calculate the standard deviation of sales growth, within firm, and add it to a collapsed data set
collapsed_data = np.log(pd.DataFrame(data.groupby(v_name)[v_sales_growth].std()))

# Rename the variable
collapsed_data = collapsed_data.rename(index=str, columns={v_sales_growth: v_log_sales_growth_sd})

# Add log average sales to the data set
v_log_mean_sales = 'log_mean_' + v_sales  # Mean sales variable
collapsed_data[v_log_mean_sales] = np.log(data.groupby(v_name)[v_sales].mean())

# Set up a DataFrame for the estimation results
est_results = pd.DataFrame(np.zeros(shape=(5, 5)),
    columns=['Period', 'theta_hat (no FE)', 'SE theta_hat', 'theta_hat (FE)', 'SE theta_hat'])

# Regress log sales growth standard deviation on log mean sales
theta_hat, V_hat_theta = OLS(collapsed_data[v_log_sales_growth_sd], np.array(collapsed_data[v_log_mean_sales]))

# Add the results to the results data frame
est_results.iloc[0, :3] = ['Full sample', -theta_hat[1,0], np.sqrt(V_hat_theta[1,1])]

# Add sector fixed effects
# Add sectors to the data set
collapsed_data[v_sector] = np.floor(data.groupby(v_name)[v_sic].max() / 1000)

# Set up a list of sector variables
sector_vars = []

# Go through all sectors
for sector in sorted(collapsed_data[v_sector].unique()):
    # Omit lowest SIC code
    if sector != min(collapsed_data[v_sector].unique()):
        # Add a dummy for that sector to the data set
        collapsed_data[v_sector + '_' + str(sector)] = (collapsed_data[v_sector] == sector).astype(int)

        # Add the new variable to the data set
        sector_vars.append(v_sector + '_' + str(sector))

# Run the same regression as before, but adding sector fixed effects
theta_hat, V_hat_theta = OLS(collapsed_data[v_log_sales_growth_sd],
    np.array(collapsed_data.loc[:, [v_log_mean_sales] + sector_vars]))

# Add the results to the results data frame
est_results.iloc[0, 3:] = [-theta_hat[1,0], np.sqrt(V_hat_theta[1,1])]

# Redo the estimation for each decade in the data set
# Go through all decades
for i, decade in enumerate(range(1980, 2020, 10)):
    # Generate the collapsed data set for this decade, starting with log sales growth standard deviation
    collapsed_data = np.log(pd.DataFrame(
        data.loc[(decade <= data[v_year]) & (data[v_year] <= (decade + 9)), :].groupby(v_name)[v_sales_growth].std()))

    # Rename the variable
    collapsed_data = collapsed_data.rename(index=str, columns={v_sales_growth: v_log_sales_growth_sd})

    # Add log mean sales for this decade
    collapsed_data[v_log_mean_sales] = np.log(
        data.loc[(decade <= data[v_year]) & (data[v_year] <= (decade + 9)), :].groupby(v_name)[v_sales].mean())

    # Run the regression
    theta_hat, V_hat_theta = OLS(collapsed_data[v_log_sales_growth_sd], np.array(collapsed_data[v_log_mean_sales]))

    # Add the results to the results data frame
    est_results.iloc[i+1, :3] = [str(decade) + ' - ' + str(decade+9), -theta_hat[1,0], np.sqrt(V_hat_theta[1,1])]

    # Add sectors to the data set
    collapsed_data[v_sector] = np.floor(
        data.loc[(decade <= data[v_year]) & (data[v_year] <= (decade + 9)), :].groupby(v_name)[v_sic].max() / 1000)

    # Set up a list of sector variables
    sector_vars = []

    # Go through all sectors
    for sector in sorted(collapsed_data[v_sector].unique()):
        # Omit lowest SIC code
        if sector != min(collapsed_data[v_sector].unique()):
            # Add a dummy for that sector to the data set
            collapsed_data[v_sector + '_' + str(sector)] = (collapsed_data[v_sector] == sector).astype(int)

            # Add the new variable to the data set
            sector_vars.append(v_sector + '_' + str(sector))

    # Run the same regression as before, but adding sector fixed effects
    theta_hat, V_hat_theta = OLS(collapsed_data[v_log_sales_growth_sd],
        np.array(collapsed_data.loc[:, [v_log_mean_sales] + sector_vars]))

    # Add the results to the results data frame
    est_results.iloc[i+1, 3:] = [-theta_hat[1,0], np.sqrt(V_hat_theta[1,1])]

# Tell pandas to display more columns
pd.set_option('display.max_columns', 5)

# Display the results
print('\n')
print('Log sales growth SD - log mean sales estimation')
print(est_results)
