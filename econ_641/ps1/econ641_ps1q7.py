########################################################################################################################
### ECON 641: PS1, Q7
### Q7.1: Calculates trade shares based on the World Input Output Database (WIOD)
###       Also provides some other international trade information along the way
### Q7.2: Uses a basic EK model to simulate the effect of a trade cost reduction on real wages
########################################################################################################################

########################################################################################################################
### Q7.1: WIOD data
########################################################################################################################

# Import necessary packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from requests import get
from os import chdir, mkdir, path, mkdir
from scipy.linalg import lstsq as ols

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

# Specify name of main data file, plus extension (doesn't need to exist, if you want to download it again)
data_file = 'wiot00_row_apr12'
data_file_ext = '.xlsx'

# Set up name of trade shares file, plus extension (the program creates this)
trade_shares_file = 'wiot00_trade_shares'
trade_shares_file_ext = '.xlsx'

# Specify names for WIOD data index levels
ind_icode = 'industry_code'
ind_iname_fgood = 'industry_name_or_final_good'
ind_country = 'country'
ind_c = 'c_num'

# Make a list of the original index order, as well as a reordered index
data_index_orig = [ind_icode, ind_iname_fgood, ind_country, ind_c]
data_index_reorder = [ind_country, ind_c, ind_iname_fgood, ind_icode]

# Change directory to data
chdir(mdir+ddir)

# Check whether to download data
if download_data:
    # Specify WIOD URL, as well as which spreadsheet to download
    wiod_url = 'http://www.wiod.org/protected3/data13/wiot_analytic/'
    web_sheet = 'wiot00_row_apr12.xlsx'

    # Access that spreadshett and save it locally
    web_file = get(wiod_url+web_sheet, stream=True)  # Access file on server
    with open(data_file+data_file_ext, 'wb') as local_file:  # Open local file
        for chunk in web_file.iter_content(chunk_size=128):  # Go through contents on server
            local_file.write(chunk)  # Write to local file

    # Read the downloaded spreadsheet into a DataFrame
    data = pd.read_excel(data_file+data_file_ext, skiprows=[x for x in range(2)],
        header=[x for x in range(4)], index_col=[x for x in range(4)], skipfooter=8)

    # Get rid of the last column, which just contains totals
    data = data.iloc[:, :-1]

    # Specify names for index levels
    data.columns.names = data_index_orig
    data.index.names = data_index_orig

    # Save the DataFrame locally
    data.to_pickle(data_file+'.pkl')
else:
    # Read in the locally saved DataFrame
    data = pd.read_pickle(data_file+'.pkl')

# Reorder the index levels to a more usable order
for x in range(2):
    data = data.reorder_levels(data_index_reorder, axis = x)

# Make a list of a c codes indicating intermediate goods (c1 - c35)
intermediate_c_range = []
for x in range(35):
    intermediate_c_range.append('c'+str(x+1))

# Make a DataFrame containing only intermediate goods, and one containing only final goods
intermediate_flows = data.iloc[:, [x in intermediate_c_range for x in data.columns.get_level_values('c_num')]]
final_flows = data.iloc[:, [x not in intermediate_c_range for x in data.columns.get_level_values('c_num')]]

# Sum both across the country level, across both axes
intermediate_flows = intermediate_flows.sum(axis=0, level='country').sum(axis=1, level='country')
final_flows = final_flows.sum(axis=0, level='country').sum(axis=1, level='country')

# Create vectors of total intermediate goods and final goods imports by country
intermediate_imports = intermediate_flows.sum(axis=0) - np.diag(intermediate_flows)
final_imports = final_flows.sum(axis=0) - np.diag(final_flows)

# Create a vector of the ratio of intermediate to total imports
intermediate_import_ratio = intermediate_imports / (intermediate_imports + final_imports)

# Create a matrix of trade flows, combining intermediate and final goods
total_flows = intermediate_flows + final_flows

# Create vectors of total imports and exports
total_imports = total_flows.sum(axis=0) - np.diag(total_flows)
total_exports = total_flows.sum(axis=1) - np.diag(total_flows)

# Calculate trade deficits
trade_deficit = total_exports - total_imports

# Make a vector of total expenditure
total_expenditure = total_imports + np.diag(total_flows)

# Calculate the ratio of trade deficits to total expenditure
trade_deficit_ratio = trade_deficit / total_expenditure

# A word of caution: Note a) that using DataFrames such as total_expenditure directly when doing linear algebra is a
# huge mess, and will screw up the results, so they have to be converted in to two-dimensional Numpy arrays before doing
# anything with them. Also note b) that np.array(<input>, ndmin=2) creates a row vector, rather than a column vector,
# hence there will be many .transpose() operations later. (Because I like column vectors much more than row vectors.)

# To get the trade shares, first make a matrix in which each column corresponds to a given country's total expenditure
# (that is, each column just repeats total expenditure of that country)
expenditure_columns = np.ones((total_expenditure.shape[0],1)) @ np.array(total_expenditure, ndmin=2)

# Now, pointwise divide the matrix of total trade flows by this expenditure column matrix. A note on reading that
# matrix: I kept it in the from -> to style of the original WIOD. That is, the [i,n] entry shows trade flows from
# country i to country n, i.e. \pi_{ni}, in the EK notation.
trade_shares = total_flows / expenditure_columns

# Store this last data set as well
trade_shares.to_pickle(trade_shares_file+'.pkl')
trade_shares.to_excel(trade_shares_file+trade_shares_file_ext, sheet_name='trade_shares')

# Change directory to figures
chdir(mdir+fdir)

# Specify a list of series to plot (graph_dfs) and names (i.e. titles) for each series (graph_names)
graph_dfs = [intermediate_import_ratio, trade_deficit_ratio]
graph_names = ['Panel A: Share of intermediate imports in total imports, by country',
    'Panel B: Trade deficit as a fraction of total expenditure, by country']

# Make a graph of intermediate imports to total imports ratios
fig, axes = plt.subplots(len(graph_dfs), 1, figsize=(15, 9))

# Go through all graphs, keeping track of their name, as well as including a counter for the subplot
for i, (df, name) in enumerate(zip(graph_dfs, graph_names)):
    # Make an x axis list of values (arbitrary)
    x = [x for x in range(df.shape[0])]

    # Plot the intermediate imports as a bar chart
    axes[i].bar(x, df.sort_values(), align='center', width=0.65)

    # Include a graph name
    axes[i].set_title(name, y=1)

    # Make sure there are ticks at every bar
    axes[i].set_xticks(x)

    # Set the country labels as bar (tick) labels
    axes[i].set_xticklabels(df.sort_values().index.get_level_values('country'), rotation=45)

    # Set the x axis limits to save on whitespace
    axes[i].set_xlim([-1, len(x)])

# Trim unnecessary whitespace
fig.tight_layout()

# Add some more space between subplots, so titles remain legible
fig.subplots_adjust(hspace=0.3)

# Save and close the figure
plt.savefig('trade_share_graphs.pdf')
plt.close()

########################################################################################################################
### Q7.2: Basic EK model
########################################################################################################################

# Note that expenditures and expenditures times trade shares don't add up in these data, which I'll need to account for
# when checking excess demand below
Z_orig = (
    trade_shares @ np.array(total_expenditure, ndmin=2).transpose() - np.array(total_expenditure, ndmin=2).transpose()
    )

# Print the average divergence as a percentage of total expenditure
print('Excess demand as a percentage of total expenditure in the data:',
    (Z_orig / np.array(total_expenditure, ndmin=2).transpose()).mean()*100, 'percent\n')

# Set theta parameter
theta = 8.25

# Specify changes to fundamentals (currently, a ten percent drop in inter-country trade cost); note that d_hat is a
# matrix, since changes there are country pair specific!
d_hat = np.ones(trade_shares.shape) * .9 + np.eye(trade_shares.shape[0]) * .1
L_hat = np.ones((trade_shares.shape[0], 1))
T_hat = np.ones((trade_shares.shape[0], 1))

# Set up initial guess for wage changes
w_hat = np.ones((trade_shares.shape[0], 1))

# Set up a flag for convergence
converged = False

# Set up an interation counter and specify the maximum number of iterations after which the program aborts
iter = 0
max_iter = 10000

# Set a tolerance level; if excess demand is below this level for all countries (in absolute value), the program counts
# that as having achieved convergence
tol = 10**(-4)

# Set adjustment factor for the pricing function (see below)
adj_factor = .2

# As long as convergence hasn't been achieved
while not converged:
    # Calculate counterfactual trade shares
    # Start with the original trade share matrix and fundamental changes, and generate a matrix where the (i,j) element
    # is pi(j,i) * T_hat(i) * (d_hat(j,i) * w_hat(i))^(-theta)
    trade_shares_prime = (
        trade_shares * d_hat**(-theta)
        * ( np.ones((trade_shares.shape[0], 1)) @ (T_hat * w_hat**(-theta)).transpose() )
        )

    # Divide that matrix by the sum across rows, i.e. across origins. This respects the organization of the trade
    # shares matrix, where the rows indicate 'from', and the columns indicate 'to' countries.
    trade_shares_prime = (
        trade_shares_prime /
        ( np.ones((trade_shares.shape[0], 1)) @ np.array(trade_shares_prime.sum(axis=0), ndmin=2) )
        )

    # Calculate excess demand
    Z = (
        trade_shares_prime @ ( np.array(total_expenditure, ndmin=2).transpose() * w_hat * L_hat )
        - w_hat * L_hat * np.array(total_expenditure, ndmin=2).transpose() - Z_orig
        )

    # Adjust wages upwards if excess demand is positive, and downwards if it is negative
    w_hat = w_hat * ( 1 + ( adj_factor * (Z / np.array(total_expenditure, ndmin=2).transpose()) ) )

    # Enforce the world GDP as numeraire normalization
    w_hat = (
        w_hat / ( w_hat * L_hat * ( np.array(total_expenditure, ndmin=2).transpose() / total_expenditure.sum() ) ).sum()
        )

    # Increase the iteration counter
    iter += 1

    # Check for convergence
    if all(np.abs(Z) < tol):
        # If it has been achieved, print a message and set the convergence flag to true to stop the loop
        print('\nConverged after', iter, 'iterations\n')
        converged = True

    # Check whether the maximum iterations have been reached and the loop hasn't converged
    if iter == max_iter and not converged:
        # If so, print a message and abort the loop
        print('Maximum iterations reached (', max_iter, ')! Aborting...\n')
        break

# Make a DataFrame containing wage changes with a country index, since that's easier to read
w_hat_df = pd.DataFrame(data=w_hat, index=trade_shares.index, columns=['Real wage change'])

# Calculate welfare changes between the two scenarios
welfare_change = (
    T_hat**(1/theta)
    * np.array((np.diag(trade_shares) / np.diag(trade_shares_prime))**(1/theta), ndmin=2).transpose()
    )

# Put these into a DataFrame
welfare_change_df = pd.DataFrame(data=welfare_change, index=trade_shares.index, columns=['Welfare change'])

# Display the results
print(w_hat_df, '\n', welfare_change_df)

# Make a DataFrame of some of the results for easy Latex integration
# Is it worth it to have a tables directory specifically for this? No. No it's not.
texdf = pd.concat((intermediate_import_ratio, trade_deficit_ratio, w_hat_df, welfare_change_df), axis=1)
texdf.columns = ['Intermediate import share', 'Trade deficit share', 'Real wage change', 'Welfare change']
texdf.to_latex('table.tex')
