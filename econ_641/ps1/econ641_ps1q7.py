########################################################################################################################
### ECON 641: PS1, Q7.1
### Calculates trade shares based on the World Input Output Database (WIOD)
### Also provides some other international trade information along the way
########################################################################################################################

# Import necessary packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from requests import get
from os import chdir, mkdir, path, mkdir

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

# Specify name of main data file, plus extension (doesn't need to exist, if you want to download it again)
data_file = 'wiot00_row_apr12'
data_file_ext = '.xlsx'

# Set up name of trade shares file, plus extension (the program creates this)
trade_shares_file = 'wiot00_trade_shares'
trade_shares_file_ext = '.xlsx'

# Specify whether to re-download the data
download_data = False

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

# To get the trade shares, I'll first make a matrix in which each column corresponds to a given country's total
# expenditure. It's easy to then pointwise divide the trade flows by that matrix. To get this, use the Kronecker
# product between the 1x41 vector of total expenditures and a 41x1 vector of ones.
expenditure_columns = np.kron(np.array(total_expenditure, ndmin=2), np.ones((total_expenditure.shape[0],1)))

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

w_hat = np.ones(trade_shares.shape[0])

converged = False

iter = 0
max_iter = 100
while not converged:
    if iter == max_iter:
        break

    iter += 1
