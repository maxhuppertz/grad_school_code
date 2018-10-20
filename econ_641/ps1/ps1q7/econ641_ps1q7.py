import pandas as pd
from requests import get
from os import chdir, mkdir, path, mkdir

# Set main directory
mdir = path.dirname(path.abspath(__file__)).replace('\\', '/')

# Set data directory
ddir = '/data'

# Create the data directory if it doesn't exist
if not path.isdir(mdir+ddir):
    mkdir(mdir+ddir)

# Set up main data file
data_file = 'wiot00_row_apr12'
data_file_ext = '.xlsx'

# Set download flag
download_data = False

# Specify names for index levels
ind_icode = 'industry_code'
ind_iname_fgood = 'industry_name_or_final_good'
ind_country = 'country'
ind_c = 'c_num'

# Make a list of the original order, as well as a reordered index
data_index_orig = [ind_icode, ind_iname_fgood, ind_country, ind_c]
data_index_reorder = [ind_country, ind_c, ind_iname_fgood, ind_icode]

# Change directory to data
chdir(mdir+ddir)

# Check whether to download data
if download_data:
    # Specify WTIOD URL, as well as which spreadsheet to download
    wtiod_url = 'http://www.wiod.org/protected3/data13/wiot_analytic/'
    web_sheet = 'wiot00_row_apr12.xlsx'

    # Access that spreadshett and save it locally
    web_file = get(wtiod_url+web_sheet, stream=True)  # Access file on server
    with open(data_file+data_file_ext, 'wb') as local_file:  # Open local file
        for chunk in web_file.iter_content(chunk_size=128):  # Go through contents on server
            local_file.write(chunk)  # Write to local file

    # Read the downloaded spreadsheet into a DataFrame
    data = pd.read_excel(data_file+data_file_ext, skiprows=[x for x in range(2)],
        header=[x for x in range(4)], index_col=[x for x in range(4)], skipfooter=8)

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

intermediate_c_range = []
for x in range(35):
    intermediate_c_range.append('c'+str(x+1))

test = data.loc[:, data.index.get_level_values('c_num') not in intermediate_c_range]
print(test)

#test = data.sum(axis=0, level='country')
