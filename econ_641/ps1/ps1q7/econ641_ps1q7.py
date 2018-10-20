import pandas as pd
from requests import get
from os import path, mkdir, chdir

# Set main directory
mdir = path.dirname(path.abspath(__file__)).replace('\\', '/')

# Set data directory
ddir = '/data'

# Set up main data file
data_file = 'wiot00_row_apr12'
data_file_ext = '.xlsx'

# Set download flag
download_data = True

data_index_orig = ['industry_code', 'industry_name_or_final_good', 'country', 'c_number']
data_index_reorder = ['country', 'c_number', 'industry_name_or_final_good', 'industry_code']

# Change directory to data
chdir(mdir+ddir)

# Check whether to download data
if download_data:
    wtiod_url = 'http://www.wiod.org/protected3/data13/wiot_analytic/'
    web_sheet = 'wiot00_row_apr12.xlsx'

    web_file = get(wtiod_url+web_sheet, stream=True)  # Access file on server
    with open(data_file+data_file_ext, 'wb') as local_file:  # Open local file
        for chunk in web_file.iter_content(chunk_size=128):  # Go through contents on server
            local_file.write(chunk)  # Write to local file

    data = pd.read_excel(data_file+data_file_ext, skiprows=[x for x in range(2)] header=[], skipfooter=8)
else:
    data = pd.read_pickle(data_file+'.pkl')

for x in [0,1]:
    data = data.reorder_levels(data_index_reorder, axis=x)

#print(data.loc[('AUS', 'c1'), ('BEL', 'c1')])
