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
download_data = False

data_index_orig = ['industry_code', 'indutry_name_or_exp_type', 'country', 'c_num']
data_index_reorder = ['country', 'c_num', 'indutry_name_or_exp_type', 'industry_code']

# Change directory to data
chdir(mdir+ddir)

# Check whether to download data
if download_data:
    wtiod_url = 'http://www.wiod.org/protected3/data13/wiot_analytic/'
    web_file = 'wiot00_row_apr12.xlsx'
else:
    data = pd.read_pickle(data_file+'.pkl')

print(data.index)
