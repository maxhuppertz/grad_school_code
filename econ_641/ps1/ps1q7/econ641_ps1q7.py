import pandas as pd
from requests import get
from os import path, mkdir, chdir

# Set main directory
mdir = path.dirname(path.abspath(__file__)).replace('\\', '/')

# Set data directory
ddir = '/data'

data_file = ''

download_data = False

chdir(mdir+ddir)

if download_data:
    wtiod_url = 'http://www.wiod.org/protected3/data13/wiot_analytic/'
    web_sheet = 'wiot00_row_apr12.xlsx'
else:
    pd.read_pickle()
