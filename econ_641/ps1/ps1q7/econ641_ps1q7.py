import pandas as pd
from requests import get
from os import chdir, mkdir, path, mkdir

# Set main directory
mdir = path.dirname(path.abspath(__file__)).replace('\\', '/')
#test again
# Set data directory
ddir = '/data'

# Create the data directory if it doesn't exist
if not path.isdir(mdir+ddir):
    mkdir(mdir+ddir)

# Set up main data file
data_file = 'wiot00_row_apr12'
data_file_ext = '.xlsx'

# Set download flag
download_data = True

ind_icode = 'industry_code'
ind_iname_fgood = 'industry_name_or_final_good'
ind_country = 'country'
ind_c = 'c_num'

data_index_orig = [ind_icode, ind_iname_fgood, ind_country, ind_c]
data_index_reorder = [ind_country, ind_c, ind_iname_fgood, ind_icode]

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

    data = pd.read_excel(data_file+data_file_ext, skiprows=[x for x in range(2)], header=[x for x in range(4)], index_col=[x for x in range(4)], skipfooter=8)

    data.columns.names = data_index_orig
    data.index.names = data_index_orig

    data.to_pickle(data_file+'pkl')
else:
    data = pd.read_pickle(data_file+'.pkl')

print(data.index)
print(data.columns)

#for x in [0,1]:
#    data = data.reorder_levels(data_index_reorder, axis=x)

#print(data.loc[('AUS', 'c1'), ('BEL', 'c1')])
