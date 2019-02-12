################################################################################
### Econ 666, PS2Q1: Multiple testing
################################################################################

# Import necessary packages and functions
import io
import numpy as np
import pandas as pd
import re
import requests
from joblib import Parallel, delayed
from linreg import ols
from multiprocessing import cpu_count
from os import chdir, mkdir, path
from shutil import copyfile, rmtree
from zipfile import ZipFile

################################################################################
### Part 1: Define necessary functions
################################################################################

def permute_p():
    pass

################################################################################
### Part 2.1: Set directories
################################################################################

# Specify name for main directory (just uses the file's directory)
mdir = path.dirname(path.abspath(__file__)).replace('\\', '/')

# Set data directory (doesn't need to exist)
ddir = '/data'

# Create the data directory if it doesn't exist
if not path.isdir(mdir+ddir):
    mkdir(mdir+ddir)

# Change directory to data
chdir(mdir+ddir)

################################################################################
### Part 2.2: Download/load data
################################################################################

# Specify name of data file
data_file = 'fertility_regressions.dta'

# Specify whether to download the data, or use a local copy instead
download_data = False

if download_data:
    # Specify URL for data zip file containing data file
    web_zip_url = 'https://www.aeaweb.org/aer/data/10407/20101434_data.zip'

    # Access file on server using requests.get(), which just connects to the raw
    # file and makes it possible to access it
    with requests.get(web_zip_url, stream=True) as web_zip_raw:
        # Use io.BytesIO() to convert the raw web file into a proper file saved
        # in memory, and use ZipFile() to make it into a zip file object
        with ZipFile(io.BytesIO(web_zip_raw.content)) as web_zip:
            # Go through all files in the zip file
            for file in web_zip.namelist():
                # Find the data file. The problem is that it will show up as
                # <path>/data_file, so this checks whether it can find such a
                # <path>/<file> combination which ends with /data_file
                if file.endswith('/'+data_file):
                    # Once it's found it, unpack it using extract(), and copy
                    # the result into the data directory
                    copyfile(web_zip.extract(file), mdir+ddir+'/'+data_file)

                    # But now, the data directory also contains
                    # <path>/data_file, which I don't need. Of course, the
                    # <path> part really consists of
                    # <path1>/<path2>/.../data_file. This regular expression
                    # takes that string, splits it at the first </>, and keeps
                    # the first part, i.e. <path1>.
                    zipdir = re.split('/', file, maxsplit=1)[0]

                    # Delete that folder
                    rmtree(zipdir)

# Load data into memory
id = 'respondentid'  # Column to use as ID
data = pd.read_stata(data_file, index_col=id, convert_categoricals=False)

################################################################################
### Part 3: Data processing
################################################################################

# Specify indicator for ITT sample
v_itt_ind = 'ittsample4'

# Retain only the ITT sample (pandas knows that the boolean vector its receiving
# as an index must refer to rows)
data = data[data[v_itt_ind]==1]

################################################################################
### Part 3.1: Who wants how many kids, responder status
################################################################################

# Generate an indicator for whether the woman believes her partner wants a
# higher minimum number of children than she does (pandas knows that the column
# names I'm giving it refer to columns)
v_minkids_self = 'e8minnumber'
v_minkids_husb = 'e19minnumber_hus'
v_husb_more_minkids = 'husb_more_kids'
data[v_husb_more_minkids] = data[v_minkids_husb] > data[v_minkids_self]

# Replace it as NaN if one of the two components is missing (here, I need to
# select both columns and rows; .loc[<rows>, <columns>] is very useful for
# getting rows using a boolean vector, and columns using column names or
# something like <:> to select all columns)
data.loc[np.isnan(data[v_minkids_husb] + data[v_minkids_self]),
    v_husb_more_minkids] = np.nan

# Generate an indicator for whether the woman believes her husband wants a
# higher ideal number of children than she wants
v_idkids_self = 'e1_ideal'
v_idkids_husb = 'e12_hus_ideal'
v_husb_more_idkids = 'husb_more_idkids'
data[v_husb_more_idkids] = data[v_idkids_husb] > data[v_idkids_self]

# Replace it as NaN if the ideal number of kids for the husband is missing
v_idkids_husb_miss = 'd_e12_hus_ideal'
data.loc[data[v_idkids_husb_miss] == 1, v_husb_more_idkids] = np.nan

# Generate an indicator for whether the woman believes her partner wants a
# higher maximum number of children than she does
v_maxkids_self = 'e7maxnumber'
v_maxkids_husb = 'e18maxnumber_hus'
v_husb_more_maxkids = 'husb_more_maxkids'
data[v_husb_more_maxkids] = data[v_maxkids_husb] > data[v_maxkids_self]

# Replace it as NaN if either of the components are missing
data.loc[np.isnan(data[v_maxkids_husb] + data[v_maxkids_self]),
    v_husb_more_maxkids] = np.nan

# Generate an indicator for whether the couple currently have fewer children
# than the husband would ideally want to have
v_num_kids = 'currentnumchildren'
v_how_many_more = 'e17morekids_hus'
v_husb_wants_kids = 'husb_wants_kids'

# A note on the variable created in the next step: The original STATA code is
#
# gen h_wantsmore_ideal_m = (((e12_hus_ideal-currentnumchildren)>0) | e17morekids>0 )
#
# but that codes observations as 1 if either e12_hus_ideal or currentnumchildren
# are missing, and if e17morekids is missing. (Since in STATA, anything
# involving missing values is infinitely large and counted as True.) That is why
# I added np.isnan(data[v_idkids_husb] + data[v_num_kids]), which replicates the
# e12_hus_ideal / currentnumchildren issue, and np.isnan(data[v_how_many_more]),
# which replicates the e17morekids issue. (Since np.nan always evaluates to
# False in Python, I have to manually add these conditions to replicates the
# STATA assignments.) The problem is that in the next step, where these
# erroneous assignments are converted to missing, they forgot one condition, I
# think. (See below.)
data[v_husb_wants_kids] = (
    ((data[v_idkids_husb] - data[v_num_kids]) > 0) | (data[v_how_many_more] > 0)
    | np.isnan(data[v_idkids_husb] + data[v_num_kids])
    | np.isnan(data[v_how_many_more])
    )

# Replace it as NaN if any of the components are missing
# The original STATA code is
#
# replace h_wantsmore_ideal_m = . if (d_e12_hus_ideal==1 | currentnumchildren==.) & (e17morekids==-9)
#
# which corrects the issue with missing values for  e12_hus_ideal or
# currentnumchildren making the variable true. But it doesn't solve the
# same issue for e17morekids, since that is sometimes coded as -9 (which means
# the responder said she they don't know), but in other cases, it's just coded
# as missing. This code will not fix the missing issue.
data.loc[((data[v_idkids_husb_miss] == 1) | np.isnan(data[v_num_kids]))
    & (data[v_how_many_more] == -9),
    v_husb_wants_kids] = np.nan

# Specify variable name for indicator of whether the woman wants kids in the
# next two years
v_kids_nexttwo = 'wantschildin2'

# Generate an indicator for responder status (luckily, Python evaluates
# np.nan == True as False, so this code works for boolean data)
v_responder = 'responder'
data[v_responder] = (
    ((data[v_husb_more_minkids] == True) | (data[v_husb_more_maxkids] == True)
    | (data[v_husb_more_idkids] == True))
    & (data[v_husb_wants_kids] == True) & (data[v_kids_nexttwo] == 0)
    )

# Replace it as missing if some of the components are missing
data.loc[(np.isnan(data[v_husb_more_minkids]) &
    np.isnan(data[v_husb_more_maxkids]) & np.isnan(data[v_husb_more_idkids]))
    | np.isnan(data[v_husb_wants_kids]) | np.isnan(data[v_kids_nexttwo]),
    v_responder] = np.nan

################################################################################
### Part 3.2: Positive effects on well-being
################################################################################

#gen satisfied = (j11satisfy==4 | j11satisfy==5)
#replace satisfied = . if j11satisfy==.

#gen healthier = (a21health==4 | a21health==5)
#replace healthier = . if a21health==.

#gen happier = (a22happy==4 | a22happy==5)
#replace happier = . if a22happy==.

#eststo D: reg satisfied Icouples if ittsample4_follow == 1 & responder_m==1
#eststo E: reg healthier Icouples if ittsample4_follow == 1 & responder_m==1
#eststo F: reg happier Icouples if ittsample4_follow == 1 & responder_m==1

################################################################################
### Part 3.3: Negative effects
################################################################################

#gen separated2 = (b1marstat==2 | b1marstat==3 | b1marstat==.)

#gen violence_follow = (f10violenc==1)
#replace violence_follow = . if f10violenc<0 | f10violenc==.

#gen cur_using_condom = (g14mc==1)
#replace cur_using_condom = . if g14mc==.

#eststo D: reg separated2 Icouples if ittsample4_follow == 1 & responder_m==1
#eststo E: reg violence_follow Icouples if ittsample4_follow == 1 & responder_m==1
#eststo F: reg cur_using_condom Icouples if ittsample4_follow == 1 & responder_m==1

################################################################################
### Part 4: Run free step down resampling
################################################################################

# Specify how many cores to use for parallel processing
ncores = cpu_count()

# Get p-values using all available cores in parallel
#results = Parallel(n_jobs=ncores)(delayed(function)(b, args) for b in range(B))
