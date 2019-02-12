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
download_data = True

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
data = pd.read_stata(data_file, index_col=id)

################################################################################
### Part 3: Data processing
################################################################################

# Specify indicator for ITT sample
itt_ind = 'ittsample4'

# Retain only the ITT sample
data = data[data[itt_ind]==1]

################################################################################
### Part 4: Run free step down resampling
################################################################################

# Specify how many cores to use for parallel processing
ncores = cpu_count()

# Get p-values using all available cores in parallel
#results = Parallel(n_jobs=ncores)(delayed(function)(b, args) for b in range(B))

n=10000
k=3
beta=np.ones(shape=(k,1))
X = np.random.normal(size=(n,k))
eps = np.random.normal(size=(n,1))
y = X@beta + eps
bhat,Vhat,t = ols(y,X)
print(bhat)
print(Vhat)
print(t)
