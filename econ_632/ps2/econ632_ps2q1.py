################################################################################
### ECON 632: PS2Q1 - Descriptive statistics
################################################################################

# Import matplotlib
import matplotlib as mpl

# Select backend that does not open figures interactively (has to be done before
# pyplot is imported); without this, Python will get confused when it tries to
# close figures, and it will send annoying warnings
mpl.use('Agg')

# Import other necessary packages and functions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from inspect import getsourcefile
from os import chdir, mkdir, path

################################################################################
### Part 1: Set graph options, set and make  directories, load custom packages
################################################################################

# Set graph options
#mpl.rcParams["text.latex.preamble"].append(r'\usepackage{amsmath}')
plt.rc('font', **{'family': 'serif', 'serif': ['lmodern']})
plt.rc('text', usetex=True)

# Specify name for main directory (just uses the file's directory)
# I used to use path.abspath(__file__), but apparently, it may be a better idea
# to use getsourcefile() instead of __file__ to make sure this runs on
# different OSs. I just give it an object, and it checks which file defined it.
# But since the object I give it is an inline function lambda, which was
# created in this file, it points to this file
mdir = path.dirname(path.abspath(getsourcefile(lambda:0))).replace('\\', '/')

# Change to main directory
chdir(mdir)

# Import custom packages (have to be in the main directory)
from linreg import larry, ols

# Set data directory (has to exist and contain insurance_data.csv)
ddir = '/data'

# Set figures/tables directory (doesn't need to exist)
fdir = '/figures'

# Create the figures directory if it doesn't exist
if not path.isdir(mdir+fdir):
    mkdir(mdir+fdir)

# Choose figure format
ffmt = '.pdf'

################################################################################
### Part 2: Load data, generate additional variables
################################################################################

# Change to data directory
chdir(mdir+ddir)

# Specify name of main data set
fname = 'insurance_data.csv'

# Specify name of individual ID variable
v_id = 'indiv_id'

# Specify name of choice situation ID variable
v_cs = 'choice_sit'

# Load the data, using individual and choice situation IDs as indices
insurance_data = pd.read_csv(fname, index_col=[v_id, v_cs])

# Specify plan ID variable
v_pid = 'plan_id'

# Specify variable containing ID of chosen plan
v_cid = 'plan_choice'

# Specify name of premium variable
v_pre = 'premium'

# Specify name of coverage variable
v_cov = 'plan_coverage'

v_svq = 'plan_service_quality'

# Make an indicator for a plan being chosen
v_chosen = insurance_data[v_pid] == insurance_data[v_cid]

# Specify a suffix for chosen quantities
sfx_c = '_chosen'

# Add variable containing chosen quantities to the data set
insurance_data =  insurance_data.join(
    insurance_data.loc[v_chosen, [v_pid, v_pre, v_cov, v_svq]],
    rsuffix=sfx_c)

# Specify name for variable indicating whether a plan dominates the chosen one
v_better = 'better_plan'

# Generate an indicator for plans which are not chosen, but have a lower premium
# and weakly better coverage and service quality (i.e. dominating plans). It's
# easier to work with this as an integer, hence the .astype(int).
insurance_data[v_better] = (
    (insurance_data[v_pre] < insurance_data[v_pre+sfx_c])
    & (insurance_data[v_cov] >= insurance_data[v_cov+sfx_c])
    & (insurance_data[v_svq] >= insurance_data[v_svq+sfx_c])
    ).astype(int)

# Specify variable name for count of number of better plans in the choice set
v_n_better = 'count_better'

# Generate the count variable
insurance_data[v_n_better] = (
    insurance_data[v_better].groupby([v_id, v_cs]).sum())

# Specify name for variable indicating whether a chosen plan is dominated
v_dom = 'dominated'

# Add dominated plan indicator to the DataFrame, by checking whether any better
# plans are available
insurance_data[v_dom] = (insurance_data[v_n_better] > 0).astype(int)

# Specify name for plan switching indicator
v_switch = 'switch'

# Start by taking the difference between the plan choice indicator and its
# preceding value, within individual
insurance_data[v_switch] = insurance_data[v_cid].groupby([v_id]).diff()

# Mark switches by checking that the difference is not zero, and non NaN. (Note
# that the first check uses !=, since it's element by element. The second uses
# ~=, since it's for a whole series.) Record this as an integer.
insurance_data[v_switch] = (
    (insurance_data[v_switch] != 0)
    & ~insurance_data[v_switch].isna()
    ).astype(int)

# Take the maximum for the switching variable within individual and choice
# situation, so it's coded for the whole situation
insurance_data[v_switch] = insurance_data[v_switch].groupby([v_id, v_cs]).max()

# Specify name for switches to dominated plans
v_swdom = 'switch_dom'

# Generate an indicator of whether a switch was to a dominated plan
insurance_data[v_swdom] = (
    insurance_data[v_switch] & insurance_data[v_dom]).astype(int)

# Specify name of indicator for whether a comparison tool was available
v_tool = 'has_comparison_tool'

# Specify name for an indicator of switches when a comparison tool was available
v_switch_tool = 'switch_tool'

# Generate an indicator of switches in the presence of the tool
insurance_data[v_switch_tool] = (
    insurance_data[v_switch] & insurance_data[v_tool]).astype(int)

# Specify a name for an indicator of dominated switches with a tool present
v_swdom_tool = 'switch_dom_tool'

# Generate an indicator of dominated switches in the presence of the tool
insurance_data[v_swdom_tool] = (
    insurance_data[v_swdom] & insurance_data[v_tool]).astype(int)

# Make a version of the data which contains only the rows for chosen plans
insurance_data_red = insurance_data.loc[v_chosen]

################################################################################
### Part 3: Calculate descriptive statistics
################################################################################

# Print general descriptives
#print('\n', insurance_data_red.describe())

# Specify names of descriptive statistics
destats_names = ['Switches', 'Switches with tool',
                 'Dominated switches', 'Dominated with tool']

# Make versions of these for plotting
destats_plotnames = ['Switches', 'Switches \n with tool',
                 'Dominated switches', 'Dominated \n with tool']

# These will be put into a DataFrame. Specify name(s) for its column(s).
destats_col = ['Value']

# Set up descriptive statistics DataFrame
destats = pd.DataFrame(np.zeros(shape=(len(destats_names),1)),
                       index=destats_names, columns=destats_col)

# Calculate fraction of switches
n_cs = insurance_data_red.shape[0]  # Number of choice situations
frac_switch = insurance_data_red[v_switch].sum() / n_cs

# Add the result to the descriptives
destats.loc[destats_names[0], destats_col] = frac_switch

# Calculate fraction of switches with tool
frac_switch_tool = (
    insurance_data_red[v_switch_tool].sum()
    / insurance_data_red[v_tool].sum())

# Add the result to the descriptives
destats.loc[destats_names[1], destats_col] = frac_switch_tool

# Calculate fraction of switches to dominated plans
frac_switch_dom = (
    insurance_data_red[v_swdom].sum() / insurance_data_red[v_switch].sum())

# Add the result to the descriptives
destats.loc[destats_names[2], destats_col] = frac_switch_dom

# Calculate fraction of switches to dominated plans when a tool was available
frac_switch_dom_tool = (
    insurance_data_red[v_swdom_tool].sum()
    / insurance_data_red[v_switch_tool].sum())

# Add the result to the descriptives
destats.loc[destats_names[3], destats_col] = frac_switch_dom_tool

print('\n', destats.to_string())

################################################################################
### Part 4: Make figures
################################################################################

# Change to figures directory
chdir(mdir+fdir)

# Specify name for switching fractions bar chart
fname = 'switch_bar'

# Set up the chart
fig, ax = plt.subplots(1, 1, num=fname, figsize=(6.5, 6.5*(9/16)))

# Plot switching fractions
ax.bar([x for x in range(len(destats_names))], destats[destats_col[0]].values,
       tick_label=destats_names)

# Use plot names, change font size
ax.set_xticklabels(destats_plotnames, fontsize=11)

# Set vertical axis label
ax.set_ylabel('Fraction', fontsize=11)

# Make sure all labels are visible (otherwise, they might get cut off)
fig.tight_layout()

# Save the figure
plt.savefig(fname+ffmt)

# Close the figure (all figures, in fact)
plt.close()
#
