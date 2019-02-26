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

# Specify name of service quality variable
v_svq = 'plan_service_quality'

# Specify name of indicator for whether a comparison tool was available
v_tool = 'has_comparison_tool'

################################################################################
### Part 2.1: Chosen quantities
################################################################################

# Make an indicator for a plan being chosen
v_chosen = (insurance_data[v_pid] == insurance_data[v_cid])

# Specify a suffix for chosen quantities
sfx_c = '_chosen'

# Add variable containing chosen quantities to the data set
insurance_data =  insurance_data.join(
    insurance_data.loc[v_chosen, [v_pid, v_pre, v_cov, v_svq]],
    rsuffix=sfx_c)

################################################################################
### Part 2.2: Better plans
################################################################################

# Specify name for variable indicating whether a plan dominates the chosen one
# in terms of premium
v_lower_pre = 'lower_pre'

# Generate the indicator. It's easier to work with this as an integer, hence the
# .astype(int).
insurance_data[v_lower_pre] = (insurance_data[v_pre]
                               < insurance_data[v_pre+sfx_c]).astype(int)

# Specify name for variable indicating whether a plan has a (weakly) higher
# coverage than the chosen one
v_higher_cov = 'higher_cov'

# Generate the indicator
insurance_data[v_higher_cov] = (insurance_data[v_cov]
                                >= insurance_data[v_cov+sfx_c]).astype(int)

# Specify name for variable indicating whether a plan has (weakly) higher
# service quality than the chosen one
v_higher_svq = 'higher_service_quality'

# Generage the indicator
insurance_data[v_higher_svq] = (insurance_data[v_svq]
                                >= insurance_data[v_svq+sfx_c]).astype(int)

# Specify name for variable indicating whether a plan dominates the chosen one
# in terms of premium and coverage
v_better_pc = 'better_plan_pre_cov'

# Generate an indicator for plans which are not chosen, but have a lower premium
# and weakly better coverage and service quality (i.e. dominating plans). It's
# easier to work with this as an integer, hence the .astype(int).
insurance_data[v_better_pc] = (insurance_data[v_lower_pre]
                               & insurance_data[v_higher_cov]).astype(int)

# Specify variable name for count of number of better plans in the choice set
v_n_better_pc = 'count_better_pre_cov'

# Generate the count variable
insurance_data[v_n_better_pc] = (
    insurance_data[v_better_pc].groupby([v_id, v_cs]).sum())

# Specify name for indicator of whether plan has lower premium and higher
# service quality than the chosen one
v_better_pq = 'better_plan_pq'

# Generate the indicator
insurance_data[v_better_pq] = (insurance_data[v_lower_pre]
                               & insurance_data[v_higher_svq]).astype(int)

# Specify variable name for count of number of better plans in the choice set
v_n_better_pq = 'count_better_pre_svq'

# Generate the count variable
insurance_data[v_n_better_pq] = (
    insurance_data[v_better_pq].groupby([v_id, v_cs]).sum())

# Specify name for indicator of whether plan has lower premium, higher coverage,
# and higher service quality than the chosen one
v_better_all = 'better_plan_all'

# Generate the indicator
insurance_data[v_better_all] = (insurance_data[v_lower_pre]
                                & insurance_data[v_higher_cov]
                                & insurance_data[v_higher_svq]).astype(int)

# Specify variable name for count of number of better plans in the choice set
v_n_better_all = 'count_better_all'

# Generate the count variable
insurance_data[v_n_better_all] = (
    insurance_data[v_better_all].groupby([v_id, v_cs]).sum())

################################################################################
### Part 2.3: Dominating plans
################################################################################

# Specify name for variable indicating whether a chosen plan is dominated in
# terms of premium and coverage
v_dom_pc = 'dominated_pc'

# Add dominated plan indicator to the DataFrame, by checking whether any better
# plans are available
insurance_data[v_dom_pc] = (insurance_data[v_n_better_pc] > 0).astype(int)

# Specify name for variable indicating whether a chosen plan is dominated in
# terms of premium and service quality
v_dom_pq = 'dominated_pq'

# Add dominated plan indicator to the DataFrame, by checking whether any better
# plans are available
insurance_data[v_dom_pq] = (insurance_data[v_n_better_pq] > 0).astype(int)

# Specify name for variable indicating whether a chosen plan is dominated in
# terms of premium, coverage, and quality
v_dom_all = 'dominated_all'

# Add dominated plan indicator to the DataFrame, by checking whether any better
# plans are available
insurance_data[v_dom_all] = (insurance_data[v_n_better_all] > 0).astype(int)

################################################################################
### Part 2.4.1: Switches, switches to dominated plans
################################################################################

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

# Specify name for switches to plans dominated in terms of premium and coverage
v_swdom_pc = 'switch_dom_pc'

# Generate an indicator of whether a switch was to a dominated plan
insurance_data[v_swdom_pc] = (
    insurance_data[v_switch] & insurance_data[v_dom_pc]).astype(int)

# Specify name for switches to plans dominated in terms of premium and quality
v_swdom_pq = 'switch_dom_pq'

# Generate an indicator of whether a switch was to a dominated plan
insurance_data[v_swdom_pq] = (
    insurance_data[v_switch] & insurance_data[v_dom_pq]).astype(int)

# Specify name for switches to plans dominated in terms of premium, coverage,
# and service quality
v_swdom_all = 'switch_dom_all'

# Generate an indicator of whether a switch was to a dominated plan
insurance_data[v_swdom_all] = (
    insurance_data[v_switch] & insurance_data[v_dom_all]).astype(int)

################################################################################
### Part 2.5: Subsetting
################################################################################

# Make a version of the data which contains only the rows for chosen plans
insurance_data_red = insurance_data.loc[v_chosen]

################################################################################
### Part 3: Calculate descriptive statistics
################################################################################

################################################################################
### Part 3.1: Switching measures
################################################################################

# Specify names of dominated choice measures
domfrac_names = ['Dominated choices (coverage)', 'Dominated choices (quality)',
                 'Dominated choices (both)']

# Specify labels for plotting these
domfrac_plotnames = ['Coverage', 'Quality', 'Both']

# These will be put into a DataFrame. Specify a name for its (only) column.
col = ['Value']

# Set up dominated choices measures DataFrame
domfrac = pd.DataFrame(np.zeros(shape=(len(domfrac_names),1)),
                       index=domfrac_names, columns=col)

# Calculate fraction of choices dominated by the premium and coverage measure
domfrac.loc[domfrac_names[0], col] = insurance_data_red[v_dom_pc].mean()

# Calculate fraction of choices dominated by the premium and quality measure
domfrac.loc[domfrac_names[1], col] = insurance_data_red[v_dom_pq].mean()

# Calculate fraction of choices dominated by both measures
domfrac.loc[domfrac_names[2], col] = insurance_data_red[v_dom_all].mean()

# Print dominated choices measures
print('\n', domfrac.to_string())

# Specify names of switching measures
switchfrac_names = ['Switches without tool', 'Switches with tool',
                    'Dominated (coverage), no tool',
                    'Dominated (coverage), tool',
                    'Dominated (quality), no tool',
                    'Dominated (quality), tool',
                    'Dominated (both), no tool', 'Dominated (both), tool']

# Make versions of these for plotting (the with / without tool will be handled
# by shading in the bars, so that's not needed here)
switchfrac_plotnames = ['Fraction switching',
                        'Dominated \n switches \n (coverage)',
                        'Dominated \n switches \n (quality)',
                        'Dominated \n switches \n (both)']

# Set up switching measures DataFrame
switchfrac = pd.DataFrame(np.zeros(shape=(len(switchfrac_names),1)),
                       index=switchfrac_names, columns=col)

# Calculate fraction of switches without the tool
switchfrac.loc[switchfrac_names[0], col] = (
    insurance_data_red.loc[(insurance_data_red[v_tool] == 0), v_switch].mean())

# Calculate fraction of switches with tool
switchfrac.loc[switchfrac_names[1], col] = (
    insurance_data_red.loc[(insurance_data_red[v_tool] == 1), v_switch].mean())

# Calculate fraction of switches to plans dominated in terms of premium and
# coverage without the tool
switchfrac.loc[switchfrac_names[2], col] = (
    insurance_data_red.loc[(insurance_data_red[v_tool] == 0)
                           & insurance_data_red[v_switch], v_swdom_pc].mean())

# Calculate fraction of switches to plans dominated in terms of premium and
# coverage when a tool was available
switchfrac.loc[switchfrac_names[3], col] = (
    insurance_data_red.loc[(insurance_data_red[v_tool] == 1)
                           & insurance_data_red[v_switch], v_swdom_pc].mean())

# Calculate fraction of switches to plans dominated in terms of premium and
# service quality without the tool
switchfrac.loc[switchfrac_names[4], col] = (
    insurance_data_red.loc[(insurance_data_red[v_tool] == 0)
                           & insurance_data_red[v_switch], v_swdom_pq].mean())

# Calculate fraction of switches to plans dominated in terms of premium and
# service quality when a tool was available
switchfrac.loc[switchfrac_names[5], col] = (
    insurance_data_red.loc[(insurance_data_red[v_tool] == 1)
                           & insurance_data_red[v_switch], v_swdom_pq].mean())

# Calculate fraction of switches to completely dominated plans without the tool
switchfrac.loc[switchfrac_names[6], col] = (
    insurance_data_red.loc[(insurance_data_red[v_tool] == 0)
                           & insurance_data_red[v_switch], v_swdom_all].mean())

# Calculate fraction of switches to completely dominated plans when a tool was
# available
switchfrac.loc[switchfrac_names[7], col] = (
    insurance_data_red.loc[(insurance_data_red[v_tool] == 1)
                           & insurance_data_red[v_switch], v_swdom_all].mean())

# Print switching measures
print('\n', switchfrac.to_string())

################################################################################
### Part 4: Make figures
################################################################################

# Change to figures directory
chdir(mdir+fdir)

# Set main color (UM official blue is #00274c)
mclr = '#00274c'

# Set secondary color (UM official maize if #ffcb05)
sclr = '#ffcb05'

# Set edge color
eclr = 'black'

# Put both in a list
colors = [mclr, sclr]

# Make a list of patterns
patterns = ['', '//'] * np.int((len(switchfrac_names) / 2))

################################################################################
### Part 4.1: Dominated choices
################################################################################

# Specify name for dominated choices bar chart
fname = 'dominated_choices_bar'

# Set up the chart
fig, ax = plt.subplots(1, 1, num=fname, figsize=(6.5, 6.5*(9/16)))

# Plot switching fractions
bars = ax.bar([x for x in range(len(domfrac_names))],
              domfrac[col[0]].values, color=mclr,
              edgecolor=eclr, hatch='')

ax.set_xticks([x for x in range(len(domfrac_plotnames))])

# Use plot names, change font size
ax.set_xticklabels(domfrac_plotnames, fontsize=11)

# Don't displat tick marks on the horizontal axis
ax.tick_params(axis='x', bottom='off')

# Set vertical axis label
ax.set_ylabel('Fraction', fontsize=11)

# Make sure all labels are visible (otherwise, they might get cut off)
fig.tight_layout()

# Save the figure
plt.savefig(fname+ffmt)

# Close the figure (all figures, in fact)
plt.close('all')

################################################################################
### Part 4.2: Dominated switches
################################################################################

# Specify name for switching fractions bar chart
fname = 'switch_bar'

# Set up the chart
fig, ax = plt.subplots(1, 1, num=fname, figsize=(6.5, 6.5*(9/16)))

# Plot switching fractions
bars = ax.bar([x for x in range(len(switchfrac_names))],
              switchfrac[col[0]].values, color=colors,
              edgecolor=eclr, hatch='')

ax.set_xticks([2*x + .5 for x in range(len(switchfrac_plotnames))])

for bar, pattern in zip(bars, patterns):
    bar.set_hatch(pattern)

# Use plot names, change font size
ax.set_xticklabels(switchfrac_plotnames, fontsize=11)

# Don't displat tick marks on the horizontal axis
ax.tick_params(axis='x', bottom='off')

# Set vertical axis label
ax.set_ylabel('Fraction', fontsize=11)

ax.legend((bars[0], bars[1]), ('Without tool', 'With tool'))

# Make sure all labels are visible (otherwise, they might get cut off)
fig.tight_layout()

# Save the figure
plt.savefig(fname+ffmt)

# Close the figure (all figures, in fact)
plt.close('all')
#
