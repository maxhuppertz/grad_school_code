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
### Part 3.1: Dominated choices
################################################################################

# Specify names of dominated choice measures, as a dictionary. Each entry should
# return the corresponding variable.
dominance_measures = {'Coverage': v_dom_pc, 'Quality': v_dom_pq,
                      'Both': v_dom_all}

# Set up dominated choices measures DataFrame
domfrac = pd.DataFrame(np.zeros(shape=(len(dominance_measures), 2)))

# Go through all dominance measures
for i, measure in enumerate(dominance_measures):
    # Get the corresponding variable
    var = dominance_measures[measure]

    # Record the name of the measures
    domfrac.loc[i, 0] = measure

    # Calculate the mean of the variable
    domfrac.loc[i, 1] = insurance_data_red[var].mean()

# Print dominated choices measures
print('\nDominance measures\n',
      domfrac.to_string(header=['Measure', 'Value']), sep='')

################################################################################
### Part 3.2: Switching measures
################################################################################

# Make an index for people not having the tool (i.e. a Boolean array)
idx_notool = (insurance_data_red[v_tool] == 0)

# Make an index for people having the tool
idx_tool = (insurance_data_red[v_tool] == 1)

# Put them into a list
ics_tool = [idx_notool, idx_tool]

# Make an index for people switching without the tool
idx_sw_notool = (insurance_data_red[v_tool] == 0) & insurance_data_red[v_switch]

# Make an index for people switching with the tool
idx_sw_tool = (insurance_data_red[v_tool] == 1) & insurance_data_red[v_switch]

# Put them into a list
ics_swtool = [idx_sw_notool, idx_sw_tool]

# Specify names of switching measures as keys of this dictionary. Each entry
# should return first the variable the measure refers to, and second the
# indices used to condition for that measure when taking the mean.
switching_measures = {'Fraction switching': [v_switch, ics_tool],
                      'Dominated \n switches \n (coverage)':
                      [v_swdom_pc, ics_swtool],
                      'Dominated \n switches \n (quality)':
                      [v_swdom_pq, ics_swtool],
                      'Dominated \n switches \n (both)':
                      [v_swdom_all, ics_swtool]}

# Make a suffix for tables, indicating whether a measure is conditioned on
# having or not having the tool available
tabsuf = [', no tool', ', tool']

# Set up switching measures DataFrame
switchfrac = pd.DataFrame(np.zeros(shape=(len(switching_measures)*2, 2)))

# Go through all switching measures
for i, measure in enumerate(switching_measures):
    # Get the variable and indices
    var, indices = switching_measures[measure]

    # Go through all conditioning indices
    for j, idx in enumerate(indices):
        # Replace the name of the measure in the results table
        switchfrac.loc[2*i+j, 0] = measure.replace('\n ', '') + tabsuf[j]

        # Calculate the conditional mean of the measure
        switchfrac.loc[2*i+j, 1] = (
            insurance_data_red.loc[idx, var].mean())

# Print switching measures
print('\nSwitching measures\n',
      switchfrac.to_string(header=['Measure', 'Value']), sep='')

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
patterns = ['', '//'] * len(switching_measures)

################################################################################
### Part 4.1: Dominated choices
################################################################################

# Specify name for dominated choices bar chart
fname = 'dominated_choices_bar'

# Set up the chart
fig, ax = plt.subplots(1, 1, num=fname, figsize=(6.5, 6.5*(9/16)))

# Plot switching fractions
bars = ax.bar([x for x in range(len(domfrac_names))],
              domfrac.loc[:, 1].values, color=mclr,
              edgecolor=eclr, hatch='')

# Set xticks
ax.set_xticks([x for x in range(len(dominance_measures))])

# Use plot names, change font size
ax.set_xticklabels(dominance_measures, fontsize=11)

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
bars = ax.bar([x for x in range(len(switching_measures)*2)],
              switchfrac.loc[:, 1].values, color=colors,
              edgecolor=eclr, hatch='')

# Set xticks
ax.set_xticks([2*x + .5 for x in range(len(switching_measures))])

# Go through all bars and their patterns
for bar, pattern in zip(bars, patterns):
    # Set the pattern for the bars
    bar.set_hatch(pattern)

# Use plot names, change font size
ax.set_xticklabels(switching_measures.keys(), fontsize=11)

# Don't display tick marks on the horizontal axis
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
