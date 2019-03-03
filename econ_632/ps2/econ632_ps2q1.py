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

# Set precision for floats
prec = 4

# Make a string so this can be used with pandas' float_format argument
fstring = '{:,.'+str(prec)+'f}'

# Set pandas print options
pd.set_option('display.max_columns', 8)
pd.set_option('display.width', 110)
pd.set_option('display.precision', prec)

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
from texaux import textable

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
### Part 2.2: Number of better plans in the choice set
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
### Part 2.3: Subsetting
################################################################################

# Make a version of the data which contains only the rows for chosen plans (the
# .copy() makes sure that changing values in the reduced data does not change
# values in the full data set)
insurance_data_red = insurance_data.loc[v_chosen].copy()

################################################################################
### Part 2.4: Dominated plans
################################################################################

################################################################################
### Part 2.4.1: Contemporary plan is dominated
################################################################################

# Specify name for variable indicating whether a chosen plan is dominated in
# terms of premium and coverage
v_dom_pc = 'dominated_pc'

# Specify name for variable indicating whether a chosen plan is dominated in
# terms of premium and service quality
v_dom_pq = 'dominated_pq'

# Specify name for variable indicating whether a chosen plan is dominated in
# terms of premium, coverage, and quality
v_dom_all = 'dominated_all'

# Put them in a dictionary, associating them with the corresponding count of
# plans which are better in that category
domvars = {v_dom_pc: v_n_better_pc, v_dom_pq: v_n_better_pq,
           v_dom_all: v_n_better_all}

# Go through all dominance variables
for var in domvars:
    # Replace the variable if the count is greater than zero
    insurance_data_red[var] = (insurance_data_red[domvars[var]] > 0).astype(int)

################################################################################
### Part 2.4.1: Next period's plan is dominated
################################################################################

# Specify names for dominance indicators for the upcoming period
v_fdom_pc = 'upcoming_dominated_pc'
v_fdom_pq = 'upcoming_dominated_pq'
v_fdom_all = 'upcoming_dominated_all'

# Put them in a dictionary, associate them with the contemporary version
fdomvars = {v_fdom_pc: v_dom_pc, v_fdom_pq: v_dom_pq, v_fdom_all: v_dom_all}

# Go through all dominance variables
for var in fdomvars:
    # Shift the variable back by one period to get upcoming version
    insurance_data_red[var] = (
        insurance_data_red[fdomvars[var]].groupby([v_id]).shift(periods=-1))

################################################################################
### Part 2.5: Switches
################################################################################

################################################################################
### Part 2.5.1: Contemporary switches
################################################################################

# Specify name for plan switching indicator
v_switch = 'switch'

# Start by taking the difference between the plan choice indicator and its
# preceding value, within individual
insurance_data_red[v_switch] = insurance_data_red[v_cid].groupby([v_id]).diff()

# Mark switches by checking that the difference is not zero, and non NaN. Record
# this a numeric data.
insurance_data_red.loc[~insurance_data_red[v_switch].isna(), v_switch] = (
    (insurance_data_red[v_switch] != 0)).astype(float)

################################################################################
### Part 2.5.2: Upcoming switches
################################################################################

# Specify name for future (next period) switching variable
v_fswitch = 'switch_next_period'

# Get next period's value for the switching variable, within individual
insurance_data_red[v_fswitch] = (
    insurance_data_red[v_switch].groupby([v_id]).shift(periods=-1))

################################################################################
### Part 2.5.3: Contemporary switches to dominated plans
################################################################################

# Specify name for switches to plans dominated in terms of premium and coverage
v_swdom_pc = 'switch_dom_pc'

# Generate an indicator of whether a switch was to a dominated plan
insurance_data_red[v_swdom_pc] = (
    (insurance_data_red[v_switch] == 1.0)
    & insurance_data_red[v_dom_pc]).astype(int)

# Specify name for switches to plans dominated in terms of premium and quality
v_swdom_pq = 'switch_dom_pq'

# Generate an indicator of whether a switch was to a dominated plan
insurance_data_red[v_swdom_pq] = (
    (insurance_data_red[v_switch] == 1.0)
    & insurance_data_red[v_dom_pq]).astype(int)

# Specify name for switches to plans dominated in terms of premium, coverage,
# and service quality
v_swdom_all = 'switch_dom_all'

# Generate an indicator of whether a switch was to a dominated plan
insurance_data_red[v_swdom_all] = (
    (insurance_data_red[v_switch] == 1.0)
    & insurance_data_red[v_dom_all]).astype(int)

################################################################################
### Part 2.5.3: Upcoming switches to dominated plans
################################################################################

# Specify name for switches to plans dominated in terms of premium and coverage
v_fswdom_pc = 'upcoming_switch_dom_pc'

# Specify name for switches to plans dominated in terms of premium and quality
v_fswdom_pq = 'upcoming_switch_dom_pq'

# Specify name for switches to plans dominated in terms of premium, coverage,
# and service quality
v_fswdom_all = 'upcoming_switch_dom_all'

# Put them in a dictionary, associate with contemporary version
fsdomvars = {v_fswdom_pc: v_swdom_pc, v_fswdom_pq: v_swdom_pq,
             v_fswdom_all: v_swdom_all}

# Go through all future switching measures
for var in fsdomvars:
    # Get the upcoming version by shifting back by one period
    insurance_data_red[var] = (
        insurance_data_red[fsdomvars[var]].groupby([v_id]).shift(periods=-1))

################################################################################
### Part 2.6: Tool becomes available next period
################################################################################

# Specify variable indicating that the tool will be available next period
v_ftool = 'tool_available_next_period'

# Get the variable by shifting the tool indicator back one period
insurance_data_red[v_ftool] = (
    insurance_data_red[v_tool].groupby([v_id]).shift(periods=-1))

# Make an indicator of not having the tool and then getting it
newtool = (insurance_data_red[v_tool] == 0) & (insurance_data_red[v_ftool] == 1)

# Check whether everything in it is zero
if newtool.sum() == 0:
    # If so, display a message to that effect
    print('\nNo observations for which the tool was newly introduced')

################################################################################
### Part 2.7: New entrants
################################################################################

# Specify name of tenure variable
v_tenure = 'years_enrolled'

# Specify name for entry variable
v_entry = 'new_entrant'

# Generate the variable as an integer
insurance_data_red[v_entry] = (insurance_data_red[v_tenure] == 1).astype(int)

################################################################################
### Part 2.8: Squared variables
################################################################################

# Specify some further variables, which will be squared in the following
v_age = 'age'
v_inc = 'income'
v_rscore = 'risk_score'

# Set prefix and suffix for squared variables
suf2 = '^2'

# Select which variables to square
create_squarevars = [v_age, v_tenure, v_inc, v_rscore]

# Go through all variables to be squaredfor var in create_squarevars:
for var in create_squarevars:
    # Generate the squared variable
    insurance_data_red[var+suf2] = insurance_data_red[var]**2

################################################################################
### Part 2.9: Taking logs
################################################################################

# Specify which variables to convert to logs
create_logs = [v_inc]

# Specify prefix for logged variables
preflog = 'log_'

# Go through all variables that need to be logged
for var in create_logs:
    # Create the log
    insurance_data_red[preflog+var] = np.log(insurance_data_red[var])

################################################################################
### Part 2.10: Interactions
################################################################################

# Specify some further variables
v_sex = 'sex'

# Specify an infix for interaction terms
infint = '_X_'

# Specify sets of variables to interact
create_interactions = [[v_rscore, v_tenure], [v_rscore, v_sex],
                       [v_tool, v_rscore], [v_rscore, v_inc],
                       [v_rscore, preflog+v_inc], [v_tool, v_sex],
                       [v_tool, v_tenure], [v_tool, v_switch],
                       [v_tool, v_dom_pc], [v_tool, v_dom_pq],
                       [v_tool, v_dom_all], [v_tool, v_pre], [v_tool, v_cov],
                       [v_tool, v_svq]]

# Go through all sets of interactions
for vars in create_interactions:
    # Set up the variable name as the name of the first variable
    varname = vars[0]

    # Set up the variable's content as the first variable
    content = insurance_data_red[vars[0]]

    # Go through all other variables in the interaction
    for var in vars[1:]:
        # Add the name to the interaction variable's name
        varname = varname + infint + var

        # Multiply the variable with the interaction
        content = content * insurance_data_red[var]

    # Add the interaction term to the data set
    insurance_data_red[varname] = content

################################################################################
### Part 3: Calculate descriptive statistics
################################################################################

# Change to figures/tables directory
chdir(mdir+fdir)

################################################################################
### Part 3.1: Dominated choices
################################################################################

# Make an index for people not having the tool (i.e. a Boolean array)
idx_notool = (insurance_data_red[v_tool] == 0)

# Make an index for people having the tool
idx_tool = (insurance_data_red[v_tool] == 1)

# Put them into a list
ics_tool = [idx_notool, idx_tool]

# Specify names of dominated choice measures, as a dictionary. Each entry should
# return the corresponding variable.
dominance_measures = {'Dominated: coverage': v_dom_pc,
                      'Dominated: quality': v_dom_pq,
                      'Dominated: both': v_dom_all}

# Make a suffix for tables, indicating whether a measure is conditioned on
# having or not having the tool available
tabsuf = [' (no tool)', ' (tool)']

# Set up dominated choices measures DataFrame
domfrac = pd.DataFrame(np.zeros(shape=(len(dominance_measures)*2, 2)))

# Go through all dominance measures
for i, measure in enumerate(dominance_measures):
    # Get the corresponding variable
    var = dominance_measures[measure]

    # Go through the tool indices
    for j, idx in enumerate(ics_tool):
        # Record the name of the measure
        domfrac.loc[i*2+j, 0] = measure.replace('\n', '') + tabsuf[j]

        # Calculate the mean of the variable
        domfrac.loc[i*2+j, 1] = insurance_data_red.loc[idx, var].mean()

################################################################################
### Part 3.2: Switching with and without the tool
################################################################################

# Make an index for people switching without the tool
idx_sw_notool = (insurance_data_red[v_tool] == 0) & insurance_data_red[v_switch]

# Make an index for people switching with the tool
idx_sw_tool = (insurance_data_red[v_tool] == 1) & insurance_data_red[v_switch]

# Put them into a list
ics_swtool = [idx_sw_notool, idx_sw_tool]

# Specify names of switching measures as keys of this dictionary. Each entry
# should return first the variable the measure refers to, and second the
# indices used to condition for that measure when taking the mean.
switching_measures = {'Any switch': [v_switch, ics_tool],
                      'Dominated \n switch: \n coverage':
                      [v_swdom_pc, ics_swtool],
                      'Dominated \n switch: \n quality':
                      [v_swdom_pq, ics_swtool],
                      'Dominated \n switch: \n both':
                      [v_swdom_all, ics_swtool]}

# Set up switching measures DataFrame
switchfrac = pd.DataFrame(np.zeros(shape=(len(switching_measures)*2, 2)))

# Go through all switching measures
for i, measure in enumerate(switching_measures):
    # Get the variable and indices
    var, indices = switching_measures[measure]

    # Go through all conditioning indices
    for j, idx in enumerate(indices):
        # Replace the name of the measure in the results table
        switchfrac.loc[i*2+j, 0] = measure.replace('\n ', '') + tabsuf[j]

        # Calculate the conditional mean of the measure
        switchfrac.loc[i*2+j, 1] = insurance_data_red.loc[idx, var].mean()

################################################################################
### Part 3.3: Correlates of switching (regression)
################################################################################

# Specify name of the year variable
v_year = 'year'

# Specify switching measures to look at
switching_measures_reg = {'Any': v_fswitch, 'Coverage': v_fswdom_pc,
                          'Quality': v_fswdom_pq, 'Both': v_fswdom_all}

# Select which variables to use on the RHS for switches
Xvars_sw = {v_year: 'Year', v_sex: 'Sex', v_tenure: 'Tenure',
            v_tenure+suf2: r'$\text{Tenure}^2$', v_age: 'Age',
            v_age+suf2: r'$\text{Age}^2$', preflog+v_inc: 'Log(income)',
            v_rscore: 'Risk', v_rscore+suf2: r'$\text{Risk}^2$', v_tool: 'Tool',
            v_pre: 'Premium', v_cov: 'Coverage', v_svq: 'Quality',
            v_switch: 'Switch', v_dom_pc: 'Dominated: coverage',
            v_dom_pq: 'Dominated: quality', v_dom_all: 'Dominated: both',
            v_rscore+infint+v_tenure: r'Risk $\times$ tenure',
            v_rscore+infint+v_sex: r'Risk $\times$ sex',
            v_rscore+infint+preflog+v_inc: r'Risk $\times$ log(income)',
            v_tool+infint+v_sex: r'Tool $\times$ sex',
            v_tool+infint+v_tenure: r'Tool $\times$ tenure',
            v_tool+infint+v_rscore: r'Tool $\times$ risk',
            v_tool+infint+v_pre: r'Tool $\times$ premium',
            v_tool+infint+v_cov: r'Tool $\times$ coverage',
            v_tool+infint+v_svq: r'Tool $\times$ quality',
            v_tool+infint+v_switch: r'Tool $\times$ switch',
            v_tool+infint+v_dom_pc: r'Tool $\times$ dom.: coverage',
            v_tool+infint+v_dom_pq: r'Tool $\times$ dom.: quality',
            v_tool+infint+v_dom_all: r'Tool $\times$ dom.: both'}

# Create an intercept
beta0 = np.ones(shape=(insurance_data_red.shape[0], 1))

# Make variables into a matrix
X_sw = larry(insurance_data_red[list(Xvars_sw)])

# Add the intercept
X_sw = np.concatenate((beta0, X_sw), axis=1)

# Generate a cluster variable
clusters = larry(insurance_data_red.index.get_level_values(v_id))

# Set up a DataFrame for the switching results
switchreg = pd.DataFrame(np.zeros(shape=(X_sw.shape[1]+2,
                                         len(switching_measures_reg)*2)),
                      index=['y', 'stat',  'Constant'] + list(Xvars_sw))

# Go through all switching measures
for i, measure in enumerate(switching_measures_reg):
    # Get the variable in question
    var = switching_measures_reg[measure]

    # Make the LHS variable into a column vector
    y = larry(insurance_data_red[var])

    # Figure out where y and X are both not missing
    I = ~np.isnan(y[:,0]) & ~np.isnan(X_sw.sum(axis=1))

    # Run OLS
    bhat, _, _, p = ols(y[I,:], X_sw[I,:], cov_est='cluster',
                        clustvar=clusters[I,:])

    # Add outcome name to results DataFrame
    switchreg.iloc[0, 2*i:2*i+2] = measure

    # Add column labels for beta_har and p-value
    switchreg.iloc[1, 2*i] = 'b'
    switchreg.iloc[1, 2*i+1] = 'p'

    # Add results
    switchreg.iloc[2:, 2*i] = bhat[:, 0]
    switchreg.iloc[2:, 2*i+1] = p[:, 0]

# Set outcomes and beta_hat / p-values as headers for switching results
switchreg = switchreg.T.set_index(['y', 'stat']).T

# Save the result as a LaTeX table
# Set file name
fname_sw = 'switching_regressions.tex'

# Rename index objects to LaTeX names
switchreg = switchreg.rename(Xvars_sw)

# Save the table (the reset_index() makes sure the index is includes as a
# column in the output)
textable(switchreg.reset_index().values, fname=fname_sw, prec=prec)

################################################################################
### Part 3.4: Correlates of dominated choices (regression)
################################################################################

# Select which variables to use on the RHS for dominance measures
Xvars_dom = {v_year: 'Year', v_sex: 'Sex', v_tenure: 'Tenure',
             v_tenure+suf2: r'$\text{Tenure}^2$', v_age: 'Age',
             v_age+suf2: r'$\text{Age}^2$', preflog+v_inc: 'Log(income)',
             v_rscore: 'Risk', v_rscore+suf2: r'$\text{Risk}^2$',
             v_tool: 'Tool', v_switch: 'Switch',
             v_rscore+infint+v_tenure: r'Risk $\times$ tenure',
             v_rscore+infint+v_sex: r'Risk $\times$ sex',
             v_rscore+infint+preflog+v_inc: r'Risk $\times$ log(income)',
             v_tool+infint+v_rscore: r'Tool $\times$ risk',
             v_tool+infint+v_sex: r'Tool $\times$ sex',
             v_tool+infint+v_tenure: r'Tool $\times$ tenure',
             v_tool+infint+v_switch: r'Tool $\times$ switch'}

# Make variables into a matrix
X_dom = larry(insurance_data_red[list(Xvars_dom)])

# Add the intercept
X_dom = np.concatenate((beta0, X_dom), axis=1)

# Set up a DataFrame for the dominance results
domreg = pd.DataFrame(np.zeros(shape=(X_dom.shape[1]+2,
                                      len(dominance_measures)*2)),
                      index=['y', 'stat',  'Constant'] + list(Xvars_dom))

# Go through all measures of dominated choices
for i, measure in enumerate(dominance_measures):
    # Get the variable in question
    var = dominance_measures[measure]

    # Make the LHS variable into a column vector
    y = larry(insurance_data_red[var])

    # Figure out where y and X are both not missing
    I = ~np.isnan(y[:,0]) & ~np.isnan(X_dom.sum(axis=1))

    # Run OLS
    bhat, _, _, p = ols(y[I,:], X_dom[I,:], cov_est='cluster',
                        clustvar=clusters[I,:])

    # Add outcome name to results DataFrame
    domreg.iloc[0, 2*i:2*i+2] = measure

    # Add column labels for beta_har and p-value
    domreg.iloc[1, 2*i] = 'b'
    domreg.iloc[1, 2*i+1] = 'p'

    # Add results
    domreg.iloc[2:, 2*i] = bhat[:, 0]
    domreg.iloc[2:, 2*i+1] = p[:, 0]

# Set outcomes and beta_hat / p-values as headers for dominance results
domreg = domreg.T.set_index(['y', 'stat']).T

# Save the result as a LaTeX table
# Set file name
fname_dom = 'dominance_regressions.tex'

# Rename index objects to LaTeX names
domreg = domreg.rename(Xvars_dom)

# Save the table (the reset_index() makes sure the index is includes as a
# column in the output)
textable(domreg.reset_index().values, fname=fname_dom, prec=prec)

################################################################################
### Part 3.5: Plan characteristics
################################################################################

# Specify plan characteristics under consideration
pchars = [v_pre, v_cov, v_svq]

# Make sure plan ID is recorded as an integer
insurance_data[v_pid] = insurance_data[v_pid].astype(int)

# Get the means by plan
charmeans = insurance_data.set_index(v_pid)[pchars].groupby([v_pid]).mean()

# Set file name
fname_chars = 'plan_characteristics.tex'

# This has integer and float data, but Numpy does not usually mix those two. Get
# the values in the results DataFrame.
charmeans = charmeans.reset_index().values

# Convert them to an object
charmeans = charmeans.astype(object)

# Make sure the first row is a row of integers
charmeans[:,0] = charmeans[:,0].astype(int)

# Save tex table
textable(charmeans, fname=fname_chars, prec=prec)

################################################################################
### Part 3.6: Correlates of having the tool (regression)
################################################################################

# Specify which tool indicators to look at
tool_measures = {'Tool': v_tool}

# Select which variables to use on the RHS for tool access
Xvars_tool = {v_year: 'Year', v_sex: 'Sex', v_tenure: 'Tenure',
             v_tenure+suf2: r'$\text{Tenure}^2$', v_age: 'Age',
             v_age+suf2: r'$\text{Age}^2$', preflog+v_inc: 'Log(income)',
             v_rscore: 'Risk', v_rscore+suf2: r'$\text{Risk}^2$'}

# Make variables into a matrix
X_tool = larry(insurance_data_red[list(Xvars_tool)])

# Add the intercept
X_tool = np.concatenate((beta0, X_tool), axis=1)

# Set up a DataFrame for the dominance results
toolreg = pd.DataFrame(np.zeros(shape=(X_tool.shape[1]+2,
                                       len(tool_measures)*2)),
                       index=['y', 'stat',  'Constant'] + list(Xvars_tool))

# Go through all measures of dominated choices
for i, measure in enumerate(tool_measures):
    # Get the variable in question
    var = tool_measures[measure]

    # Make the LHS variable into a column vector
    y = larry(insurance_data_red[var])

    # Figure out where y and X are both not missing
    I = ~np.isnan(y[:,0]) & ~np.isnan(X_tool.sum(axis=1))

    # Run OLS
    bhat, _, _, p = ols(y[I,:], X_tool[I,:], cov_est='cluster',
                        clustvar=clusters[I,:])

    # Add outcome name to results DataFrame
    toolreg.iloc[0, 2*i:2*i+2] = measure

    # Add column labels for beta_har and p-value
    toolreg.iloc[1, 2*i] = 'b'
    toolreg.iloc[1, 2*i+1] = 'p'

    # Add results
    toolreg.iloc[2:, 2*i] = bhat[:, 0]
    toolreg.iloc[2:, 2*i+1] = p[:, 0]

# Set outcomes and beta_hat / p-values as headers for tool access results
toolreg = toolreg.T.set_index(['y', 'stat']).T

# Save the result as a LaTeX table
# Set file name
fname_tool = 'tool_regressions.tex'

# Rename index objects to LaTeX names
toolreg = toolreg.rename(Xvars_tool)

# Save the table (the reset_index() makes sure the index is includes as a
# column in the output)
textable(toolreg.reset_index().values, fname=fname_tool, prec=prec)

################################################################################
### Part 4: Make figures
################################################################################

# Set main color (UM official blue is #00274c)
mclr = '#00274c'

# Set secondary color (UM official maize if #ffcb05)
sclr = '#ffcb05'

# Set edge color
eclr = 'black'

# Put both in a list
colors = [mclr, sclr]

# Make an extended palette of color. These are all taken from UM's style
# guide[1], and have cute names.
#
# [1]:  https://vpcomm.umich.edu/brand/style-guide/design-principles/colors
manycolors = [mclr, sclr, '#7a121c', '#655a52', '#cc6600', '#83b2a8', '#575294',
              '#9b9a6d']

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
bars = ax.bar([x for x in range(domfrac.shape[0])],
              domfrac.loc[:, 1].values, color=colors,
              edgecolor=eclr, hatch='')

# Set xticks
ax.set_xticks([2*x + .5 for x in range(len(dominance_measures))])

# Go through all bars and their patterns
for bar, pattern in zip(bars, patterns):
    # Set the pattern for the bars
    bar.set_hatch(pattern)

# Use plot names, change font size
ax.set_xticklabels(dominance_measures, fontsize=11)

# Don't display tick marks on the horizontal axis
ax.tick_params(axis='x', bottom='off')

# Set vertical axis label
ax.set_ylabel('Fraction', fontsize=11)

# Add a legend
ax.legend((bars[0], bars[1]), ('Without tool', 'With tool'), fontsize=9)

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
bars = ax.bar([x for x in range(switchfrac.shape[0])],
              switchfrac.loc[:, 1].values, color=colors,
              edgecolor=eclr)

# Set xticks
ax.set_xticks([2*x + .5 for x in range(len(switching_measures))])

# Go through all bars and their patterns
for bar, pattern in zip(bars, patterns):
    # Set the pattern for the bars
    bar.set_hatch(pattern)

# Use plot names, change font size and alignment
ax.set_xticklabels(switching_measures.keys(), fontsize=11,
                   verticalalignment='center')

# Don't display tick marks on the horizontal axis, add some padding so labels
# stay out of the chart area
ax.tick_params(axis='x', bottom='off', pad=20)

# Set vertical axis label
ax.set_ylabel('Fraction', fontsize=11)

# Add a legend
ax.legend((bars[0], bars[1]), ('Without tool', 'With tool'), fontsize=9)

# Make sure all labels are visible (otherwise, they might get cut off)
fig.tight_layout()

# Save the figure
plt.savefig(fname+ffmt)

# Close the figure (all figures, in fact)
plt.close('all')

################################################################################
### Part 4.3: Switching with and witout the tool over time
################################################################################

# Get years in the data
years = np.sort(insurance_data_red[v_year].unique())

# Specify name for switching fractions over time figure
fname = 'switch_over_time'

# Set up the chart
fig, ax = plt.subplots(1, 1, num=fname, figsize=(6.5, 1.5*6.5*(9/16)))

# Make a list of line styles
lstys = ['-', ':', '--', '-.']

# Go through all switching measures
for i, measure in enumerate(switching_measures):
    # Get variable and conditioning indices
    var, indices = switching_measures[measure]

    # Go through all conditioning indices
    for j, idx in enumerate(indices):
        # Get the name of the measure
        measure_name = measure.replace('\n ', '') + tabsuf[j]

        # Get only data for the chosen indexed group
        y = insurance_data_red.loc[idx, :]

        # Get only data for the years after the first one
        y = y.set_index(v_year).loc[years[1:], :]

        # Get the mean for the variable under consideration
        y = y[var].groupby([v_year]).mean()

        # Plot the line for this measure
        ax.plot(years[1:], y, label=measure_name,
                linestyle=lstys[i], linewidth=2+j, color=manycolors[i])

# Make a legend
fig.legend(loc='lower center', ncol=2, fontsize=9, handlelength=2.5)

# Set horizontal axis label
ax.set_xlabel('Year', fontsize=11)

# Set horizontal axis limits
ax.set_xlim(np.amin(years[1:]), np.amax(years[1:]))

# Set vertical axis label
ax.set_ylabel('Fraction', fontsize=11)

# Add some space below the figure
fig.subplots_adjust(bottom=0.24)

# Save the figure, making sure there is no unnecessary whitespace (this is
# similar to calling fig.tight_layout(), but better in this case, because it
# won't cut off the legend)
plt.savefig(fname+ffmt, bbox_inches='tight')

# Close the figure (all figures, in fact)
plt.close('all')

################################################################################
### Part 4.4: Dominated choices over time
################################################################################

# Specify name for dominated choices over time figure
fname = 'dominated_choices_over_time'

# Set up the chart
fig, ax = plt.subplots(1, 1, num=fname, figsize=(6.5, 6.5*(9/16)))

# Go through all dominated choice measures
for i, measure in enumerate(dominance_measures):
    # Get the variable
    var = dominance_measures[measure]

    # Get the name of the measure
    measure_name = measure.replace('\n ', '')

    # Get the data across years, take the mean within year
    y = insurance_data_red.set_index(v_year)[var].groupby([v_year]).mean()

    # Plot the line for this measure
    ax.plot(years, y, label=measure_name, linestyle=lstys[i], linewidth=2,
            color=manycolors[i])

# Make a legend
ax.legend(fontsize=9)

# Set horizontal axis limits
ax.set_xlim(np.amin(years[1:]), np.amax(years[1:]))

# Set the horizontal axis label
ax.set_xlabel('Year', fontsize=11)

# Set the vertical axis label
ax.set_ylabel('Fraction', fontsize=11)

# Save on whitespace
fig.tight_layout()

# Save the figure
plt.savefig(fname+ffmt)

# Close the figure (all figures, in fact)
plt.close('all')

################################################################################
### Part 4.5: Dominated choices against tenure
################################################################################

# Get tenure lengths in the data
tenures = np.sort(insurance_data_red[v_tenure].unique())

# Specify name for switching fractions over time
fname = 'dominated_choices_vs_tenure'

# Set up the chart
fig, ax = plt.subplots(1, 1, num=fname, figsize=(6.5, 6.5*(9/16)))

# Go through all dominated choice measures
for i, measure in enumerate(dominance_measures):
    # Get the variable
    var = dominance_measures[measure]

    # Get the name of the measure
    measure_name = measure.replace('\n ', '')

    # Get the data across tenure lengths, take the mean within tenure length
    y = insurance_data_red.set_index(v_tenure)[var].groupby([v_tenure]).mean()

    # Plot the line for this measure
    ax.plot(tenures, y, label=measure_name,
            linestyle=lstys[i], linewidth=2, color=manycolors[i])

# Make a legend
ax.legend(handlelength=2.5, fontsize=9)

# Set horizontal axis limits
ax.set_xlim(np.amin(tenures), np.amax(tenures))

# Set the horizontal axis label
ax.set_xlabel('Tenure length', fontsize=11)

# Set the vertical axis label
ax.set_ylabel('Fraction', fontsize=11)

# Save on whitespace
fig.tight_layout()

# Save the figure
plt.savefig(fname+ffmt)

# Close the figure (all figures, in fact)
plt.close('all')

################################################################################
### Part 4.6: Histograms (demographics)
################################################################################

# Specify name for histograms of demographic variables
fname = 'hist_dem'

# Specify variables for which to make histograms
histvars_dem = {v_age: 'Age', v_tenure: 'Tenure',
                preflog+v_inc: 'Log(income)', v_rscore: 'Risk score',
                v_year: 'Year', v_tool: 'Comparison tool'}

# Specify some variables which should be used as integer valued, i.e. the
# histogram bins should just be their unique values
intvars = [v_age, v_tenure, v_year, v_tool]

# Choose variables to censor
censor = [v_rscore, v_inc]

# Pick which perentile to censor at
censat = .01

# Calculate number of rows needed with two columns
nrows = np.int(np.ceil(len(histvars_dem)/2))

# Set up the chart
fig, ax = plt.subplots(nrows, 2, num=fname, figsize=(6.5, 1.5*6.5*(9/16)))

# Go through all variables to plot
for i, var in enumerate(histvars_dem):
    # Get the row and column index in the subplots
    ridx = np.int(np.floor(i/2))
    cidx = i - 2*ridx

    # Check whether the variable has to be winsorized
    if censor.count(var) > 0:
        # Get the winsorization cutoffs
        cutoffs = insurance_data_red[var].quantile([censat, 1-censat])

        I = ((cutoffs[censat] < insurance_data_red[var])
             & (insurance_data_red[var] < cutoffs[1-censat]))

        # Get the winsorized variable
        y = insurance_data_red.loc[I, var]
    else:
        # If not, just use it as is
        y = insurance_data_red[var]

    # Check whether this variable should be used as an integer valued one
    if (intvars.count(var) > 0):
        # If so, use its values as bins (the left edge of the lowest value is
        # used, hence the minimum minus .5, and then everything else plus that)
        bins = (
            [np.amin(y.unique()) -.5]
            + list(y.unique() + .5))

        # Sort them
        bins = np.sort(bins)
    else:
        # Otherwise, use the auto method to find bins
        bins = 'auto'

    # plot the histogram
    ax[ridx, cidx].hist(y, label = histvars_dem[var], color = sclr,
                        edgecolor = mclr, density = True, bins = bins,
                        linewidth = .5)

    # Check whether there are few enough bins
    if (len(bins) <= 16) and (type(bins) is not str):
        # If so, set the ticks to use each value
        ax[ridx, cidx].set_xticks(list(y.unique()))

        # Check if there are more than 5 bins
        if len(bins) >= 5:
            # If so, add some rotation to their labels
            ax[ridx, cidx].set_xticklabels(list(y.unique()), rotation = 45,
                                           horizontalalignment = 'right')

    # Add a horizontal axis label
    ax[ridx, cidx].set_xlabel(histvars_dem[var], fontsize=11)

    # Check whether this is at the left side of the graphs
    if cidx == 0:
        # Add a vertical axis label
        ax[ridx, cidx].set_ylabel('Density', fontsize=11)

# Save on whitespace
fig.tight_layout()

# Save the figure
plt.savefig(fname+ffmt)

################################################################################
### Part 4.7: Histograms (plan characteristics)
################################################################################

# Specify name for histograms
fname = 'hist_plan'

# Specify variables for which to make histograms
histvars_plan = {v_pre: 'Premium', v_cov: 'Coverage', v_svq: 'Service quality',
                 v_pid: 'Plan ID'}

# Prepare legend entries
lcs = 'Choice set'
lcho = 'Chosen plan'

# Specify some variables which should be used as integer valued, i.e. the
# histogram bins should just be their unique values
intvars = [v_pid]

catvars = [v_pid]

# Calculate number of rows needed with two columns
nrows = np.int(np.ceil(len(histvars_plan)/2))

# Set up the chart
fig, ax = plt.subplots(nrows, 2, num=fname, figsize=(6.5, 1.5* 6.5*(9/16)))

# Go through all variables to plot
for i, var in enumerate(histvars_plan):
    # Get the row and column index in the subplots
    ridx = np.int(np.floor(i/2))
    cidx = i - 2*ridx

    # Get plan variables, first for the choice set, then chosen quantities
    y1 = insurance_data[var]
    y2 = insurance_data_red[var]

    # Check whether the variable is categorical
    if (catvars.count(var) > 0):
        # If so, make a histogram
        # Check whether this variable should be used as an integer valued one
        if (intvars.count(var) > 0):
            # If so, use its values as bins (the left edge of the lowest value is
            # used, hence the minimum minus .5, and then everything else plus that)
            bins1 = (
                [np.amin(y1.unique()) -.5]
                + list(y1.unique() + .5))

            # Sort them
            bins1 = np.sort(bins1)
        else:
            # Otherwise, use the auto method to find bins
            bins1 = 'auto'

        # Plot the choice set histogram
        hgram1 = ax[ridx, cidx].hist(y1, color = sclr, alpha = .8,
                                     edgecolor = eclr, density = True,
                                     bins = bins1, linewidth = .5,)

        # Plot the chosen plan histogram, using the same bins
        hgram2 =ax[ridx, cidx].hist(y2, color = mclr, alpha = .3,
                                     edgecolor = eclr, density = True,
                                     bins = hgram1[1], fill = True,
                                     linewidth = .5)

        # Check whether there are few enough bins
        if (len(bins1) <= 16) and (type(bins1) is not str):
            # If so, set the ticks to use each value
            ax[ridx, cidx].set_xticks(list(y1.unique()))
    else:
        # Otherwise, plot empirical distribution functions
        edf1 = ax[ridx, cidx].plot(np.sort(y1),
                                     np.linspace(0, 1, len(y1), endpoint=True),
                                     color=sclr, linestyle = lstys[0])

        edf2 = ax[ridx, cidx].plot(np.sort(y2),
                                     np.linspace(0, 1, len(y2), endpoint=True),
                                     color=mclr, linestyle = lstys[1])

    # Add a horizontal axis label
    ax[ridx, cidx].set_xlabel(histvars_plan[var], fontsize=11)

    # Check whether this is at the left side of the graphs
    if cidx == 0:
        # Add a vertical axis label
        ax[ridx, cidx].set_ylabel('Density', fontsize=11)

# Add a legend, both for the EDFs and histograms
fig.legend(handles=[edf1[0], edf2[0], hgram1[2][0], hgram2[2][0]],
           labels=[lcs, lcho, lcs, lcho],
           loc='lower center', fontsize=9, ncol=2)

# Add some space below the figure
fig.subplots_adjust(bottom=0.17, hspace=.3)

# Save the figure
plt.savefig(fname+ffmt, bbox_inches='tight')

# Print a message that this is done
print('Done')
