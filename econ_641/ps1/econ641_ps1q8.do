qui{
// I don't need to see how the sausage gets made, hence the quietly wrapper

// How to clean everything
clear*

set maxvar 32767
set matsize 11000

// Get current working directory
loc mdir: pwd

// Specify data directory (doesn't have to exist)
loc ddir = "data"

// Specify whether to download gravity data
loc download_data = 0

// Specify name of gravity data file (doesn't need to exist, if you specify that
// you'd like to download it)
loc data_file = "col_regfile09.dta"

// Try to change to data directory
cap cd "`mdir'/`ddir'"

// If there's an error code, it doesn't exist, so create it
if _rc{
	mkdir "`ddir'"
	
	// Change to data directory
	cd "`ddir'"
	}

if `download_data'{
	// Specify CEPII gravity data set URL (it's in a .zip archive)
	loc cepii_url = "http://econ.sciences-po.fr/sites/default/files/file/tmayer/data/col_regfile09.zip"
	
	// Specify name for local copy of the .zip archive
	loc local_file = "temp.zip"
	
	// Download archive
	copy "`cepii_url'" "`local_file'"
	
	// Unzip it
	unzipfile "`local_file'"
	
	// Delete the .zip archive
	erase "`local_file'"
	}

// Read in gravity data
u "`data_file'"

// Specify year variable
loc v_year = "year"

// Specify indicator variables for origin and destination countries
loc i_orig = "iso_o"
loc i_dest = "iso_d"

// Put these into a macro to 'destring' them (this doesn't use the destring
// command, but it does create a categorical version of each variable)
loc destring_vars = "`i_orig' `i_dest'"
loc suf_ds = "_ds"

// 'Destring' them
foreach var of loc destring_vars{
	egen `var'`suf_ds' = group(`var')
	}

// Specify a reference country
loc reference_country = "USA"

// Set this to be the omitted level for regressions later on
foreach var of loc destring_vars{
	su `var'`suf_ds' if `var' == "`reference_country'"
	loc omit_`var' = r(mean)
	}

// Specify trade flows and origin/destination GDP variables
loc v_flow = "flow"
loc v_Y_orig = "gdp_o"
loc v_Y_dest = "gdp_d"

// Generate a variable capturing the ratio of trade flows to the product of GDPs
loc v_expratio = "exp_ratio"
gen `v_expratio' = `v_flow' / (`v_Y_orig' * `v_Y_dest')

// Specify distance variable
loc v_dist = "distw"

// Specify which variables to take logs of, and a prefix for that
loc logvars = "`v_dist' `v_flow' `v_Y_orig' `v_Y_dest' `v_expratio'"
loc pref_log = "log_"

// Take logs
foreach var of loc logvars{
	gen `pref_log'`var' = log(`var')
	}

// Specify dummies for contiguity, common language, and colonial ties
loc i_contig = "contig"
loc i_lang = "comlang_off"
loc i_colo = "col_hist"

// I want to use the opposite of these variables, though
loc flipdummies = "`i_contig' `i_lang' `i_colo'"

foreach var of loc flipdummies{
	replace `var' = 1 - `var'
	}

// Put all of the RHS variables (other than fixed effects) into a macro
// loc RHS = "`pref_log'`v_dist' `i_contig' `i_lang' `i_colo' `pref_log'`v_Y_orig' `pref_log'`v_Y_dest'"
loc RHS = "`pref_log'`v_dist' `i_contig' `i_lang' `i_colo'"

// Put the dependent variable (not its log!) into a macro
loc depvar = "`v_expratio'"

// Specify a condition
loc cond = "if `v_year' == 2000"

// Estimate the log model using OLS
reg `pref_log'`depvar' `RHS' ///
	io`omit_`i_orig''.`i_orig'`suf_ds' io`omit_`i_dest''.`i_dest'`suf_ds' ///
	`cond', vce(robust)
/*
// Display only coefficients of interest (i.e. not fixed effects)
noi est table, keep(`RHS' _cons) b se t p

// Estimate the Poisson PML
poisson `depvar' `RHS' i.`i_orig'`suf_ds' i.`i_dest'`suf_ds' `cond', ///
	vce(robust)

// Display only coefficients of interest (i.e. not fixed effects)
noi est table, keep(`RHS' _cons) b se t p

// Now, to test equality of the coefficient on log distance, STATA provides
// suest, which is nice, but also needs to reestimate the models I just
// estimated without robust standard errors, since it tags those on itself

// Reestimate the OLS model (without robust errors)
reg `pref_log'`depvar' `RHS' i.`i_orig'`suf_ds' i.`i_dest'`suf_ds' `cond'

// Store the results
est sto ols_model

// Reestimate the Poisson model (without robust errors)
poisson `depvar' `RHS' i.`i_orig'`suf_ds' i.`i_dest'`suf_ds' `cond'

// Store the results
est sto ppml_model

// Use suest to compare the coefficients (using robust errors); the names of the
// two models are slightly altered by suest, in that it adds _mean to the OLS
// and _exp_ratio to the Poisson model's name
suest ols_model ppml_model, vce(robust)
noi test [ols_model_mean]`pref_log'`v_dist' = ///
	[ppml_model_exp_ratio]`pref_log'`v_dist'
*/
// Change back to main directory
cd "`mdir'"
}
