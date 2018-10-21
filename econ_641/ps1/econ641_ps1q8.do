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

// Specify trade flows and origin/destination GDP variables
loc v_flow = "flow"
loc v_Y_orig = "gdp_o"
loc v_Y_dest = "gdp_d"

// Generate a variable capturing the ratio of trade flows to the product of GDPs
loc v_expratio = "exp_ratio"
gen `v_expratio' = `v_flow' / (`v_Y_orig' * `v_Y_dest')

// Generate the log of that
loc v_log_expratio = "log_`v_expratio'"
gen `v_log_expratio' = log(`v_expratio')

// Specify distance variable
loc v_dist = "distw"

// Generate the log of that
loc v_log_dist = "log_`v_dist'"
gen `v_log_dist' = log(`v_dist')

// Specify dummies for contiguity, common language, and colonial ties
loc i_contig = "contig"
loc i_lang = "comlang_off"
loc i_colo = "col_hist"

// Specify year variable
loc v_year = "year"

// Put all of the log iceberg trade cost variables into a macro
loc log_iceberg = "`v_log_dist' `i_contig' `i_lang' `i_colo'"

// Specify a condition
loc cond = "if `v_year' == 2000"

// Regress log expenditure ratios on log trade costs and fixed effects
reg `v_expratio' `log_iceberg' i.`i_orig'`suf_ds' i.`i_dest'`suf_ds' ///
	`cond', vce(robust)

// Display only the interesting estimates (i.e. not the fixed effects)
noi est table, keep(`log_iceberg') b se t p

// Change back to main directory
cd "`mdir'"
}
