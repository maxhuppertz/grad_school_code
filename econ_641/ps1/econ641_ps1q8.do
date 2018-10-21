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

loc i_orig = "iso_o"
loc i_dest = "iso_d"
loc destring_vars = "`i_orig' `i_dest'"
loc suf_ds = "_ds"

foreach var of loc destring_vars{
	egen `var'`suf_ds' = group(`var')
	}

loc v_year = "year"

loc v_flow = "flow"
loc v_Y_orig = "gdp_o"
loc v_Y_dest = "gdp_d"
loc v_expratio = "exp_ratio"
gen `v_expratio' = `v_flow' / (`v_Y_orig' * `v_Y_dest')
loc v_log_expratio = "log_`v_expratio'"
gen `v_log_expratio' = log(`v_expratio')

loc v_dist = "distw"
loc v_log_dist = "log_`v_dist'"
gen `v_log_dist' = log(`v_dist')

loc i_contig = "contig"
loc i_lang = "comlang_off"
loc i_colo = "col_hist"

loc log_iceberg = "`v_log_dist' `i_contig' `i_lang' `i_colo'"

noi reg `v_expratio' `log_iceberg' `i_orig'`suf_ds'#`v_year' `i_dest'`suf_ds'#`v_year', vce(robust)

// Change back to main directory
cd "`mdir'"
}
