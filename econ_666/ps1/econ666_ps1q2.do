qui{
********************************************************************************
*** Econ 666, PS1Q2: Replication and randomization inference
*** Replicates results from Ashraf, Field, and Lee (2014) --  AFL
*** Also computes randomization inference to see how robust the results are to
*** that
********************************************************************************

// Clear everything
clear *

// Get current working directory
loc mdir: pwd

// Specify data directory (doesn't have to exist)
loc ddir = "data"

// Specify whether to download data
loc download_data = 0

// Specify name of AFL data file (doesn't need to exist, if you specify that
// you'd like to download it)
loc data_file = "fertility_regressions.dta"

********************************************************************************
*** Part 1: Download data, load data set
********************************************************************************

// Try to change to data directory
cap cd "`mdir'/`ddir'"

// If there's an error code, it doesn't exist, so create it
if _rc{
	mkdir "`ddir'"
	
	// Change to data directory
	cd "`ddir'"
	}

if `download_data'{
	// Specify URL for AFL data set (it's in a .zip archive)
	loc afl2014_url = "https://www.aeaweb.org/aer/data/10407/20101434_data.zip"
	
	// Specify name for temporary local copy of the .zip archive
	loc local_file = "temp.zip"
	
	// Download archive
	copy "`afl2014_url'" "`local_file'"
	
	// Unzip it
	unzipfile "`local_file'"
	
	// Delete the .zip archive
	erase "`local_file'"
	
	// Specify name of parent directory (the folder this just created)
	loc pardir = "20101434_Data"
	
	// Specify path to data file I actually need
	loc subdir = "20101434_Data/Data"
	
	// Copy the data file into the data directory
	copy "`mdir'/`ddir'/`subdir'/`data_file'" "`mdir'/`ddir'/`data_file'", ///
		replace
	
	// Delete everything else
	shell rmdir "`pardir'" /s /q
	}

// Read in AFL data
u "`data_file'"

// Change back to main directory
cd "`mdir'"

********************************************************************************
*** Part 2: Replicate original ITT
********************************************************************************

// Specify name of main estimation sample identifier
loc v_insample = "ittsample4"

// Keep only units in the main sample
keep if `v_insample' == 1

// Specify outcome of interest
loc v_usedvoucher = "usedvoucher"

// Specify couple treatment variable
loc v_coupletreatment = "Icouples"

// Specify individual treatment variable
loc v_indivtreatment = "Iindividual"

// Define independent variables for the RHS, other than the treatment dummy (I
// copied and pasted this from AFL's code)
loc indep = "a16_3_ageinyrs hus3_3_ageinyears school hus1_highestschool step3_numchildren diffht_wtchild e1_ideal hus12_ideal step7_injectables step7_pill step7_usingany monthlyinc husmonthlyinc fertdesdiff2 mostfertile age40 timesincelastbirth"

// Define more independent variables for the RHS, coded as dummies (again, this
// was lifted straight from AFL's code)
loc dummy = "flag_survey d_a16_3_ageinyrs d_hus3_3_ageinyears d_school d_hus1_highestschool d_step3_numchildren d_diffht_wtchild d_e1_ideal d_hus12_ideal d_monthlyinc d_husmonthlyinc d_fertdesdiff2 d_mostfert d_age40 d_flag_s d_compound_num d_step7* d_times"

// Specify name for main results
loc res_main = "main_results"

// Specify name for main results without covariates
loc res_main_nocov = "main_results_nocov"

// Run the main regression (note that, like AFL, this uses homoskedastic
// standard errors)
reg `v_usedvoucher' `v_coupletreatment' `indep' `dummy'

// Store the estimates
est sto `res_main'

// Get mean outcome for individual treatment recipients
su `v_usedvoucher' if `v_indivtreatment' == 1  

// Add the individual treatment mean
estadd r(mean)

// Run the regression using only the treatment dummy, without covariates
reg `v_usedvoucher' `v_coupletreatment'

// Store the estimates
est sto `res_main_nocov'

// Get mean outcome for individual treatment recipients
su `v_usedvoucher' if `v_indivtreatment' == 1

// Add the individual treatment mean to this regression as well
estadd r(mean)

// Display the result using esttab. (This requires the estout package. If you
// don't have it, type ssc install estout, which should download and install it
// automatically.)
noi esttab `res_main' `res_main_nocov', keep(`v_coupletreatment') b(%8.3f) ///
	se mtitles("Main results" "No covariates") star(* .1 ** .05 *** .01) ///
	stats(N mean, label("N" ///
	"Mean indiv. treat.") fmt(0 3)) ///
	coeflabel(`v_coupletreatment' "Couple treament") ///
	varwidth(18) modelwidth(13)
}
