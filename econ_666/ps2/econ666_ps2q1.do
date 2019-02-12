qui{
********************************************************************************
*** Econ 666, PS2Q1: Multiple testing corrections
*** Replicates results from Ashraf, Field, and Lee (2014) --  AFL
*** Uses various multiple testing corrections to see whether they are robust
*** to that
********************************************************************************

// Clear everything
clear *

// Set random number generator's seed
set se 666

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
*** Part 2: Variable generation
********************************************************************************

// Generate an indicator for whether the woman believes her partner wants at
// least more children than she does
loc v_minkids_self = "e8minnumber"
loc v_minkids_husb = "e19minnumber_hus"
loc v_husb_more_minkids = "husb_more_kids"
gen `v_husb_more_minkids' = (`v_minkids_husb' > `v_minkids_self') 
replace `v_husb_more_minkids' = . if `v_minkids_husb'==. | `v_minkids_self'==.

// Generate an indicator for whether the woman believes her husband ideally
// wants more children than she does
loc v_idkids_self = "e1_ideal"
loc v_idkids_husb = "e12_hus_ideal"
loc v_idkids_husb_miss = "d_e12_hus_ideal"
loc v_husb_more_idkids = "husb_more_idkids"
gen `v_husb_more_idkids' = (`v_idkids_husb'>`v_idkids_self') 
replace `v_husb_more_idkids' = . if `v_idkids_husb_miss'==1

// Generate an indicator for whether the woman believes her partner wants
// a higher maximum number of children than she does
loc v_maxkids_self = "e7maxnumber"
loc v_maxkids_husb = "e18maxnumber_husb"
loc v_husb_more_maxkids = "husb_more_maxkids"
gen `v_husb_more_maxkids' = (`v_maxkids_husb'>`v_maxkids_self')
replace `v_husb_more_maxkids' = . if `v_maxkids_husb'==. | `v_maxkids_self'==.

// Generate an indicator for whether the couple currently have fewer children
// than the husband would ideally want to have
loc v_num_kids = "currentnumchildren"
loc v_how_many_more = "e17morekids"  // How many more kids does the woman
// believe her husband wants?
loc v_husb_wants_kids = "husb_wants_kids"
gen `v_husb_wants_kids' = (((`v_idkids_husb'-`v_num_kids')>0) | `v_how_many_more'>0 )
replace `v_husb_wants_kids' = . if (`v_idkids_husb_miss'==1 | `v_num_kids'==.) & (`v_how_many_more'==-9)

// Specify variable name for indicator of whether the woman wants kids in the
// next two years
loc v_nokids_now = "wantschildin2"

// Generate an indicator for responder status
loc v_responder = "responder"
gen `v_responder' = ((`v_husb_more_minkids'==1 | `v_husb_more_idkids'==1 | `v_husb_more_maxkids'==1) &  `v_husb_wants_kids'==1 & `v_nokids_now'==0)
replace `v_responder' = . if ((`v_husb_more_minkids'==. & `v_husb_more_idkids'==. & `v_husb_more_maxkids'==.) | `v_husb_wants_kids'==. | `v_nokids_now'==.)

********************************************************************************
*** Part 3: ITT using only necessary covariates
********************************************************************************

// Specify name of main estimation sample identifier
loc v_insample = "ittsample4"

// Keep only responders in the main sample
keep if `v_insample' == 1

// Specify outcome of interest
loc v_usedvoucher = "usedvoucher"

// Specify couple treatment variable
loc v_coupletreatment = "Icouples"

keep `v_usedvoucher' `v_coupletreatment' separated2 violence_follow cur_using_condom satisfied healthier happier
}
