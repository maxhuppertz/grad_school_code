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
}
