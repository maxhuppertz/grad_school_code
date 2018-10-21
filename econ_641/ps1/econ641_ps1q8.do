qui{
// I don't need to see how the sausage gets made, hence the quietly wrapper

// How to clean everything
clear*

// Get current working directory
loc mdir: pwd

// Specify data directory (doesn't have to exist)
loc ddir = "data"

// Specify whether to download gravity data
loc download_data = 1

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

// Change back to main directory
cd "`mdir'"
}
