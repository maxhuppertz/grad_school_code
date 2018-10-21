qui{
// I don't need to see how the sausage gets made, hence the quietly wrapper

// How to clean everything
clear*

// Get current working directory
loc mdir: pwd

// Specify data directory (doesn't have to exist)
loc ddir = "data"

// Specify whether to download data
loc download_data = 1

// Specify name of data file
loc data_file = "gravdata.dta"

// Try to change to data directory
cap cd "`mdir'/`ddir'"

// If there's an error code, it doesn't exist, so create it and change to it
if _rc{
	// Create the data directory
	mkdir `ddir'
	
	// Change to data directory
	cd `ddir'
	}

if `download_data'{
	// Specify CEPII gravity data set URL (it's in a .zip archive)
	loc cepii_url = ///
		"https://www.dropbox.com/s/dsxwq07j73yyjek/gravdata.zip?dl=1"
	
	// Specify name for local copy of the .zip archive
	loc local_file = "temp.zip"
	
	// Download archive
	copy "`cepii_url'`web_file'" "`local_file'"
	
	// Unzip it
	unzipfile `local_file'
	
	// Delete the .zip archive
	erase `local_file'
	}

// Read in gravity data
u `data_file'

cd `mdir'
}
