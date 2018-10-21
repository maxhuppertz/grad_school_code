qui{
// I don't need to see how the sausage gets made, hence the quietly wrapper

// How to clean everything
clear*

// Get current working directory
loc mdir: pwd

// Specify data directory (has to exist, because this file uses trade flow
// data created by another question in this problem set!)
loc ddir = "data"

// Specify whether to download gravity data
loc download_data = 1

// Specify name of gravity data file (doesn't need to exist, if you specify that
// you'd like to download it)
loc data_file_grav = "gravdata.dta"

// 

// Try to change to data directory
cap cd "`mdir'/`ddir'"

// If there's an error code, it doesn't exist, which is a problem
if _rc{
	// Display an error message
	noi di _n "The data directory does not exist!" _n ///
		"This file uses data created by another program" _n ///
		"It expects to find these in the data directory" _n ///
		"It craves them!" _n ///
		"It needs them!" _n ///
		"Please fix this issue and try running this file again" _n ///
		"Thank you :)"
	
	// Abort the program
	exit
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
	
	// There's another nuisance folder in there I don't need
	loc nuisance = "__MACOSX"
	
	// rmdir deletes folders, but only with the /s option; /q makes sure that
	// Windows doesn't ask whether you're really sure you want to delete the
	// directory in question
	!rmdir `nuisance' /s /q
	}

// Read in gravity data
u `data_file_grav'

cd `mdir'
}
