qui{
// I don't need to see how the sausage gets made, hence the quietly wrapper

// How to clean everything
clear*

// Get current working directory
loc mdir: pwd

// Specify data directory (doesn't have to exist)
loc ddir = "data"

// Specify whether to download data


// Check whether data directory exists
cap cd "`mdir'/`ddir'"

// If not, i.e. if there's an error core (_rc != 0), create it
if _rc{
	mkdir `ddir'
	}

// Change to data directory
cd `ddir'
cd `mdir'
}
