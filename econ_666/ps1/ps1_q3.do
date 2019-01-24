qui{
// Clear everything
clear *

// Set random number generator's seed
set seed 666

// Set number of tuples
loc T = 100

// Set the number of observations equal to the number of tuples
set obs `T'

// Generate all variables, to make memory allocation easier
gen X = .
gen Y0 = .
gen tau = .

// Specify correlation pairs, where the first element is Corr(X,Y0), and the
// second is Corr(X,tau). The two are connected by an ampersand
loc Corrs = "0&0 0.1&0.1 0.6&0.1 0.1&0.6"

// Specify sample sizes
loc Ns = "10 25 `T'"

// Specify treatment probabilities
loc Ps = "0.3 0.5"

// Go through all correlation pairs
foreach corr of loc Corrs{
	// Split the correlation pair at the ampersand
	di regexm("`corr'", "^([^&]+)&([^&]+)")
	
	// Retrieve the two parts of the pair
	loc corrXY0 = regexs(1)
	loc corrXtau = regexs(2)
	
	// Specify correlation matrix for the current iteration
	mat C = (1, `corrXY0', `corrXtau' \ `corrXY0', 1, 0 \ `corrXtau', 0, 1)
	
	// Generate data for the current iteration, which will have the specified
	// correlation structure
	corr2data X Y0 tau, corr(C) clear
	
	// Go through all combinations of sample sizes and treatment probabilities
	foreach n of loc Ns{
		foreach p of loc Ps{
			di ""
		}
	}
}
}
