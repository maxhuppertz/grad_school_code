qui{
// How to clean everything
clear *

// PS9Q3: Empirical application

// Check whether bcuse is installed, which is super useful for downloading
// Wooldridge's data sets
cap which bcuse

// If it's not, get it from SSC
if _rc{
	ssc install bcuse
	}

// Load the data
bcuse fringe

// 3a)
// Estimate OLS model
noi di _n "3a) OLS estimates"
loc X1 = "exper age educ tenure married male white nrtheast nrthcen south union"
loc y = "hrbens"
noi reg `y' `X1', r

// 3b)
// Estimate Tobit model
noi di _n "3b) Tobit estimates"
noi tobit `y' `X1', ll(0) vce(r)

// 3c)
// Add squared terms, estimate Tobit, test joint significance of squared terms
noi di _n "3c) Tobit including squared terms"
loc X2 = "expersq tenuresq"
noi tobit `y' `X1' `X2', ll(0) vce(r)
noi test `X2'
}
