qui{
// How to clean everything
clear *

// PS3Q6: Empirical application

// Check whether bcuse is installed, which is super useful for downloading
// Wooldridge's data sets
cap which bcuse

// If it's not, get it from SSC
if _rc{
	ssc install bcuse
	}

// Load the data
bcuse attend

// 4.14a) Regress standardized final exam score on on attendance rate and two
// dummies for freshman (which is NOT called fresh? Why?!) and sophomore status

// Specify y and X variables (this is not super beautiful code; specifically, it
// contains lots of basically hard coded variable names, but it's only a problem
// set, so cut me some slack, will you)
loc y = "stndfnl"
loc X1 = "atndrte frosh soph"

// Estimate the model
noi reg `y' `X1'

// 4.14c) Add prior cumulative GPA and ACT score to the model

// Add variables to the model
loc X2 = "priGPA ACT"

// Estimate the model
noi reg `y' `X1' `X2'

// 4.14e) Add squares of prior GPA and ACT score to the model

// In 4.14f), I'll have to add a square of the attendance rate as well, so I'll
// get ahead of myself here and just code that up and have STATA square it
// together with these variables
loc X4 = "atndrte"

// Generate squared variables (for priGPA the double precision actually might
// matter I think; without, some checks for priGPA2 != priGPA^2 come out
// positive)
foreach x of var `X2' `X4'{
	gen double `x'2 = `x'^2
	la var `x'2 "Squared version of `x'"
	
	// If this is one of the variables from c), add it to
	if regexm("`X2'", "`x'"){
		loc X3 = "`X3' `x'2"
		}
	else{
		loc X4 = "`x'2"
		}
	}

// Estimate the model
noi reg `y' `X1' `X2' `X3'

// Test for joint significance of the two squares
noi te `X3'

// 4.14f) Add a squared version of the attendance rate to the model

// Estimate the model
noi reg `y' `X1' `X2' `X3' `X4'

// Test for joint significance of attendance rate and its squared value, because
// why not?
noi te `=word("`X1'", 1)' `X4'
}
