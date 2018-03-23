qui{
// How to clean everything
clear *

// PS8Q3: Empirical application

// Check whether bcuse is installed, which is super useful for downloading
// Wooldridge's data sets
cap which bcuse

// If it's not, get it from SSC
if _rc{
	ssc install bcuse
	}

// Load the data
bcuse grogger

// 3a/b)
// Specify variable counting the number of times people were arrested in 1986,
// generate indicator based on that
loc v_narr86 = "narr86"
loc i_arr86 = "arr86"
gen `i_arr86' = (`v_narr86' > 0)

// Specify set of regressors, estimate linear probability model
loc X1 = "pcnv avgsen tottime ptime86 inc86 black hispan born60"
noi di _n "3a/b) LPM, homoskedastic standard errors and test:"
noi reg `i_arr86' `X1'
noi test avgsen tottime
noi di _n "LPM, heteroskedasticity-robust standard errors and test:"
noi reg `i_arr86' `X1', r
noi test avgsen tottime

// 3c)
// Estimate probit model, display partial effect of pcnv at the means of avgsen,
// tottime, inc86, ptime86, and with black = 1, hispan = 0, born60 = 1
// Note that using 'robust' standard errors is kind of problematic, since the
// MLE of the probit model is in fact inconsistent in the presence of hetero-
// skedasticity, and using an inconsistent estimator sort of makes getting the
// standard errors right a moot point; fracreg uses an MLE estimator that is
// actually consistent in the face of some kinds of heteroskedasticity
// (specifying robust is not necessary, it's the default option)
// See also:
// https://blog.stata.com/2016/08/30/two-faces-of-misspecification-in-maximum-likelihood-heteroskedasticity-and-robust-standard-errors/
// http://davegiles.blogspot.com/2013/05/robust-standard-errors-for-nonlinear.html
noi di _n "3c) Probit plus marginal effect of pcnv"
noi fracreg probit `i_arr86' `X1'
noi margins, at(black=1 hispan=0 born60=1 pcnv=(.25 .75)) atmeans

// 3d)
// Add regressors, reestimate probit model
loc X2 = "pcnvsq pt86sq inc86sq"
noi di _n "3d) Probit with additional regressors plus test," ///
	" predicted probability curve:"
noi fracreg probit `i_arr86' `X1' `X2'
noi test `X2'

// Set up a table of marginal effects at various values of pcnv
loc mcol = 27
noi di _n "Probability of conviction" _col(`mcol') "{c |} Predicted probability"
noi di _"{hline `=`mcol'-1'}{c +}{hline 23}" _n

// Set up locald containing the previous marginal effect and an indicator of
// whether the tipping point was reached
loc meff_p = .
loc tpreached = 0

// Specify stepsize for the table using a power of ten (i.e. this goes from 0 to
// 1 in steps of 10^(-`decpower')
loc decpower = 4

// Choose how many steps to report
loc nsteps = 20

// Go through values for conviction proabilities
forval pconv = 0(`=10^(-`decpower')')1{
	// Estimate predicted probability
	margins, at(black=1 hispan=0 born60=1 ///
		pcnv=(`pconv') pcnvsq=(`=`pconv'^2')) atmeans
	
	// Get the coefficient matrix, extract marginal effect (well, it's really
	// the predicted probability, but it'll be used to gauge whether the
	// marginal effect turned negative, so...)
	mat b = r(b)
	loc meff = b[1,1]
	
	// If the marginal effect turned negative, record the tipping point (note
	// that the whole tipping point thing is in here only for convenience; this
	// is not meant to be an efficient way to find the point at which the
	// marginal effect turns negative, it's a terribly wasteful algorithm, and
	// also there exists an analytical solution to that question; it's just a
	// little add on to save me some algebra because I am lazy)
	if (`meff' < `meff_p') & (`meff_p' != .) & !`tpreached'{
		loc tp = `pconv'
		loc tpreached = 1
		}
	
	// If this is one of the steps that's being reported, report it
	if mod(round(`pconv'*10^`decpower'), (1/`nsteps')*10^`decpower') ///
		< 10^(-`decpower'){
		noi di "`=string(`pconv', "%12.02f")'" ///
			_col(`mcol') "{c |} `meff'"
		}
	// Replace previous predicted probability with current one
	loc meff_p = `meff'
	}
// Display the point at which the marginal effect turns negative
noi di _n "The marginal effect turns negative at: " ///
	"`=string(round(`tp', 10^(-`decpower')), "%12.0`decpower'f")'"	
}
