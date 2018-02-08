qui{
// How to clean everything
clear *

// PS4Q5: Empirical application

// Check whether bcuse is installed, which is super useful for downloading
// Wooldridge's data sets
cap which bcuse

// If it's not, get it from SSC
if _rc{
	ssc install bcuse
	}

// Load the data
bcuse card

// 5a) Regress log wage on education, potential experience, an indicator for
// being black, and various location indicator variables
noi di _n "5a)"
noi reg lwage educ exper expersq black south smsa reg661-reg668 smsa66

// 5b) Regress education on all controls from a) plus a college proximity dummy
noi di _n "5b)"
noi reg educ exper expersq black south smsa reg661-reg668 smsa66 nearc4

// 5c) Same regression as a), but instrumenting education via college proximity
noi di _n "5c)"
noi ivregress 2sls lwage exper expersq black south smsa reg661-reg668 ///
	smsa66 (educ=nearc4)

// 5d) Include proximity to a two year college as another instrument for
// education
noi di _n "5d)"
noi ivregress 2sls lwage exper expersq black south smsa reg661-reg668 ///
	smsa66 (educ=nearc2 nearc4), first
	
// 5e) Check correlation between IQ and proximity to a for year college
noi di _n "5e)"
noi reg IQ nearc4

// 5f) Regress IQ on location dummies and college proximity
noi di _n "5f)"
noi reg IQ nearc4 smsa66 reg661 reg662 reg669
}
