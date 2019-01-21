qui{
// Clear everything
clear *

// 1a)
// Specify number (N) of people assigned to treatment (a) who end up being
// treated (t), as well as their (mean) outcome Y
loc Nat = 200
loc Yat = .5

// Specify number of people assigned to treatment who end up not being treated
// (n), and their Y
loc Nan = 100
loc Yan = .2

// Calculate total number of people assigned to treatment NA
loc NA = `Nat' + `Nan'

// Specify number of people not assigned to treatment NN, and their Y
loc NN = 300
loc YN = .3

// Calculate ITT
loc ITT = (`Nat'*`Yat' + `Nan'*`Yan')/`NA' - `YN'

// Display the result
noi di _n "1a)" _n "ITT = `ITT'"

// 1b)
// Calculate CCM
loc CCM = (`YN' - (`Nan'/`NA')*`Yan')/(1 - (`Nan'/`NA'))

// Display the result
noi di _n "1b)" _n "CCM = `CCM'"

// 1c)
// Specify number of people assigned to control who end up being treated, and
// their Y
loc Nnt = 50
loc Ynt = .4

// Specify number of people assigned to control who end up not being treated,
// and their Y
loc Nnn = 250
loc Ynn = .15

// Calculate ITT
loc ITT = (`Nat'*`Yat' + `Nan'*`Yan')/`NA' - (`Nnt'*`Ynt' + `Nnn'*`Ynn')/`NN'

// Display the result
noi di _n "1b)" _n "ITT = `ITT'"

// 1d)
// Calculate CCM

}
