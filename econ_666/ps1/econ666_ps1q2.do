qui{
********************************************************************************
*** Econ 666, PS1Q2: Replication and randomization inference
*** Replicates results from Ashraf, Field, and Lee (2014) --  AFL
*** Also computes randomization inference to see how robust the results are to
*** that
********************************************************************************

// Clear everything
clear *

// Set random number generator's seed
set seed 666

// Get current working directory
loc mdir: pwd

// Specify data directory (doesn't have to exist)
loc ddir = "data"

// Specify whether to download data
loc download_data = 0

// Specify name of AFL data file (doesn't need to exist, if you specify that
// you'd like to download it)
loc data_file = "fertility_regressions.dta"

********************************************************************************
*** Part 1: Download data, load data set
********************************************************************************

// Try to change to data directory
cap cd "`mdir'/`ddir'"

// If there's an error code, it doesn't exist, so create it
if _rc{
	mkdir "`ddir'"
	
	// Change to data directory
	cd "`ddir'"
	}

if `download_data'{
	// Specify URL for AFL data set (it's in a .zip archive)
	loc afl2014_url = "https://www.aeaweb.org/aer/data/10407/20101434_data.zip"
	
	// Specify name for temporary local copy of the .zip archive
	loc local_file = "temp.zip"
	
	// Download archive
	copy "`afl2014_url'" "`local_file'"
	
	// Unzip it
	unzipfile "`local_file'"
	
	// Delete the .zip archive
	erase "`local_file'"
	
	// Specify name of parent directory (the folder this just created)
	loc pardir = "20101434_Data"
	
	// Specify path to data file I actually need
	loc subdir = "20101434_Data/Data"
	
	// Copy the data file into the data directory
	copy "`mdir'/`ddir'/`subdir'/`data_file'" "`mdir'/`ddir'/`data_file'", ///
		replace
	
	// Delete everything else
	shell rmdir "`pardir'" /s /q
	}

// Read in AFL data
u "`data_file'"

// Change back to main directory
cd "`mdir'"

********************************************************************************
*** Part 2: Replicate original ITT
********************************************************************************

// Specify name of main estimation sample identifier
loc v_insample = "ittsample4"

// Keep only units in the main sample
keep if `v_insample' == 1

// Specify outcome of interest
loc v_usedvoucher = "usedvoucher"

// Specify couple treatment variable
loc v_coupletreatment = "Icouples"

// Specify individual treatment variable
loc v_indivtreatment = "Iindividual"

// Define independent variables for the RHS, other than the treatment dummy (I
// copied and pasted this from AFL's code)
loc indep = "a16_3_ageinyrs hus3_3_ageinyears school hus1_highestschool step3_numchildren diffht_wtchild e1_ideal hus12_ideal step7_injectables step7_pill step7_usingany monthlyinc husmonthlyinc fertdesdiff2 mostfertile age40 timesincelastbirth"

// Define more independent variables for the RHS, coded as dummies (again, this
// was lifted straight from AFL's code)
loc dummy = "flag_survey d_a16_3_ageinyrs d_hus3_3_ageinyears d_school d_hus1_highestschool d_step3_numchildren d_diffht_wtchild d_e1_ideal d_hus12_ideal d_monthlyinc d_husmonthlyinc d_fertdesdiff2 d_mostfert d_age40 d_flag_s d_compound_num d_step7* d_times"

// Specify name for main results
loc res_main = "main_results"

// Run the main regression (note that, like AFL, this uses homoskedastic
// standard errors)
reg `v_usedvoucher' `v_coupletreatment' `indep' `dummy'

// Store the estimates
est sto `res_main'

// Get mean outcome for individual treatment recipients
su `v_usedvoucher' if `v_indivtreatment' == 1  

// Add the individual treatment mean
estadd r(mean)

********************************************************************************
*** Part 3: ITT using only necessary covariates
********************************************************************************

// Specify name for main results without covariates
loc res_main_nocov = "main_results_nocov"

// Run the regression using only the treatment dummy, without covariates
reg `v_usedvoucher' `v_coupletreatment'

// Store the estimates
est sto `res_main_nocov'

// Get mean outcome for individual treatment recipients
su `v_usedvoucher' if `v_indivtreatment' == 1

// Add the individual treatment mean to this regression as well
estadd r(mean)

********************************************************************************
*** Part 4: Fake covariate #1
********************************************************************************

// Specify name for fake covariate #1
loc v_fc1 = "fc1"

// Specify mean for fake covariate #1
loc mean_fc1 = 0

// Specify variance for error term of fake covariate #1
loc var_eps_fc1 = 1

// Summarize dependent variable (voucher use indicator), save the variance
su `v_usedvoucher'
loc var_Y = r(sd)^2

// Specify correlation between voucher use indicator and fake covariate #1
loc rho_Yfc1 = .7

// Generate fake covariate #1
// How does this work? Let Y denote the voucher use indicator. Let Z denote fake
// covariate #1. I want to achieve
//
// Corr(Y,Z) = Cov(Y,Z) / sqrt(Var(Y) Var(Z)) = gamma                     (1)
//
// for some gamma. I can generate
//
// Z = alpha + beta_Z*Y + Z_eps                                           (2)
//
// where Z_eps is an error term, if you will. Expanding Cov(Y,Z) and
// plugging in (2) yields Cov(Y,Z) = beta_Z*Var(Y). Also, taking the
// variance of (2), I have Var(Z) = beta_Z^2*Var(Y) + Var(Z_eps). Plugging
// both of these into (1) gives
//
// beta_Z = sqrt( (Var(Y) / Var(Z_eps)) * (gamma^2 / (1 - gamma^2)) )
//
// and since I get to choose beta_Z, I can thereby generate random
// variables with arbitrary correlation structure. I can then use alpha to
// adjust the mean of the generated variable.

// Calculate beta_Z for fake covariate #1
loc beta_fc1 = sqrt((`var_eps_fc1'/`var_Y') * (`rho_Yfc1'^2/(1 - `rho_Yfc1'^2)))

// Generate fake covariate #1
gen `v_fc1' = ///
	`mean_fc1' + `beta_fc1'*`v_usedvoucher' + rnormal(0, sqrt(`var_eps_fc1'))

// Specify name for main results with fake covariate number one
loc res_main_fc1 = "main_results_fc1"
	
// Run the regression using only the treatment dummy, without covariates
reg `v_usedvoucher' `v_coupletreatment' `v_fc1'

// Store the estimates
est sto `res_main_fc1'

// Get mean outcome for individual treatment recipients
su `v_usedvoucher' if `v_indivtreatment' == 1

// Add the individual treatment mean to this regression as well
estadd r(mean)

********************************************************************************
*** Part 5: Fake covariate #2
********************************************************************************

// Specify name for fake covariate #2
loc v_fc2 = "fc2"

// Specify mean for fake covariate #2
loc mean_fc2 = 0

// Specify variance for error term of fake covariate #2
loc var_eps_fc2 = 10

// Get residuals from the regression of the voucher use indicator on only the
// treatment dummy (only a temporary variable)
est res `res_main_nocov'
predict temp  // Get fitted values
replace temp = `v_usedvoucher' - temp  // Get residuals

// Summarize residuals, save the variance
su temp
loc var_Y_res = r(sd)^2

// Specify correlation between voucher use indicator residuals and fake
// covariate #2
loc rho_Yfc2 = .7

// Calculate beta_Z for fake covariate #2 (see the note in the preceding section
// on how this works)
loc beta_fc2 = ///
	sqrt((`var_eps_fc2'/`var_Y_res') * (`rho_Yfc2'^2/(1 - `rho_Yfc2'^2)))

// Generate fake covariate #2
gen `v_fc2' = ///
	`mean_fc2' + `beta_fc2'*temp + rnormal(0, sqrt(`var_eps_fc2'))

// Drop the temporary variable
drop temp	

// Specify name for main results with fake covariate number one
loc res_main_fc2 = "main_results_fc2"
	
// Run the regression using only the treatment dummy, without covariates
reg `v_usedvoucher' `v_coupletreatment' `v_fc2'

// Store the estimates
est sto `res_main_fc2'

// Get mean outcome for individual treatment recipients
su `v_usedvoucher' if `v_indivtreatment' == 1

// Add the individual treatment mean to this regression as well
estadd r(mean)

********************************************************************************
*** Part 6: Permutation p-value
********************************************************************************

noi est res `res_main_nocov'
mat b_hat_orig = e(b)
mat V_hat_orig = e(V)
loc t_tau_orig = b_hat_orig[1,1] / sqrt(V_hat_orig[1,1])

// Calculate number of treated units in the data
su `v_coupletreatment'
loc n_treat = r(sum)

// Specify name for treatment reassignment variable, generate it as zeros
loc v_treat_reassign = "treat_reassign"
gen `v_treat_reassign' = 0

// Specify how many simulations draws to use. (Getting the exact randomization
// distribution is not going to work here. There are 749 units in the sample, of
// whom 371 are treated. And
//
// 749! / (371!*(749-317)!)
//
// is very large.)
loc nrdmax = 100

// Set up a counter for how many results were more extreme
loc n_more_extreme = 0

// Go through all simulations
forval i=1/`nrdmax'{
	// Draw an N(0,1) random variable as the basis for randomization
	replace `v_treat_reassign' = rnormal(0,1)
	
	// Sort observations based on their random draws
	sort `v_treat_reassign'
	
	// Get the randomized treatment assignment
	replace `v_treat_reassign' = (_n <= `n_treat')
	
	// Run regression for this new assignment
	reg `v_usedvoucher' `v_treat_reassign'
	
	// Get t-statistic for the treatment coefficient
	mat b_hat_rnd = e(b)
	mat V_hat_rnd = e(V)
	loc t_tau_rnd = b_hat_rnd[1,1] / sqrt(V_hat_rnd[1,1])
	
	// If it is more extreme than the one observed originally, increase the
	// counter
	if abs(`t_tau_rnd') > abs(`t_tau_orig'){
		loc ++n_more_extreme  // Increases the counter by one
		}
}

// Display the result
noi di _n "Permutation p-value: " `n_more_extreme'/`nrdmax'

********************************************************************************
*** Part 7: Display the results
********************************************************************************

// Display the result using esttab. (This requires the estout package. If you
// don't have it, type ssc install estout, which should download and install it
// automatically.)
noi esttab `res_main' `res_main_nocov' `res_main_fc1' `res_main_fc2', ///
	keep(`v_coupletreatment') b(%8.3f) ///
	se mtitles("Main results" "No covariates" ///
	"Fake covariate 1" "Fake covariate 2") ///
	star(* .1 ** .05 *** .01) ///
	stats(N mean, label("N" ///
	"Mean indiv. treat.") fmt(0 3)) ///
	coeflabel(`v_coupletreatment' "Couple treament") ///
	varwidth(18) modelwidth(16)
}
