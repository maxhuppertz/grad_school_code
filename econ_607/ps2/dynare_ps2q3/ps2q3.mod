// Set up variables
var ITC K I P;
varexo eps;

// Set up parameters
parameters rho tau_K theta alpha r delta A ITC_bar sig
	K_st I_st P_st;

// Load parameter values from .mat file Matlab created
load ps2q3_init_params.mat;

// Go through all parameter names and set them to those values
// For deep parameters, doing this rather than just using load may matter
// The index is not i, because this is called on in a Matlab loop, and if
// indices overlap between this and the loop it runs in, things get really
// funky
for pnum=1:length(M_.params);
	// Find the parameter name matching the current value
    param_name = M_.param_names(pnum, :);
    
	// Set the parameter to that value
	eval(['M_.params(pnum)  = ' param_name ' ;'])
end;

// Set up model
model;
ITC = (1 - rho)*ITC_bar + rho * ITC(-1) + eps;
exp(K) = (alpha * (1 - tau_K) / ((delta + r) * (1 - ITC) * 
    exp(P)))^(1 / (1 - alpha));
exp(P) = (exp(I) / A)^(1 / theta);
exp(I) = exp(K) - (1 - delta)*exp(K(-1));
end;

// Specify initial state
initval;
K = log(K_st);
I = log(I_st);
P = log(P_st);
ITC = ITC_bar;
end;

// Specify shock process
shocks;
var eps = sig^2;
end;

// Calculate steady state
steady;

// Do a stochastic simulation, save impulse response graphs
stoch_simul(order=1, irf=200, graph_format=eps, nodisplay, noprint);