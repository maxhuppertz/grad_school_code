// Set up variables
var y pi eps;
varexo eta;

// Set up parameters
parameters beta kappa rho phi_y phi_pi theta sig;

// Load parameter values from .mat file Matlab created
load ps8q2_init_params.mat;

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

// Set up model (somewhat lazy log-linearization, I'll admit)
model;
eps = rho * eps(-1) + eta;
y = -(1 - beta*rho) * (1 / ((1 - beta*rho) * (theta*(1 - rho) + phi_y) + kappa*(phi_pi - rho))) * eps;
pi = -kappa * (1 / ((1 - beta*rho) * (theta*(1 - rho) + phi_y) + kappa*(phi_pi - rho))) * eps;
end;

// Specify initial state
initval;
eps = 0;
y = 0;
pi = 0;
end;

// Specify shock process
shocks;
var eta = sig^2;
end;

// calculate steady state
steady;

// Do a stochastic simulation, save impulse response graphs
stoch_simul(order=1, irf=40, graph_format=eps, nodisplay, noprint);