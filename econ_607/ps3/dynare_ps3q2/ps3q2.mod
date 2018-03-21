// Set up variables
var C K I N delta;
varexo eps;

// Set up parameters
parameters beta rho alpha eta phi delta_bar C_st K_st I_st N_st delta_st sig;

// Load parameter values from .mat file Matlab created
load ps3q2_init_params.mat;

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
delta - log(delta_st) = rho * (delta(-1) - log(delta_st)) + eps;
K - log(K_st) = (1 - delta_st) * (K(-1) - log(K_st)) + delta_st * (I - log(I_st) - (delta - log(delta_st)));
C_st * (C - log(C_st) - alpha * (K - log(K_st))) = I_st * ((1 - alpha) * (N - log(N_st)) - (I - log(I_st)));
1 - (C - log(C_st)) = C_st * N_st^((1 + alpha*eta)/eta) * K_st^(-alpha) * (phi/(1-alpha)) *
	((N - log(N_st) + eta + alpha*eta*(N - log(N_st) - (K - log(K_st))))/eta);
1 - (C - log(C_st)) + (C(+1) - log(C_st)) = K_st^(alpha-1) * N_st^(1-alpha) * alpha * beta * ((alpha-1)*(K(+1) - log(K_st)) +
	(1-alpha)*(N(+1) - log(N_st))) + beta * (1 - delta_st * (1 + delta(+1) - log(delta_st)));
end;

// Specify initial state
initval;
K = 0;
I = 0;
N = 0;
C = 0;
delta = 0;
end;

// Specify shock process
shocks;
var eps = sig^2;
end;

// Calculate steady state
steady;

// Do a stochastic simulation, save impulse response graphs
stoch_simul(order=1, irf=20, graph_format=eps, nodisplay, noprint);