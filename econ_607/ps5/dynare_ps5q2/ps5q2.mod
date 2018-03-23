// Set up variables
var A n k c I;
varexo eps;

// Set up parameters
parameters beta rho alpha v chi A_bar A_st n_st k_st c_st I_st delta sig;

// Load parameter values from .mat file Matlab created
load ps5q2_init_params.mat;

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
A - log(A_st) = rho * (A(-1) - log(A_st)) + eps;
k - log(k_st) = (1 - delta) * (k(-1) - log(k_st)) + delta * (I - log(I_st));
c_st * (1 + c - log(c_st)) + I_st * (1 + I - log(I_st)) = A_st * k_st^alpha * n_st^(1-alpha) *
	(1 + (A - log(A_st)) + alpha * (k - log(k_st)) + (1-alpha) * (n - log(n_st)));
n_st^(1/v + alpha) * (1 + (1/v + alpha)*(n - log(n_st))) = 
	((1-alpha) * k_st^alpha / (chi*c_st)) * 
	(1 + (A - log(A_st)) + alpha*(k - log(k_st)) - (c - log(c_st)));
1/c_st - (1/c_st)*(c - log(c_st)) = (1/c_st) * beta * (alpha * k_st^(alpha-1) * n_st^(1-alpha) *
	(1 + (alpha-1)*(k(+1) - log(k_st)) + (1-alpha)*(n(+1) - log(n_st)) + (A(+1) - log(A_st)) - (c(+1) - log(c_st)))
	+ (1 - delta) * (1 - (c(+1) - log(c_st))));
end;

// Specify initial state
initval;
k = 0;
I = 0;
n = 0;
c = 0;
A = 0;
end;

// Specify shock process
shocks;
var eps = sig^2;
end;

// calculate steady state
steady;

// Do a stochastic simulation, save impulse response graphs
stoch_simul(order=1, irf=150, graph_format=eps, nodisplay, noprint);

// Simulate for 1,000 periods
stoch_simul(order=1, periods=1000, nodisplay, noprint);