// Set up variables
var A N K Y C I;
varexo eps;

// Set up parameters
parameters beta rho alpha B A_bar A_st N_st K_st Y_st C_st I_st delta sig;

// Load parameter values from .mat file Matlab created
load ps3q3_init_params.mat;

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
K - log(K_st) = (1 - delta) * (K(-1) - log(K_st)) + (I_st / K_st) * (I - log(I_st));
C_st * (1 + C - log(C_st)) + I_st * (1 + I - log(I_st)) = A_st * K_st^alpha * N_st^(1-alpha) *
	(1 + (A - log(A_st)) + alpha * (K - log(K_st)) + (1-alpha) * (N - log(N_st)));
1 + C - log(C_st) = ((1-alpha)/(B*C_st)) * A_st * N_st^(-alpha) * K_st^alpha * (1 + (A - log(A_st))
	+ alpha * ((K - log(K_st)) - (N - log(N_st))));
1 - (C - log(C_st)) + (C(+1) - log(C_st)) = beta * (A_st * K_st^(alpha-1) * N_st^(1-alpha) *
	(1 + (A(+1) - log(A_st)) + (alpha-1)*(K(+1) - log(K_st)) + (1-alpha)*(N(+1) - log(N_st))) + 1 - delta);
Y_st * (1 + Y  - log(Y_st)) = A_st * K_st^alpha * N_st^(1-alpha) *
	(1 + (A - log(A_st)) + alpha * (K - log(K_st)) + (1-alpha) * (N - log(N_st)));
end;

// Specify initial state
initval;
K = 0;
I = 0;
N = 0;
C = 0;
Y = 0;
A = 0;
end;

// Specify shock process
shocks;
var eps = sig^2;
end;

// Calculate steady state
steady;

// Retrieve theoretical moments
stoch_simul(order=1, periods=0, nodisplay, nograph);

// Simulate for 1,000 periods
stoch_simul(order=1, periods=1000, nodisplay, noprint);