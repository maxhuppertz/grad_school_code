// Set up variables
var ITC K I P;
varexo eps;

// Set up parameters
parameters rho tau_K theta alpha r delta A ITC_bar sig;
rho = .25;
tau_K = 0;
theta = 1;
alpha = .5;
r = .25*.04;
delta = .25*.1;
A = 1;
ITC_bar = 0;
sig = 1;

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
K = log(69);
I = log(1.7);
P = log(1.7);
ITC = 0;
end;

// Specify shock process
shocks;
var eps = sig^2;
end;

// Calculate steady state
steady;

// Do a stochastic simulation, save impulse response graphs
stoch_simul(order=1, irf=200, graph_format=eps, nodisplay, noprint);