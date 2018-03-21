% How to clean everything
clear

% Save current working directory, which should be the parent directory of
% the Dynare file this code calls - ps3q3.mod
dir_orig = pwd;

% Specify name of Dynare file
dyn = 'ps3q3';

% Specify graph format (has to match the format specified in the Dynare
% file!)
gform = '.eps';

% Specify folder containing Dynare file
dir_dyn = './dynare_ps3q3';

% Change to Dynare directory
cd(dir_dyn)

% Set up parameters
beta = .99;
rho = .95;
alpha = .35;
B = 8/3;
A_bar = 1;
delta = .025;

% Set parameters only used in Dynare file
sig = .01;

% Calculate steady state values
A_st = A_bar;
N_st = (1 - alpha) / (alpha * B) * ((1/beta) + delta - 1) / ...
    ((1/alpha) * ((1/beta) + delta - 1) - delta);
K_st = ((1/(alpha*A_st)) * ((1/beta) + delta - 1))^(1/(alpha - 1)) * N_st;
I_st = delta * K_st;
Y_st = A_st * K_st^alpha * N_st^(1-alpha);
C_st = Y_st - I_st;

% Save parameter values as .mat file, so Dynare can access them
% This includes the steady state values, which Dynare uses as initial
% values when solving the model
save(strcat(dyn, '_init_params.mat'))

% Display steady state values
disp(['K = ', num2str(K_st), '; I = ', num2str(I_st), ...
    '; N = ', num2str(N_st), '; C = ', num2str(C_st)])

% Run Dynare
dynare ps3q3

% Change back to the parent directory
cd(dir_orig);

% Set smoothing parameter for HP filter
mu = 1600;

% List of variable names (has to be in the same order as in the Dynare
% file, else there'll be lots of mix ups!), excluding A (which would
% otherwise be first on the list)
vnames = {'N', 'K', 'Y', 'C', 'I'};

% List of steady state values, in the same order!
stval = [N_st, K_st, Y_st, C_st, I_st];

for i = 1:length(vnames)
    % Retrieve the time series for the current variable
    y = oo_.endo_simul(i+1,:);
    
    % Apply HP filter
    [y_dt, T] = hp_filter(y, mu);
    sd = std(y_dt);
    
    % Display the simulated 'volatility' (standard deviation)
    disp(strcat('SD(', vnames(i), '):', {' '}, num2str(sd)))
end