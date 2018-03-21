% How to clean everything
clear

% Save current working directory, which should be the parent directory of
% the Dynare file this code calls - ps3q2.mod
dir_orig = pwd;

% Specify name of Dynare file
dyn = 'ps3q2';

% Specify graph format (has to match the format specified in the Dynare
% file!)
gform = '.eps';

% Specify folder containing Dynare file
dir_dyn = './dynare_ps3q2';

% Change to Dynare directory
cd(dir_dyn)

% Set up parameters
beta = .99;
rho = .95;
alpha = .35;
eta = .5;
phi = .2886;
delta_bar = .025;

% Set parameters only used in Dynare file
sig = .01;

% Calculate steady state values
% Specify whether you're forcing N = 1 in steady state
restricted = 0;

% This remains the same
delta_st = delta_bar;

% This really only makes a difference for N_st
if restricted == 1
    N_st = 1;
else
    N_st = (((1 - alpha) * (1 + beta*(delta_bar-1))) / ...
        (phi * (1 + beta * (delta_bar*(1-alpha)-1))))^(eta/(1+eta));
end

% These remain the same
K_st = ((1/alpha) * ((1/beta) + delta_st - 1))^(1/(alpha-1)) * ...
    N_st^(alpha - 1);
I_st = delta_st * K_st;
C_st = K_st^alpha * N_st^(1-alpha) - I_st;

% Save parameter values as .mat file, so Dynare can access them
% This includes the steady state values, which Dynare uses as initial
% values when solving the model
save(strcat(dyn, '_init_params.mat'))

% Display steady state values
disp(['K = ', num2str(K_st), '; I = ', num2str(I_st), ...
    '; N = ', num2str(N_st), '; C = ', num2str(C_st)])

rhos = [rho, .2];

% Loop over the values of rho
for i = 1:length(rhos)
   if i == 1
       % For the first iteration, call full Dynare (hilariously, you have
       % to hard code the name of the Dynare file here, because it's always
       % taken as a literal; what a terrible design choice (but maybe I'm
       % just not getting it))
       dynare ps3q2 noclearall;
       info = stoch_simul(var_list_);
   else
       % For all other iterations, only change the parameter value and call
       % the stochastic simulation, otherwise Dynare resets to default
       set_param_value('rho', rhos(i));
       info = stoch_simul(var_list_);
   end
   
   % Rename the graph that was created so it's not overwritten on the next
   % iteration
   movefile(strcat(dyn, '_IRF_', gform(2:end), gform), ...
       strcat(dyn, '_IRF_rho_', num2str(i), gform));
end

% Change back to the parent directory
cd(dir_orig);