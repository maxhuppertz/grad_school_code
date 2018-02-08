% This file calculates steady state values for a simple investment model
% featuring an investment tax credit (ITC) from problem set 2, and then
% calls Dynare to solve the model and calculate impulse response functions

% How to clean everything
clear

% Save current working directory, which should be the parent directory of
% the Dynare file this code calls - ps2q3.mod
dir_orig = pwd;

% Specify name of Dynare file
dyn = 'ps2q3';

% Specify graph format (has to match the format specified in the Dynare
% file!)
gform = '.eps';

% Specify folder containing Dynare file
dir_dyn = './dynare';

% Set parameters
tau_K = 0;
theta = 1;
alpha = .5;
r = .25*.04;
delta = .25*.1;
A = 1;
ITC = 0;

% Calculate steady state values
P_st = ((delta / A)^(1 - alpha) * alpha * (1 - tau_K) / ...
    ((delta + r)*(1 - ITC)))^(1 / (1 + theta - theta*alpha));
I_st = A*P_st^theta;
K_st = I_st / delta;

% Display steady state values
disp(['K = ', num2str(K_st), '; I = ', num2str(I_st), ...
    '; P = ', num2str(P_st)])

% Change to Dynare directory
cd(dir_dyn)

% Set up a vector of values for rho (the initial one should be the default
% choice for rho implemented in the Dynare file, since you can't change
% parameters before you've set up the model, and it makes sense to use the
% first instance of the model to get some impulse responses instead of
% estimating it for no reason)
rhos = [.25, .5, .75, .9, 1];

% Loop over the values of rho
for i = 1:length(rhos)
   if i == 1
       % For the first iteration, call full Dynare (hilariously, you have
       % to hard code the name of the Dynare file here, because it's always
       % taken as a literal; what a terrible design choice (but maybe I'm
       % just not getting it))
       dynare ps2q3 noclearall;
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

% Set up a vector of deltas
deltas = [.2, .1, .05, .02, .01];

% Same thing as for rho, but for different deltas (note that this also
% changes the value of rho one last time)
for i = 1:length(deltas)
   if i == 1
       % Fix rho at desired value
       set_param_value('rho', rhos(4));
   end
   
   set_param_value('delta', .25*deltas(i));
   info = stoch_simul(var_list_);
   
   movefile(strcat(dyn, '_IRF_', gform(2:end), gform), ...
       strcat(dyn, '_IRF_del_', num2str(i), gform));
end

% Change back to the parent directory
cd(dir_orig);