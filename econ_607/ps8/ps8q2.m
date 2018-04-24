% How to clean everything
clear

% Save current working directory, which should be the parent directory of
% the Dynare file this code calls - ps8q2.mod
dir_orig = pwd;

% Specify name of Dynare file
dyn = 'ps8q2';

% Specify graph format (has to match the format specified in the Dynare
% file!)
gform = '.eps';

% Specify folder containing Dynare file
dir_dyn = './dynare_ps8q2';

% Change to Dynare directory
cd(dir_dyn)

% Set up base case parameters
beta = .99;
kappa = .023;
theta = 1 / 0.16;
phi_pi = 1.5;
phi_y = 0.5;
rho = .9;
sig = .65;

% Save parameter values as .mat file, so Dynare can access them
save(strcat(dyn, '_init_params.mat'), ...
    'beta', 'kappa', 'theta', 'phi_pi', 'phi_y', 'rho', 'sig')

% Run Dynare
dynare ps8q2

% Make a vector of base parameters
params_base = [beta, kappa, theta, phi_pi, phi_y, rho, sig];

% Make a vector of changed parameters in the same order (you can' skip any,
% but you can leave some off at the end); this is super hacky but it does
% what it's supposed to do, so hey
params_change = [beta*.9, theta*.9, kappa*1.2, phi_pi*1.2, phi_y*1.2];

% Go through parameter changes
for k=1:length(params_change)
    % Create vector of current parameter values
    params = params_base;
    params(k) = params_change(k);
    
    % Set parameter values
    beta = params(1);
    kappa = params(2);
    theta = params(3);
    phi_pi = params(4);
    phi_y = params(5);
    rho = params(6);
    sig = params(7);
    
    % Save parameter values as .mat file, so Dynare can access them
    save(strcat(dyn, '_init_params.mat'), ...
        'beta', 'kappa', 'theta', 'phi_pi', 'phi_y', 'rho', 'sig')

    % Run Dynare
    dynare ps8q2 noclearall;
    
    % Rename impulse response graph
    movefile(strcat(dyn, '_IRF_eta', gform), ...
        strcat(dyn, '_IRF_', num2str(k), gform));
end

% Change back to parent directory
cd(dir_orig)