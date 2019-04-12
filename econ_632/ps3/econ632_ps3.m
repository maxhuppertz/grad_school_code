% Clear everything
clear

% Set random number generator's seed
rng(632)

% Get main directory, as current file's directory
fname_script = mfilename;  % Name of this script
mdir = mfilename('fullpath');  % Its directory plus name
mdir = mdir(1:end-length(fname_script));  % Its directory without the name

% Set data directory
ddir = 'data/';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Part 1: Load data, generate variables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Change directory to data
cd(strcat(mdir,ddir))

% Specify name of data file
fname_data = 'firm_entry.csv';

% Load data set, as a table
entry_data = readtable(fname_data);

% Specify action ID variable
v_act = 'i';

% Get unique values of action variable
actions = unique(entry_data{:, {v_act}});

% Count the total number of actions
J = length(actions);

% Actions are labeled 0 and 1, which cannot be used to index the columns of
% a matrix. (It will later become clear why that would be useful.) Specify
% a name for an index version of the action variable.
v_cidx = strcat(v_act, '_idx');

% Add the index version to the data
entry_data = addvars(entry_data, entry_data{:, {v_act}} + 1, ...
    'NewVariableNames', v_cidx);

% Get a shifted (forward, by one period) version of the action
shifted_act = circshift(entry_data{:, {v_act}}, 1);

% Specify a name for this as a variable
v_shifted_act = strcat('shifted_', v_act);

% Add it to the data set
entry_data = addvars(entry_data, shifted_act, ...
    'NewVariableNames', v_shifted_act);

% Specify name of market ID variable
v_mkt = 'Market';

% Get a shifted (forward, by one period) version of this variable
shifted_mkt = circshift(entry_data{:, {v_mkt}}, 1);

% Specify a name for the variable
v_shifted_mkt = strcat('shifted_', v_mkt);

% Add it to the data set
entry_data = addvars(entry_data, shifted_mkt, ...
    'NewVariableNames', v_shifted_mkt);

% Tag the first period in each market
fper = (entry_data{:, {v_mkt}} ~= entry_data{:, {v_shifted_mkt}});

% Get an indicator for whether a firm is currently in a market
is_in = (entry_data{:, {v_shifted_act}} == 1) & ~fper;

% Specify a name for the variable
v_is_in = 'in_market';

% Add variable to the data set
entry_data = addvars(entry_data, is_in, 'NewVariableNames', v_is_in);

% Specify name of market state variable (x)
v_stt_x = 'x';

% Get unique values of market state variable
mktstates = unique(entry_data{:, {v_stt_x}});

% Count number of market states
K = length(mktstates);

% Get combined market state - action variable. This will be equal to the
% market state variable - i.e. 1, 2, 3, 4, 5 - if i = 0, and equal to the
% market state variable plus 5, i.e. 6, 7, 8, 9, 10, otherwise, that is, if
% i = 1. This is needed to calculate transition probabilities, which may
% depend on both the current market state, and the observed action.
stt_act = entry_data{:, {v_act}} * K + entry_data{:, {v_stt_x}};

% Specify a name for the combined market state action variable
v_stt_act = 'action_x_combination';

% Add the variable to the data set
entry_data = addvars(entry_data, stt_act, ...
    'NewVariableNames', v_stt_act);

% Get complete state variable. A state is a combination of a market state
% and a past action, which is the same as the being in the market
% indicator. This will equal the market state - 1, 2, 3, 4, 5 - if i_{-1} =
% 0, and equal the market state plus 5 - 6, 7, 8, 9, 10 - otherwise. Note
% that this is not the same as the combined market state action variable,
% because that depends on the current market state and contemporary action,
% whereas this depends on the current market state and last period's
% action.
stt = entry_data{:, {v_stt_x}} + entry_data{:, {v_is_in}} * K;

% Specify a name for the variable
v_stt = 'state';

% Add it to the data set
entry_data = addvars(entry_data, stt, 'NewVariableNames', v_stt);

% Get version of state variable shifted back by one period
backshifted_x = circshift(entry_data{:, {v_stt_x}}, -1);

% Specify name for shifted state variable
v_backshifted_x = strcat('backshifted_', v_stt_x);

% Add the shifted state to the data set
entry_data = addvars(entry_data, backshifted_x, ...
    'NewVariableNames', v_backshifted_x);

% Get version of market ID shifted back by one period
backshifted_id = circshift(entry_data{:, {v_mkt}}, -1);

% Mark observations which should be used when estimating transition
% probabilities. This excludes the last observation for a given market,
% since the following state is unobserved.
use_tprob = (entry_data{:, {v_mkt}} == backshifted_id);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Part 2: Estimate transition probabilities
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get current and next period's market states as a matrix
mkt_trans = entry_data{:, {v_stt_x, v_backshifted_x}};

% Count transitions between market states
p = accumarray(mkt_trans(use_tprob,:), ones(sum(use_tprob),1));

% Get transition probabilities, by dividing by the row sum
p = p ./ (sum(p,2) * ones(1,K));

% Set up a cell array for transition probabilities between complete states
% (i.e. market state - action combinations) conditional on the chosen
% action
P = cell(J,1);

% Make a J by J identity matrix
l = eye(J);

% Go through all actions
for i=1:J
    % Add the conditional transition probabilities to the array. The way
    % this works is that, when i = 0, for any state 6, 7, 8, 9, or 10, the
    % probability of transitioning to it is zero. So I can just stack the P
    % matrix twice, and add an array of zeros of equal size next to it, to
    % get the conditional transition probabilities. This Kronecker product
    % does exactly that.
    %
    % Get stacked probabilities conditional on i
    pstack = [p;p];
    
    % Use Kronecker product to add zeros where needed
    P{i,1} = kron(l(i,:),pstack);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Part 3: Structural estimation (MLE)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Change back to main directory
cd(mdir)

% Set discount factor
beta = .95;

% Make vector to base flow utilities off of, for the case in which the firm
% enters a market. Has to be such that S * theta = U.
S = [ones(K*J,1), [(1:K), (1:K)].', [ones(1,K), zeros(1,K)].'];

% Set up initial guess for value function iteration
V0 = zeros(K*J,J);

% Set tolerance for value function iteration
tolEV = 10^(-14);

% Get vector of observed choices
C = entry_data{:, {v_act}};

% Get vector of observed states, i.e. market state - past action
% combinations
chi = entry_data{:, {v_stt}};

% Set up initial guess for the elements of theta
theta0 = randn(3,1)*100;

% Set function tolerance for solver
ftol = 10^(-14);

% Set optimality tolerance for solver
otol = 10^(-14);

% Set step tolerance level for solver
stol = 10^(-14);

% Set variable multiplier (for MLE)
vmul = 500;

% Set optimization options for fminunc
options = optimoptions('fminunc', ...  % Which solver to apply these to
    'Algorithm', 'quasi-newton', ...  % Which solution algorithm to use
    'HessUpdate', 'bfgs', ...  % Hessian method
    'UseParallel', true, ...  % Parallel computing
    'OptimalityTolerance', otol, 'FunctionTolerance', ftol, ...
    'StepTolerance', stol, ...  % Tolerances
    'MaxFunctionEvaluations', vmul*3, ... % Max. function evaluations
    'FiniteDifferenceType', 'central', ...  % Finite difference method
    'Display', 'off', 'SpecifyObjectiveGradient', false);

% Get current parallel pool, don't create one if there is none
checkpool = gcp('nocreate');

% Check whether no parallel pool is open
if isempty(checkpool)
    % If so, start a parallel pool on the local profile
    parpool('local');
end

% Run MLE, record the time it takes to run
tic;
[theta_hat,~,~,~,G,I] = fminunc( ...
    @(theta)ll(C, S, chi, V0, theta, P, beta, tolEV), ...
    theta0, options);
time = toc;

% Get covariance matrix as inverse of the Fisher information
V_hat = inv(I);

% Get analytical standard errors
SE_a = sqrt(diag(V_hat));

% Set display format
format long g

% Display the MLE results
fprintf('\n')
disp('MLE results')
fprintf('\n')
disp(round([theta_hat, SE_a],4))
disp(['MLE time:', ' ', num2str(time), ' seconds'])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Part 4: Robustness check 1 - bootstrap
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set number of bootstrap iterations
B = 5000;

% Set up vector of bootstrap estimates
T = zeros(B,length(theta0));

% Get market ID values
markets = unique(entry_data{:, {v_mkt}});

% Get maximum market ID value
M = max(markets);

% Specify number of periods each market is observed
taumax = 100;

% Go through all bootstrap iterations, record the time this takes
tic;
parfor b=1:B
    % Set random number generator's seed, for reproducibility
    rng(b)
    
    % Draw bootstrap sample, by getting indices of markets
    mi = randi([1,M],M,1);
    
    % Get indices of periods within market, which always range from 1 to
    % taumax; stack them, one for each market
    ii = kron(ones(M,1),(1:taumax).');
    
    % Stack market indices, taumax times for each market, and convert them
    % to the leading digit of the corresponding row ID
    mi = kron((mi - 1) * taumax,ones(taumax,1));
    
    % Add the two to get row indices for bootstrap sample
    I = mi + ii;
    
    % Get market transitions for the bootstrap sample
    mkt_trans_b = mkt_trans(I,:);
    
    % Count transitions between market states; luckily, the first period is
    % still located in the same place for each bootstrap market
    pb = accumarray( ...
        mkt_trans_b(use_tprob,:), ...
        ones(sum(use_tprob),1));

    % Get transition probabilities, by dividing by the row sum
    pb = pb ./ (sum(pb,2) * ones(1,K));

    % Set up a cell array for transition probabilities between states
    Pb = cell(J,1);

    % Make a J by J identity matrix might save some broadcast overhead)
    l = eye(J);
    
    % Go through all actions
    for i=1:J
        % Add the conditional transition probabilities to the array. Get
        % stacked probabilities conditional on i.
        pbstack = [pb;pb];

        % Use Kronecker product to add zeros where needed
        Pb{i,1} = kron(l(i,:),pbstack);
    end
    
    % Run MLE on the bootstrap sample
    T(b,:) = fminunc( ...
        @(theta)ll(C(I,:), S, chi(I,:), V0, theta, Pb, beta, tolEV), ...
        theta0, options);
end
% Stop the time
time = toc;

% Get the bootstrapped standard errors
SE_b = sqrt(sum((T - ones(B,1) * theta_hat.').^2,1) / B);

% Display the bootstrap results
fprintf('\n')
disp('Bootstrap results')
fprintf('\n')
disp(round(SE_b.',4))
disp(['Bootstrap time:', ' ', num2str(time), ' seconds'])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Part 5: Robustness check 2 - particle swarm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Create an initial population for the particle swarm solver
%
% Set number of initial points
Nps = 100;

% Set mean value for initial points
mups = zeros(1,length(theta0));

% Set standard deviation for initial points
sdps = 10;

% Create initial population as N(mups,sdps^2) random variables
initpop = 5 * randn(Nps,length(theta0)) + ones(Nps,1) * mups;

% Create optimization options for particle swarm
options_ps = optimoptions('particleswarm', ...  % Which solver
    'InitialSwarmMatrix', initpop, ...  % Starting population
    'FunctionTolerance', ftol, ...  % Tolerance
    'UseParallel', true, ...  % Parallel computing
    'SwarmSize', Nps, ...  % Swarm size, has to be at least Nps
    'Display', 'off');  % Display options

% Run particle swarm, record the time it takes to run
tic;
theta_hat_ps = particleswarm( ...
    @(theta)ll(C, S, chi, V0, theta, P, beta, tolEV), ...
    3, [], [], options_ps);
time = toc;

% Display the particle swarm results
fprintf('\n')
disp('Particle swarm results')
fprintf('\n')
disp(round(theta_hat_ps.',4))
disp(['Particle swarm time: ', num2str(time), ' seconds'])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Part 6: Counterfactuals
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Specify number of entry costs delta to plot
nplot = 1000;

% Set up vector of resulting entry probabilities
rho = zeros(nplot,1);

% Set up minimum and maximum entry cost
delta_min = -20;
delta_max = 20;

% Make a vector of entry costs
delta = linspace(delta_min, delta_max, nplot);

% Save the first two elements of the estimated coefficient vector as their
% own vector, to save some overhead
t_hat12 = theta_hat(1:2);

init_i = 1;
tolP = 10^(-6);

% Go through all points to plot
for i = 1:nplot
    % Get the counterfactual coefficient vector
    theta_tilde = [t_hat12;delta(i)];
    
    % Calculate the entry probability for the current entry cost
    rho(i) = ...
        probin(S, V0, P, p, theta_tilde, beta, tolEV);
end

% Set up a figure
fig1 = figure;

% Plot entry probability against entry cost
plot(delta,rho);

% Set up reverse x axis and use Latex to print axis labels
set(gca,'XDir', 'reverse', 'TickLabelInterpreter', 'latex');

% Set background color and figure size
set(gcf, 'color', 'w', ...  % Background color
    'Units', 'Inches', 'Position', [0, 0, 6.5, 6.5*(9/16)], ...
    'PaperUnits', 'Inches', 'PaperSize', [6.5, 6.5*(9/16)])  % Figure size

% Set y axis limits
ylim([min(rho) - .05, max(rho) + .05]);

% Set x axis label
xlabel('Counterfactual entry cost $\smash{-\tilde{\delta}}$', ...
    'interpreter', 'latex', 'fontsize', 11);

% Set y axis label
ylabel('Fraction of periods active', 'interpreter', 'latex', ...
    'fontsize', 11);

% Set figures directory
fdir = 'figures';

% Check whether it exists
if ~exist(strcat('./',fdir), 'dir')
    % If not, make it
    mkdir(fdir);
end

% Set name of figure to save it
fname = 'entry_prob.pdf';

% Set print options, to make sure Matlab does not cut off the figure
%set(groot,'defaultFigurePaperPositionMode','manual')

% Save the figure
saveas(fig1,strcat(fdir,'/',fname));

% Close the figure
close;