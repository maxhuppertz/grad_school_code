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
fname_data = 'insurance_data.csv';

% Load data set, as a table
insurance_data = readtable(fname_data);

% Specify name of plan ID variable
v_pid = 'plan_id';

% Specify name of chosen plan indicator
v_chosen = 'plan_choice';

% Make an indicator for a plan being chosen
cidx = insurance_data{:, {v_chosen}} == insurance_data{:, {v_pid}};

% Get only values for chosen plans, as a new table
insurance_data_red = insurance_data(cidx, :);

% Get shifted version of chosen plan ID variable
shifted_choice = circshift(insurance_data_red{:, {v_chosen}}, 1);

% Specify name for that variable in the data set
v_shifted_choice = strcat(v_pid, '_shifted');

% Add it to the data set
insurance_data_red = addvars(insurance_data_red, shifted_choice, ...
    'NewVariableNames', v_shifted_choice);

% Specify name of individual ID variable
v_id = 'indiv_id';

% Get shifted version of individual ID variable
shifted_id = circshift(insurance_data_red{:, {v_id}}, 1);

% Specify name for it in the data set
v_shifted_id = strcat(v_id, '_shifted');

% Add it to the data set
insurance_data_red = addvars(insurance_data_red, shifted_id, ...
    'NewVariableNames', v_shifted_id);

% Specify name of choice situation ID
v_csid = 'choice_sit';

% Add shifted plan and individual IDs to main data set
insurance_data = join(insurance_data, ...
    insurance_data_red(:, {v_id, v_csid, v_shifted_id, v_shifted_choice}));

% Specify name of plan premium , coverage and service quality variables
v_pre = 'premium';  % Premium
v_cov = 'plan_coverage';  % Coverage
v_svq = 'plan_service_quality';  % Service quality

% Specify whether to include an outside option in the data set
include_outopt = 0;

% Check whether an outside option is needed
if include_outopt == 1
    % Get a copy of the data to make the outside option
    p0_data = insurance_data_red;

    % Set its plan ID, premium, coverage, and service quality to zero
    p0_data{:, {v_pid v_pre v_cov v_svq}} = 0;

    % Add it to the data set
    insurance_data = [insurance_data; p0_data];
end

% Update indicator for a plan being chosen
cidx = insurance_data{:, {v_chosen}} == insurance_data{:, {v_pid}};

% Specify name of switching indicator, that is, a variable which is one for
% any plan that is not the same as the one chosen during the previous
% period, and also for any plan that is the first one anyone chooses
v_switch = 'switched_plans';

% Calculate the indicator
switched = ...
    (insurance_data{:, {v_shifted_id}} ~= insurance_data{:, {v_id}}) | ...
    (insurance_data{:, {v_shifted_choice}} ~= insurance_data{:, {v_pid}});

% Add it to the data set
insurance_data = addvars(insurance_data, switched, ...
    'NewVariableNames', v_switch);

% Specify name for (analogous) plan retention variable
v_ret = 'retained_plan';

% Add it to the data set
insurance_data = addvars(insurance_data, ~switched, ...
    'NewVariableNames', v_ret);

% Specify name of comparison tool access indicator
v_tool = 'has_comparison_tool';

% Specify name for tool access and plan switching interaction
v_switch_tool = strcat(v_switch, '_times_', v_tool);

% Add the interaction to the data set
insurance_data = addvars(insurance_data, ...
    insurance_data{:, {v_switch}} .* insurance_data{:, {v_tool}}, ...
    'NewVariableNames', v_switch_tool);

% Specify name for tool access and plan retention variable
v_ret_tool = strcat(v_ret, '_times', v_tool);

% Add the interaction to the data set
insurance_data = addvars(insurance_data, ...
    insurance_data{:, {v_ret}} .* insurance_data{:, {v_tool}}, ...
    'NewVariableNames', v_ret_tool);

% Specify name of income variable
v_inc = 'income';

% Specify name of log income variable
v_loginc = 'log_income';

% Add log income to the data set
insurance_data = addvars(insurance_data, ...
    log(insurance_data{:, {v_inc}}), 'NewVariableNames', v_loginc);

% Specify name for log premium variable
v_logpre = 'log_premium';

% Add variable to the data set
insurance_data{:, {v_logpre}} = log(insurance_data{:, {v_pre}});

% Specify various other variables
v_age = 'age';
v_risk = 'risk_score';
v_year = 'year';
v_sex = 'sex';
v_tenure = 'years_enrolled';

% Specify which variables to interact with the plan premium
vars_pre_dem = {v_inc v_risk};

% Specify name for matrix of demographics interacted with premium
varn_pre_dem = 'premium_times_demographics';

% Add variables to the data
insurance_data = addvars(insurance_data, ...
    insurance_data{:, vars_pre_dem} ...
    .* (insurance_data{:, {v_pre}} * ones(1,length(vars_pre_dem))), ...
    'NewVariableNames', varn_pre_dem);

% Specify variables to interact with coverage
vars_cov_dem = {v_inc v_risk};

% Specify name for matrix of demographics interacted with coverage
varn_cov_dem = 'coverage_times_demographics';

% Add variables to the data set
insurance_data = addvars(insurance_data, ...
    insurance_data{:, vars_cov_dem} ...
    .* (insurance_data{:, {v_cov}} * ones(1,length(vars_cov_dem))), ...
    'NewVariableNames', varn_cov_dem);

% Specify variables to interact with service quality
vars_svq_dem = {v_inc v_risk};

% Specify name for matrix of demographics interacted with coverage
varn_svq_dem = 'quality_times_demographics';

% Add variables to the data set
insurance_data = addvars(insurance_data, ...
    insurance_data{:, vars_svq_dem} ...
    .* (insurance_data{:, {v_svq}} * ones(1,length(vars_svq_dem))), ...
    'NewVariableNames', varn_svq_dem);

% Specify variables to interact with plan retainment
vars_ret_dem = {v_tenure};

% Specify name for the matrix of variables
varn_ret_dem = 'retainment_times_demographics';

% Add plan to the data
insurance_data = addvars(insurance_data, ...
    insurance_data{:, vars_ret_dem} ...
    .* (insurance_data{:, {v_ret}} * ones(1,length(vars_ret_dem))), ...
    'NewVariableNames', varn_ret_dem);

% Get unique values of plan ID variable
plans = unique(insurance_data{:, {v_pid}});

% Specify name for set of plan dummies
varn_pland = 'plan_dummies';

% Set up plan dummies
vars_pland = zeros(size(insurance_data,1), length(plans)-1);

% Set up plan ID counter
k=1;

% Go through all plan ID values except the first
for plan=min(plans)+1:max(plans)
    % Replace indicator for the current plan
    vars_pland(:,k) = insurance_data{:,{v_pid}} == plan;
    
    % Increase counter
    k = k + 1;
end

% Add dummies to the data
insurance_data = addvars(insurance_data, ...
    vars_pland, 'NewVariableNames', varn_pland);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Part 2: Structural estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Change back to main directory
cd(mdir)

% Set precision for sparse grids integration
sgprec = 10;

% Get sparse grids quadrature points
[qp, qw] = nwspgr('KPN', 3, sgprec);

% Make data set
X = insurance_data{:, {v_pre, v_cov, v_svq, ...  % Plan characteristics
    v_ret, v_ret_tool, ...  % Switching cost and tool access
    varn_pre_dem, varn_cov_dem, varn_svq_dem}};  % Demographics

% Get choice situation ID as vector
sit_id = insurance_data{:, {v_csid}};

% Set initial values
%
% Mean of random coefficients, mu_beta
mu_beta0 = [-0.03, 0.22, 0.02];

% Lower Cholesky factor of random coefficient covariance matrix, C_Sigma
Csigma0 = [-4.90, 1.23, -12.26, ...  % Diagonal elements
    0.03, 0.02, 1.14];  % Off-diagonal elements

% Coefficient on plan retainment and plan retainment times tool, alpha
alpha0 = [3.78, -0.62];

% Coefficient on demographics interacted with premium, gamma
gamma0 = [-0.00, -0.00, ...
    0.01, 0.07, ...
    0.00, 0.00];

% Choose whether to use only a subset of the data
use_subset = 0;

% Check whether subsetting is necessary
if use_subset == 1
    % Specify how many individuals to use in the subset sample
    n_subset = 100;
    
    % Get the subset of people, by using only the first n_subset ones
    subset = insurance_data{:, {v_id}} <= n_subset;
    
    % Subset the data set
    X = X(subset, :);
    
    % Subset the choice index
    cidx = cidx(subset, :);
    
    % Subset the choice situation ID
    sit_id = sit_id(subset, :);
end

% Calculate some counters, which will be helpful for telling fminunc which
% parts of the parameter vector it uses (which is just a single column
% vector) correspond to which parts of the input vectors for the log
% likelihood function (which takes several arguments)
bmax = length(mu_beta0);  % End of beta
Csdiagmin = length(mu_beta0)+1;  % Start of diagonal elements of C_Sigma
Csdiagmax = length(mu_beta0)*2;  % End of diagonal elements of C_Sigma
Csodiagmin = Csdiagmax+1;  % Start of off-diagonal elements of C_Sigma
Csodiagmax = length(mu_beta0)+length(Csigma0);  % End of off-diag. C_Sigma
amin = Csodiagmax+1;  % Start of alpha
amax = amin+length(alpha0)-1;  % End of alpha
gmin = amax+1;  % Start of gamma
gmax = gmin+length(gamma0)-1;  % End of gamma

% Set function tolerance for solver
ftol = 1e-14;

% Set optimality tolerance for solver
otol = 1e-6;

% Set step tolerance level for solver
stol = 1e-14;

% Set multiplier on number of variables used to get maximum number of
% iterations
vmul = 500;  % Default is 100

% Get current parallel pool, don't create one if there is none
checkpool = gcp('nocreate');

% Check whether no parallel pool is open
if isempty(checkpool)
    % If so, start a parallel pool on the local profile
    parpool('local');
end

% Set optimization options
options = optimoptions('fminunc', ...  % Which solver to apply these to
    'Algorithm', 'quasi-newton', ...  % Which solution algorithm to use
    'HessUpdate', 'bfgs', ...
    'UseParallel', true, ...
    'OptimalityTolerance', otol, 'FunctionTolerance', ftol, ...
    'StepTolerance', stol, ...  % Tolerances
    'MaxFunctionEvaluations', vmul*gmax, ...  % Max. function evaluations
    'Display', 'iter', ...  % Display options
    'FiniteDifferenceType', 'central');

% It helps to scale all variables such that they are of about equal order
% of magnitude
%
% Set master scaling factor
msc = 1;

% Get number of dimensions of mu_beta
d = length(mu_beta0);

% Make scaling vectors for elements of parameter vector
a1 = ones(1,d);  % mu_beta
a21 = ones(1,d);  % Diagonal C_Sigma
a22 = ones(1,d);  % Off-diag. C_Sigma
a3 = ones(1,2) * msc;  % alpha
a4 = ones(1,6) * msc;  % Interactions with premium, risk score, and quality

% Scale mu_beta scaling vector by master scale factor
a1 = a1 * msc;

% Put all adjustment factors into a vector for later use
theta_adj = [a1, a21, a22, a3, a4];

% Make vector of lower bounds on parameters, set those to negative infinity
lower = zeros(1, gmax) - Inf;

% Replace lower bounds on diagonal elements of the random coefficient
% covariance matrix as slightly above zero, to enfore strict inequality
lower(Csdiagmin:Csdiagmax) = 1e-6;

% Set upper bounds to positive infinity
upper = zeros(1, gmax) + Inf;

% Perform MLE, stop the time it takes to run. I use constrained
% optimization because the diagonal elements of the random coefficient
% covariance matrix have to be positive.
tic
[theta_hat,ll,~,~,G,I] = fminunc( ...
    @(theta)ll_structural(theta(1:bmax).*a1, ...  % mu_beta
    [theta(Csdiagmin:Csdiagmax).*a21, ...  % C_sigma diagonal
    theta(Csodiagmin:Csodiagmax).*a22], ...  % C_Sigma off-diagonal
    theta(amin:amax).*a3, ...  % alpha
    theta(gmin:gmax).*a4, ...  % gamma
    X, sit_id, cidx, qp, qw, 0), ...  % Non-parameter inputs
    [mu_beta0, Csigma0, alpha0, gamma0], ...  % Initial values
    options);  % Other optimization options
time = toc;

% Display log likelihood
disp(strcat('Log-likelihood: ', num2str(-ll)))
fprintf('\n')

% Get inverse of Fisher information; don't display potential warnings about
% this not being invertible
warning('off', 'MATLAB:singularMatrix')
warning('off', 'MATLAB:nearlySingularMatrix')
V1 = inv(I);

% Get analytic standard errors, based on properties of correctly specified
% MLE (variance is the negative inverse of Fisher information, estimate
% this using sample analogue)
SE_a = sqrt(diag(V1));

% Get alternative estimator, based on the score outer product. Since this
% is the last dubious inversion, turn warnings back on afterwards.
V2 = inv(G*G.');
warning('on', 'MATLAB:singularMatrix')
warning('on', 'MATLAB:nearlySingularMatrix')

% Get analytic standard errors, again using properties of correctly
% specified MLE
SE_sop = sqrt(diag(V2));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Part 3: Display the results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set up a cell array to display the results
D = cell(length(theta_hat)+1,4);

% Add headers
D(1,:) = {'Parameter', 'Estimate', 'SE_ih', 'SE_sop'};

% Fill in the first column of the cell array with coefficient labels
% Set up a counter to mark which row needs to be filled in next
k = 2;

% Add labels for the parts of mu_beta
for i=1:length(mu_beta0)
    D(k,1) = {strcat('beta_', num2str(i))};
    k = k + 1;
end

% Add labels for the diagonal elements of C_Sigma
for i=1:length(mu_beta0)
    D(k,1) = {strcat('C_Sigma_', num2str(i), num2str(i))};
    k = k + 1;
end

% Add labels for the off-diagonal elements of C_Sigma
for i=1:length(mu_beta0)-1
    for j=i+1:length(mu_beta0)
        D(k,1) = {strcat('C_Sigma_',num2str(j),num2str(i))};
        k = k + 1;
    end
end

% Add labels for the elements of alpha
for i=1:length(alpha0)
    D(k,1) = {strcat('alpha_', num2str(i))};
    k = k + 1;
end

% Add labels for elements of gamma
for i=1:length(gamma0)
    D(k,1) = {strcat('gamma_', num2str(i))};
    k = k + 1;
end

% Set number of digits to display results
rdig = 4;

%theta_hat = theta_hat .* theta_adj;

% Add theta_hat and its standard error to the results
D(2:end,2:end) = num2cell(round([theta_hat.', SE_a, SE_sop], rdig));

% Set display format
format long g

% Display how much time the estimation took
disp(['Time elapsed: ', num2str(time), ' seconds'])
fprintf('\n')

% Display the results
disp('Parameter estimates:')
disp(D)

% Get part of the estimates corresponding to covariance Cholesky factor
sigma_hat = [theta_hat(Csdiagmin:Csdiagmax), ...
    theta_hat(Csodiagmin:Csodiagmax)];

% Get diagonal elements of Cholesky factor
C_hat = diag(exp(sigma_hat(1:d)));

% Fill in (lower triangular) off-diagonal elements
%
% Set up a counter for the elements in the sigma vector
k = 1;

% Go through all but the last rows of the covariance matrix
for i = 1:d-1
    % Go through all elements of that row past the diagonal
    for j = i+1:d
        % Replace the j,i element
        C_hat(j,i) = sigma_hat(d + k);
        
        % Increase the element counter
        k = k + 1;
    end
end

% Get covariance matrix from Cholesky factor
Sigma_hat = C_hat*C_hat.';

% Display estimated covariance matrix
disp('Covariance estimate:')
disp(round(Sigma_hat, rdig))