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

% Get a copy of the data to use for an outside option
p0_data = insurance_data_red;

% Set its plan ID, premium, coverage, and service quality to zero
p0_data{:, {v_pid v_pre v_cov v_svq}} = 0;

% Add it to the data set
insurance_data = [insurance_data; p0_data];

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

% Specify name of age variable
v_age = 'age';

% Specify name of risk score variable
v_risk = 'risk_score';

% Specify name for matrix of demographics interacted with premium
vars_pre_dem = 'premium_times_demographics';

% Add variables to the data
insurance_data = addvars(insurance_data, ...
    insurance_data{:, {v_age v_inc v_risk}} ...
    .* (insurance_data{:, {v_pre}} * ones(1,3)), ...
    'NewVariableNames', vars_pre_dem);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Part 2: Structural estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Change back to main directory
cd(mdir)

% Set precision for sparse grids integration
sgprec = 4;

% Get sparse grids quadrature points
[qp, qw] = nwspgr('KPN', 3, 4);

% Specify number of quadrature points for Monte Carlo integration
np = 500;

% Make data set
X = insurance_data{:, {v_pre, v_cov, v_svq, ...  % Plan characteristics
    v_ret, v_ret_tool, ...  % Switching cost and tool access
    vars_pre_dem}};  % Demographics interacted with plan premium

% Get choice situation ID as vector
sit_id = insurance_data{:, {v_csid}};

% Set optimization options
options = optimset('GradObj','off','HessFcn','off', ... 'Display','off', ...
    'TolFun',1e-6,'TolX',1e-6);

% Set initial values
mu_beta0 = [-.1, .2, .05];
sigma0 = [.2, .4, .02, -.05];
alpha0 = [2, -.5];
gamma0 = [.2, .2, .2];

% Divide some parts of X by 1000
X(:,[1, end-length(gamma0):end]) = X(:,[1, end-length(gamma0):end]) / 1000;

% Calculate some counters, which will be helpful for telling fminunc which
% parts of the parameter vector it uses (which is just a single column
% vector) correspond to which parts of the input vectors for the log
% likelihood function (which takes several arguments)
bmax = length(mu_beta0);  % End of beta
sdiagmin = length(mu_beta0)+1;  % Start of diagonal elements of Sigma
sdiagmax = length(mu_beta0)*2;  % End of diagonal elements of Sigma
amin = length(mu_beta0)+length(sigma0)+1;  % Start of alpha
amax = amin+length(alpha0)-1;  % End of alpha
gmin = amax+1;  % Start of gamma
gmax = gmin+length(gamma0)-1;  % End of gamma

% optional subsetting
%subset = insurance_data{:, {v_id}} <= 300;
%X = X(subset, :);
%cidx = cidx(subset, :);
%sit_id = sit_id(subset, :);

% Make vector of lower bounds on parameter, set those to negative infinity
lower = zeros(1, gmax) - Inf;

% Replace lower bounds on diagonal elements of the random coefficient
% covariance matrix as zero
lower(sdiagmin:sdiagmax) = 0;
lower(sdiagmax+1) = -1;

% Upper bounds are all positive infinity
upper = zeros(1, gmax) + Inf;
upper(sdiagmax+1) = 1;

% Perform MLE, stop the time it takes to run. I use constrained
% optimization because the diagonal elements of the random coefficient
% covariance matrix have to be positive.
tic
[theta_hat,ll,~,~,~,~,I] = fmincon( ...
    @(theta)ll_structural(theta(1:bmax), ...  % mu_beta
    [theta(sdiagmin:sdiagmax), ...  % Diagonal elements of Sigma
    0, 0, theta(sdiagmax+1) ...
    * sqrt(theta(sdiagmin+1) ...
    * theta(sdiagmin+2))], ...  % Off-diagonal elements of Sigma
    theta(amin:amax), ...  % alpha
    theta(gmin:gmax), ...  % gamma
    X, sit_id, cidx, qp, qw), ...
    [mu_beta0, sigma0, alpha0, gamma0], ...  % Initial values
    [], [], [], [], ...  % Linear constraints, of which there are none
    lower, upper, ...  % Lower and upper bounds on parameters
    [], ...  % Non-linear constraints, of which there are none
    options);  % Other optimization options
time = toc;

% Display log likelihood
disp(strcat('Log-likelihood', num2str(-ll)))

% Get analytic standard errors, based on properties of correctly specified
% MLE (variance is the negative inverse of Fisher information, estimate
% this using sample analogue)
V = inv(I);
SE_a = sqrt(diag(V));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Part 3: Display the results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set up a cell array to display the results
D = cell(length(theta_hat)+1,3);

% Add headers
D(1,:) = {'Parameter', 'Estimate', 'SE'};

% Fill in the first column of the cell array with coefficient labels
% Set up a counter to mark which row needs to be filled in next
k = 2;

% Add labels for the parts of mu_beta
for i=1:length(mu_beta0)
    D(k,1) = {strcat('beta_', num2str(i))};
    k = k + 1;
end

% Add labels for the diagonal elements of Sigma
for i=1:length(mu_beta0)
    D(k,1) = {strcat('Sigma_', num2str(i), num2str(i))};
    k = k + 1;
end

% Add labels for the off-diagonal elements by hand, since these aren't
% always all estimated, and I don't want to figure out how to automate this
D(k,1) = {'Sigma_23'};
k = k + 1;  % Keep track of the rows!

% Add labels for the elements of alpha
for i=1:length(alpha0)
    D(k,1) = {strcat('alpha_', num2str(i))};
    k = k + 1;
end

% Add labels for the elements of gamma
for i=1:length(gamma0)
    D(k,1) = {strcat('gamma_', num2str(i))};
    k = k + 1;
end

% Add theta_hat and its standard error to the results
D(2:end,2:end) = num2cell([theta_hat', SE_a]);

% Display the results
disp(D)

% Display how much time the estimation took
disp(['Time elapsed: ', num2str(time), ' seconds'])