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

% Get shifted version of plan ID variable
shifted_plan = circshift(insurance_data_red{:, {v_pid}}, 1);

% Check whether people switched plans
switched = (shifted_plan ~= insurance_data_red{:, {v_pid}});

% Specify name for plan retainment variable
v_ret = 'retained';

% Add the variable to the data set
insurance_data_red = addvars(insurance_data_red, ~switched, ...
    'NewVariableNames', v_ret);

% Specify name of individual ID variable
v_id = 'indiv_id';

% Specify name of choice situation ID
v_csid = 'choice_sit';

% Add plan retainment indicator to full data set, by joining on the
% individual and plan ID variables
insurance_data = join(insurance_data, ...
    insurance_data_red(:, {v_id, v_csid, v_ret}));

% Note that the plan retainment indicator should be one only for the
% actually chosen plan, so enforce that
insurance_data{:, {v_ret}} = (insurance_data{:, {v_ret}} & cidx);

% Specify name of comparison tool access indicator
v_tool = 'has_comparison_tool';

% Specify name for tool access and plan retention interaction
v_ret_tool = strcat(v_ret, '_times_', v_tool);

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

% Specify name of plan premium variable
v_pre = 'premium';

% Specify name of age variable
v_age = 'age';

v_pre_age = strcat(v_pre, '_times_', v_age);
v_pre_loginc = strcat(v_pre, '_times_', v_loginc);
vars_pre_dem = 'premium_times_demographics';
insurance_data = addvars(insurance_data, ...
    insurance_data{:, {v_age v_loginc}} ...
    .* (insurance_data{:, {v_pre}} * ones(1,2)), ...
    'NewVariableNames', vars_pre_dem);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Part 2: Structural estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Change back to main directory
cd(mdir)

% Specify name of plan coverage and service quality variables
v_cov = 'plan_coverage';  % Coverage
v_svq = 'plan_service_quality';  % Service quality

% Get quadrature points and weights
[qp, qw] = nwspgr('KPN', 3, 4);

% Transpose them
qp = qp.';
qw = qw.';

% Make data set
X = insurance_data{:, {v_pre, v_cov, v_svq, ...  % Plan characteristics
    v_ret, v_ret_tool, ...  % Switching cost and tool access
    vars_pre_dem}};  % Demographics interacted with plan premium

% Get choice situation ID as vector
sit_id = insurance_data{:, {v_csid}};

% Set optimization options
options = optimset('GradObj','off','HessFcn','off','Display','off', ...
    'TolFun',1e-6,'TolX',1e-6);
mu_beta0 = [-.2, .5, .5];
sigma0 = [1, 1, 1, -.2];
alpha0 = [.5, -.1];
gamma0 = [10, 20];

% Perform MLE
tic
[theta_hat,~,~,~,~,I] = fminunc( ...
    @(theta)ll_structural(theta(1:3), ...  % mu_beta
    [theta(4:6), 0, 0, theta(7)], ... % sigma
    theta(8:9), ...  % alpha
    theta(10:11), ... % gamma
    X, sit_id, cidx, qp, qw), ...
    [mu_beta0, sigma0, alpha0, gamma0], ... % initial values
    options);
time = toc;

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