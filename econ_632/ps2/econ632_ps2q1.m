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

insurance_data_red = addvars(insurance_data_red, );

% Change back to main directory
cd(mdir)

% Get quadrature points and weights
[qp, qw] = nwspgr('KPN', 3, 4);

% Transpose them
qp = qp.';
qw = qw.';

% test values
mu_beta = [1,2,3].';
sigma = [10, 20, 30, 12, 13, 23].';

%check = ll_structural(mu_beta, sigma, 1, 1, 1, 1, 1, 1, 1, 1, qp, qw);
%disp(sum(check,2))