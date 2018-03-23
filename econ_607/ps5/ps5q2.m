% How to clean everything
clear

% Save current working directory, which should be the parent directory of
% the Dynare file this code calls - ps5q2.mod
dir_orig = pwd;

% Specify name of data directory
ddir = './dat_ps5q2';

% Create it if it doesn't exist
if ~exist(ddir, 'dir')
    mkdir(ddir)
end

% Specify name of Dynare file
dyn = 'ps5q2';

% Specify graph format (has to match the format specified in the Dynare
% file!)
gform = '.eps';

% Specify folder containing Dynare file
dir_dyn = './dynare_ps5q2';

% Choose start date for time series to be retrieved (YYYY-MM-DD)
sdate = '1948-01-01';

% Make a string containing today's date (to be able to get most recent
% data when entering other optional arguments)
curdate = clock;
curdate = strcat(num2str(curdate(1)), '-', ...
    num2str(curdate(2), '%02d'), '-', ...
    num2str(curdate(3), '%02d'));

% Get data on TFP from San Francisco FED
% Specify URL and file name
tfpurl = ...
    'https://www.frbsf.org/economic-research/files/quarterly_tfp.xlsx';
tfpfnm = 'sffed_tfp.xlsx';

% Change to data directory and save data
cd(ddir)
websave(tfpfnm, tfpurl, weboptions('Timeout',Inf));

% Read data
tfpset = xlsread(tfpfnm, 'quarterly');

% Extract TFP data (okay, so the date is hard-coded here, and this better
% match up with your start date, but basically it's quarters since 1947:Q1
% plus one; the eleventh column is the TFP data I need) 
tfp = tfpset(5:end, 11);

% Convert it to levels
tfp(1) = 1;

% The architecture of the spreadsheet is such that there's a NaN row before
% some summary statistics at the end, so cut out before that happens
cut = find(isnan(tfp));
tfp = tfp(1:cut-1);

% Go through all TFP values
for i = 2:length(tfp)
    % Iteratively convert them back to levels
    tfp(i) = tfp(i-1) * exp(tfp(i)/400);
end

% Regress log TFP on an intercept and a time trend
B = regress(log(tfp), [ones(length(tfp), 1), (1:1:length(tfp))']);

% Get residuals
u = log(tfp) - (ones(length(tfp), 1)*B(1) + (1:1:length(tfp))'*B(2));

% Regress residuals on an intercept and themselves
R = regress(u(2:end), [ones(length(u)-1, 1), u(1:end-1)]);

% Get residuals from this as well
epsilon = u(2:end)  - (ones(length(u)-1, 1)*R(1) + u(1:end-1)*R(2));

% Change to Dynare directory
cd(dir_orig)
cd(dir_dyn)

% Set up parameters
beta = .99;
rho = floor(R(2)*100)/100;
alpha = 1/3;
v = .72;  % For iv), I have no idea how to get close to SD(n) = .019
A_bar = 1;
delta = .025;

% chi is set up so that people work a third of their time
chi = (1/3)^((1-v)/v) * (((1 - beta * (1 - delta)) * (1-alpha)) / ...
    (1 - beta * (1 - delta) - delta * alpha * beta));

% Set parameters only used in Dynare file
sig = std(epsilon);

% Calculate steady state values
A_st = A_bar;
n_st = (((1 - beta * (1 - delta)) * (1 - alpha)) / ...
    ((1 - beta * (1 - delta) - delta * alpha * beta) * chi))^(v / (1 - v));
k_st = (n_st^(1 - alpha) * ((alpha*beta) / ...
    (1 - beta * (1 - delta))))^(1/(1 - alpha));
I_st = delta * k_st;
c_st = k_st^alpha * n_st^(1 - alpha) - I_st;

% Save parameter values as .mat file, so Dynare can access them
% This includes the steady state values, which Dynare uses as initial
% values when solving the model
save(strcat(dyn, '_init_params.mat'))

% Display steady state values
disp(['ln(k) = ', num2str(log(k_st)), '; ln(I) = ', num2str(log(I_st)), ...
    '; ln(n) = ', num2str(log(n_st)), '; ln(c) = ', num2str(log(c_st))])

% Run Dynare
dynare ps5q2

% Change back to the parent directory
cd(dir_orig);

% Set smoothing parameter for HP filter
mu = 1600;

% List of variable names (has to be in the same order as in the Dynare
% file, else there'll be lots of mix ups!), excluding A (which would
% otherwise be first on the list)
vnames = {'n', 'k', 'c', 'I'};

% List of steady state values, in the same order!
stval = [n_st, k_st, c_st, I_st];

for i = 1:length(vnames)
    % Retrieve the time series for the current variable
    y = oo_.endo_simul(i+1,:);
    
    % Apply HP filter
    [y_dt, T] = hp_filter(y, mu);
    sd = std(y_dt);
    
    % Display the simulated 'volatility' (standard deviation)
    disp(strcat('SD(', vnames(i), '):', {' '}, num2str(sd)))
end