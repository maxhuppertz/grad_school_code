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

% Specify name of state ID variable
v_state = 'x';

% Get version of state ID shifted back by one period
shifted_state = circshift(entry_data{:, {v_state}}, -1);

% Specify name of market ID variable
v_mid = 'Market';

% Get 'backshifted' version of market ID
shifted_id = circshift(entry_data{:, {v_mid}}, -1);

% Mark observations which should be used when estimating transition
% probabilities. This excludes the last observation for a given market,
% since the following state is unobserved.
use_tprob = (entry_data{:, {v_mid}} == shifted_id);

% Specify action variable
v_act = 'i';

% Get unique values of action variable
actions = unique(entry_data{:, {v_act}});

% Count the total number of actions
J = length(actions);

% Get unique values of state variable
states = unique(entry_data{:, {v_state}});

% Count number of states
K = length(states);

% Make a transition matrix
P = zeros(K,K*J);

% Set up an action counter, starting at zero
k = 0;

% Go through all actions
for c = min(actions):max(actions)
    % Make an indicator for this action
    Ic = (entry_data{:, {v_act}} == c) & use_tprob;
    
    % Count the number of times the action was chosen
    nc = sum(Ic);
    
    % Go through all 'from' states
    for i = min(states):max(states)
        % Make an indicator for this action being chosen
        ti = (entry_data{:, {v_state}} == i) & Ic & use_tprob;
        
        % Count how often this action was chosen
        ni = sum(ti);
        
        % Go through all 'to' states
        for j = min(states):max(states)
            % Count the number of transitions from - to
            tij = (entry_data{:, {v_state}} == i) ...
                & (shifted_state == j) & Ic & use_tprob;
            
            % Replace the entry in the transition matrix as the sum
            P(i,k+j) = sum(tij);
        end
        % Convert sums to estimated probabilities
        P(i,k+1:k+K) = P(i,k+1:k+K) / ni;
    end
    % Increase the action counter
    k = k + K;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Part 2: Structural estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Change back to main directory
cd(mdir)

% Set discount factor
beta = .95;