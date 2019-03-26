function L = ll(C, S, chi, V0, theta, P, beta, tolEV)
% Computes the (negative) log-likelihood
%
% Inputs
% C: [N,1] vector, observed choices
% S: [K*J,3] matrix, contains all components of flow utility for each
%                    possible state, for each action. Since a state is a
%                    market state - past choice combination, and there are
%                    K market states and J choices, there are K*J possible
%                    states. Flow utility is zero if the firm chooses not
%                    to enter the market, and a constant plus a coefficient
%                    on the market state plus a coefficient on the past
%                    choice if the firm enters the market. S allows me to
%                    calculate a matrix of flow utilities for each possible
%                    choice, given the current state.
% xi: [N,1] vector, observed state
% V0: [K*J,J] matrix, initial guess for the value function
% theta: [3,1] vector, parameters for flow utility
% P: {1,J} array, conditional transition probabilities. The j-th element
%                 has to be a [K*J,K*J] matrix giving the transition
%                 probabilites between all states, conditional on choosing
%                 the j-th possible action.
% beta: scalar, discount factor
% tolEV: scalar, tolerance for value function iteration
% 
% Outputs
% L: scalar, negative log-likelihood

% Make sure theta is a column vector
if size(theta,1) < size(theta,2)
    theta = theta.';
end

% Get number of options in the choice set
J = length(unique(C));

% Set up flow utilities
U = zeros(size(V0));

% Replace flow utilities in the case of option 2 being chosen
U(:,J) = S * theta;

% Use value function iteration to get expected value function
EV = EVsolve(V0, U, P, beta, tolEV);

% Add flow utility to get actual value function
V = U + beta * EV;

% Get number of choice situations
N = length(C);

% Set up matrix of value for given data set
W = zeros(N,1+J);

% Go through all posible options
for i=1:J
    % Make an index indicating where the current option is being chosen
    cidx = (C == (i-1));
    
    % In situations where the current option is being chosen, replace the
    % entry in the first column with the value function for being in the
    % observed state, and making the given choice
    W(cidx,1) = V(chi(cidx),i);
    
    % Replace the column corresponding to this choice with the value
    % function for being in the observed state, and making the given choice
    W(:,i+1) = V(chi,i);
end

% Get the row-wise maximum, which will be useful for preventing overflow
A = max(W,[],2);

% Get the choice probability
L = exp(W(:,1) - A) ./ sum(exp(W(:,2:end) - A * ones(1,J)),2);

% Get the negative log-likelihood
L = -sum(log(L));
end