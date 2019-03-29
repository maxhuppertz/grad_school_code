function p1 = probin(SD, S, V0, P, theta, beta, tolEV)
% Calculate the unconditional probability of entry
%
% Inputs
% SD: [K,1] vector, stationary distribution across market states
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
% V0: [K*J,J] matrix, initial guess for the value function
% theta: [3,1] vector, parameters for flow utility
% P: {1,J} array, conditional transition probabilities. The j-th element
%                 has to be a [K*J,K*J] matrix giving the transition
%                 probabilites between all states, conditional on choosing
%                 the j-th possible action.
% beta: scalar, discount factor
% tolEV: scalar, tolerance for value function iteration
%
% Get number of options
J = size(V0,2);

% Set up flow utilities
U = zeros(size(V0));

% Replace flow utilities in the case of option 2 being chosen
U(:,J) = S * theta;

% Get value function
V = Vsolve(V0, U, P, beta, tolEV);

% Get number of market states
K = length(SD);

% Get maximum of value function for each state
A = max(V,[],2);

% Get conditional probability of choosing i = 1, i.e. the second option,
% for each state
CCP = exp(V(:,J) - A) ./ sum(exp(V - A * ones(1,J)),2);

% Sum up within market state, i.e. find the two states corresponding to
% a given market state and add the probabilites up
CCP = accumarray([(1:K),(1:K)].',CCP);

% Multiply with the stationary probability of being in each market state
% and sum up
p1 = SD.' * CCP;
end