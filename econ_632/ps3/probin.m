function p1 = probin(S, V0, P, p, theta, beta, tolEV)
% Calculate the unconditional probability of entry
%
% Inputs
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
% P: {1,J} array, conditional transition probabilities. The j-th element
%                 has to be a [K*J,K*J] matrix giving the transition
%                 probabilites between all states, conditional on choosing
%                 the j-th possible action.
% p: [K,K] matrix, transition probabilities for market states
% theta: [3,1] vector, parameters for flow utility
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
K = size(p,1);

% Set up transition matrix for complete state as transition probabilities
% for x part of the state, repeated as many times as there are actions
% (since x is Markov independent of the action)
p_chi = kron(ones(J,J),p);

% Get maximum of value function along second dimension (i.e. betwee choices
% within a given state)
A = max(V,[],2);

% Calculate conditional choice probabilites
CCP = exp(V - A * ones(1,J)) ...
    ./ (sum(exp(V - A * ones(1,J)),2) * ones(1,J));

% Restack them, in a way that they can be expanded using a Kronecker
% product, and then pointwise multiplied to the x state transition matrices
% to get the full transition matrix for the complete state
CCP = [[CCP(1:K,1).';CCP(K+1:end,1).'],[CCP(1:K,2).';CCP(K+1:end,2).']];

% Do the Kronecker product
CCP = kron(CCP,ones(5,1));

% Get the transition matrix for the complete state
p_chi = p_chi .* CCP;

% Get right eigenvectors of the transition matrix
[vec,lambda] = eig(p_chi.');

% Get the index of the eigenvector associated with the unit eigenvalue
statidx = (round(diag(lambda),5) == 1);

% Get the associated eigenvector
statdist = vec(:,statidx);

% Normalize it
statdist = statdist / sum(statdist,1);

% Get the fraction of time spent in the market
p1 = sum(statdist(end-K:end));
end