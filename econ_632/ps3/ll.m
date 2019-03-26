function [L,G] = ll(C, S, Z, V0, theta, P, beta, tolEV)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Part 1: Log-likelihood
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Make sure theta is a column vector
if size(theta,1) < size(theta,2)
    theta = theta.';
end

% Get number of options in the choice set
J = length(unique(C));

% Set up flow utilities
U = zeros(size(V0));

%theta(1:2) = exp(theta(1:2));

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

g = zeros(N,2);

% Go through all posible options
for i=1:J
    % Make an index indicating where the current option is being chosen
    cidx = (C == (i-1));
    
    % In situations where the current option is being chosen, replace the
    % entry in the first column with the value function for being in the
    % observed state, and making the given choice
    W(cidx,1) = V(Z(cidx),i);
    
    % Replace the column corresponding to this choice with the value
    % function for being in the observed state, and making the given choice
    W(:,i+1) = V(Z,i);
    
    g(cidx,1) = Z(cidx) - 5 * (Z(cidx) > 5);
    g(cidx,2) = (Z(cidx) <= 5);
end

% Get the row-wise maximum, which will be useful for preventing overflow
A = max(W,[],2);

% Get the choice probability
L = exp(W(:,1) - A) ./ sum(exp(W(:,2:end) - A * ones(1,J)),2);

% Get the negative log-likelihood
L = -sum(log(L));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Part 2: Gradient
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

D = exp( sum(W(:,2:end), 2) - 2*A) ...
    ./ (exp(W(:,2) - A) + exp(W(:,3) - A)).^2;

G = zeros(3,1);

G(1,1) = sum(D,1);

G(2,1) = sum(g(:,1) .* D,1);

G(3,1) = sum(g(:,2) .* D,1);

G = -G;

end