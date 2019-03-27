function p1 = probin(SD, S, V0, P, beta, theta, tolEV)

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

% Sum up across market states
p1 = SD.' * CCP;
end