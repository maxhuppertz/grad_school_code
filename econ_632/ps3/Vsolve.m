function V = Vsolve(V0, U, P, beta, tolEV)
% Solves for the value function
%
% Inputs
% V0: [K*J,J] matrix, initial guess for the value function
% U: [K*J,J] matrix, flow utilities for each state and each action, where
%                    states are indexed by rows, and action are indexed by
%                    columns. (That is, the [i,j] element is the flow
%                    utility of being in state i and choosing action j.)
% P: {1,J} array, conditional transition probabilities. The j-th element
%                 has to be a [K*J,K*J] matrix giving the transition
%                 probabilites between all states, conditional on choosing
%                 the j-th possible action.
% beta: scalar, discount factor
% tolEV: scalar, tolerance for value function iteration
%
% Outputs
% V: [K*J,J] matrix, resulting value function. K is the number of market
%                    states, and J is the number of actions. A state is a
%                    market state - past action combinatin, of which there
%                    are K*J. The [i,j] element gives the value of being
%                    in state i, and choosing action j.

% Set convergence indicator to zero
converged = 0;

% Set up next iteration of the value function
V1 = V0;

% Get number of actions
J = size(U,2);

% Iterate to convergence
while converged == 0
    % Flow utility of being in a given state, plus discounted value
    % function for the next period (this is [state, action] indexed)
    v = U + beta * V0;
    
    % Exponentiate, sum up across actions (hence along the second axis),
    % then take the log. Each element of the resulting column vector
    % corresponds to a given state.
    v = logsumexp_safe(v,2);
    
    % Go through all possible choices
    for i=1:J
        % Iterate value function for each state, conditional on the current
        % choice. The only thing that changes is the conditional
        % probability matrix.
        V1(:,i) = cell2mat(P(i)) * v;
    end
    
    % Check for convergence
    if max(abs(V1 - V0)) < tolEV
        % If it has occured, change the convergence flag
        converged = 1;
    end
    
    % Update the value function for the next iteration
    V0 = V1;
end

% Add flow utility to get actual value function
V = U + beta * V1;
end