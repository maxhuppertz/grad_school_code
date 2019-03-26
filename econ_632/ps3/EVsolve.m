function EV = EVsolve(V0, U, P, beta, tolEV)
% Solves for the expected value function
% Outputs
% EV: [K,J] matrix, resulting value function. K is the number of states,
%                   and J is the number of actions. The [i,j] element is
%                   the value function for being in state i and making
%                   choice j.

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
    
    % Exponentiate, sum up across actions (hence along the second axis,
    % then take the log. Each element of the resulting column vector
    % corresponds to a given state
    v = logsumexp_safe(v,2);
    
    % Go through all possible choices
    for i=1:J
        % Iterate value function for each state, conditional on the current
        % choice. The only thing that changes is the conditional
        % probability matrix. The way this works is that the transition
        % probabilities between the x part of the state are always the
        % same, regardless of the i_{-1} part of the state. The only
        % difference is that, depending on the current period's i, next
        % period's state will either have i_{-1} = 0, or i_{-1} = 1, with
        % certainty. So I can 
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

% Use final value function as output
EV = V1;
end