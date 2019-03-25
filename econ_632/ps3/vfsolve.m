function V = vfsolve(V0, P, actions, states, R, beta, theta, tol)

K = length(states);

J = length(actions);

converged = 0;

while converged==0
    V1 = zeros(size(V0));
    
    % If i = 0
    V1(1:K) = P(:,1:K) * beta * V0(1:K);
    
    V1(1:K) = V1(1:K) + ...
        P(:,1:K) * (theta(1) + theta(2)*X - theta(3) + beta * V0(1:K));
    
    % If i = 1
    
    diff = max(abs(V1 - V0));
    
    if diff<tol
        converged = 1;
    end
    
    V0 = V1;
end
end