function S = logsumexp_safe(X)
    % Calculates the log of the sum of the exponentiated elements of X,
    % using the 'log-exp-sum trick' to prevent overflow and limit the error
    % caused by underflow
    S = max(X) + log(sum(exp(X - max(X))));
end