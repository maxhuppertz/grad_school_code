function S = logsumexp_safe(X,dim)
    % Calculates the log of the sum of the exponentiated elements of X,
    % using the 'log-exp-sum trick' to prevent overflow and limit the error
    % caused by underflow, along dimension dim
    S = max(X,[],dim) + log(sum(exp(X - max(X,[],dim)),dim));
end