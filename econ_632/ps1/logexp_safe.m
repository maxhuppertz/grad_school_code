function S = logexp_safe(X)
    S = max(X) + log(exp(X - max(X)) * );
end