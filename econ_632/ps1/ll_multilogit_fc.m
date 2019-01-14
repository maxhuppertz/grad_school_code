function [L,G] = ll_multilogit_fc(beta,xi,p,c)
% Determine number of individuals and options
[n,J] = size(p);

% Make an index of observed choices
cidx = sub2ind([n,J],(1:n)',c);

% Make a matrix where column j contains xi(j)
X = ones(n,1) * xi;

% Calculate log-likelihood
L = -sum(beta*p(cidx) + X(cidx) - logsumexp_safe(beta*p + X,2));

% Calculate the derivative with respect to beta
dbeta = ...
    -sum(p(cidx) - sum(p.*exp(beta*p + X),2) ./ sum(exp(beta*p + X),2));

% Calculate the derivatives for each xi_k
dxi = zeros(J,1);
for k = 1:J
    dxi(k,1) = ...
        -sum((c==k) - exp(beta*p(:,k) + X(:,k)) ./ sum(exp(beta*p + X),2));
end

% Combine the derivatives to a gradient
G = [dbeta; dxi];

dbeta2 = sum( (sum((p.^2).*exp(beta*p + X),2) ...
    .* sum(exp(beta*p + X),2) - sum(p.*exp(beta*p + X),2).^2) ...
    ./ (sum(exp(beta*p + X),2).^2) );

dxi2(k,1) = zeros(J,1);
for k = 1:J
    dxi2(k,1) = sum( (exp(beta*p(:,k) + X(:,k)) ...
        .* sum(exp(beta*p + X),2) - exp(beta*p(:,k) + X(:,k)).^2) ...
        ./ (sum(exp(beta*p + X),2).^2) );
end
end