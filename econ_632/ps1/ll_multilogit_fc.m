function [L,G,H] = ll_multilogit_fc(beta,xi,p,c,normalize)
% Calculates the log-likelihood for a logit model with a fixed coefficient
% on price and fixed intercepts for each of the J alternatives
%
% Inputs
% beta: scalar, coefficient on price
% xi: [1,J] vector, alternative-specific intercepts
% p: [n,J] matrix, prices
% c: [n,1] vector, index of chosen goods
% normalize: True/False or 0/1, if evaluated as true, the function
%            normalizes xi[1,J] = 0, which means it does not return the
%            last element of the Jacobian, and also leaves out the last
%            row and column of the Hessian
%
% Outputs
% L: scalar, log-likelihood
% G: [J+1,1] vector, Jacobian (will be [J,1] if normalize is true)
% H: [J+1,J+1] matrix, Hessian (will be [J,J] if normalize is true)

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

% If normalization is required, leave out xi_J
if normalize
    G = G(1:J,1);
end

% Calculate second derivative w.r.t. beta
dbeta2 = sum( (sum((p.^2).*exp(beta*p + X),2) ...
    .* sum(exp(beta*p + X),2) - sum(p.*exp(beta*p + X),2).^2) ...
    ./ (sum(exp(beta*p + X),2).^2) );

% Calculate second derivatives w.r.t. each xi_k
dxi2 = zeros(J,1);
for k = 1:J
    dxi2(k,1) = sum( (exp(beta*p(:,k) + X(:,k)) ...
        .* sum(exp(beta*p + X),2) - (exp(beta*p(:,k) + X(:,k)).^2)) ...
        ./ (sum(exp(beta*p + X),2).^2) );
end

% Set up Hessian as diagonal matrix of second derivatives
H = diag([dbeta2; dxi2]);

% Fill in cross derivatives w.r.t. to beta and xi_k or xi_k and beta
% (note that these are symmetric)
for k = 1:J
    H(1,k+1) = sum( (p(:,k).*exp(beta*p(:,k) + X(:,k)) ...
        .* sum(exp(beta*p + X),2) - sum(p.*exp(beta*p + X),2) ...
        .* exp(beta*p(:,k) + X(:,k))) ...
        ./ (sum(exp(beta*p + X),2).^2) );
    H(k+1,1) = H(1,k+1);
end

% Fill in cross derivatives w.r.t. xi_j and xi_k (also symmetric)
for j = 1:J
    for k = j+1:J
        H(j+1,k+1) = -sum( (exp(beta*p(:,j) + X(:,j)) ...
            .* exp(beta*p(:,k) + X(:,k))) ...
            ./ (sum(exp(beta*p + X),2).^2) );
        H(k+1,j+1) = H(j+1,k+1);
    end
end

% If normalization is required, leave out xi_J
if normalize
    H = H(1:J,1:J);
end
end