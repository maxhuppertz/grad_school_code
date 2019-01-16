function L = ll_multilogit_rc(beta_bar,sigma2,xi,p,c,method)
% Determine number of individuals and options
[n,J] = size(p);

% Make an index of observed choices
cidx = sub2ind([n,J],(1:n)',c);

% Make a matrix where column j contains xi(j)
X = ones(n,1) * xi;

function I = wnormpdf(b,beta_bar,sigma2)
    I = (1/sqrt(2*pi*sigma2)) ...
        * (exp(b.*p(cidx) + X(cidx)) ./ sum(exp(b.*p + X),2)) ...
        * exp(-((b-beta_bar).^2)/(2*sigma2));
    I(isnan(I)) = 0;
end

% Calculate log-likelihood
if method == 'integral'
    L = -sum(log(integral(@(b)wnormpdf(b,beta_bar,sigma2),-Inf,Inf, ...
        'ArrayValued',true)));
    if L == Inf
        L = 0;
    end
end
end