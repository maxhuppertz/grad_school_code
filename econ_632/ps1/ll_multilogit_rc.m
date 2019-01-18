function L = ll_multilogit_rc(beta_bar,sigma2,xi,p,c,method,tol,qp,w)
% Calculates the log-likelihood for a logit model with a random coefficient
% on price, distributed as an N(beta_bar,sigma2) random variable, and fixed
% intercepts for each of the J alternatives
%
% Inputs
% beta_bar: scalar, mean of the price distribution
% sigma2: scalar, variance of the price distribution
% xi: [1,J] vector, alternative-specific intercepts
% p: [n,J] matrix, prices
% c: [n,1] vector, index of chosen goods
% method: string, either 'direct' for direct numerical integration using
%                 Matlab's intergral() function, or 'points' if you want
%                 to provide quadrature points and weights instead
% tol: scalar, tolerance for direct numerical integration
% qp: either [1,K] vector or [n,K] matrix, quadrature points
% w: [1,K] vector, quadrature weights
%
% Output
% L: scalar, log-likelihood

% Determine number of individuals and options
[n,J] = size(p);

% Make an index of observed choices
cidx = sub2ind([n,J],(1:n)',c);

% Make a matrix where column j contains xi(j)
X = ones(n,1) * xi;

% Define the choice probability for a given beta_i = b, which this is all
% based on, as a function. This is needed for all methods of integration
% used in this function. It has an option for only getting the exponential
% ratio part, i.e. just the choice probability conditional on beta_i = b,
% which is used for Monte Carlo and sparse grids integration. If this
% option is turned off, the function instead returns the exponential ratio
% times the N(beta_bar,sigma2) PDF, which is used for direct integration.
function I = CP(b,beta_bar,sigma2,probonly)
    % If b is provided as a scalar, make it into a vector
    % The comment beside the next line suppresses an unnecessary warning
    if size(b) == [1,1] %#ok<BDSCA>
        b = b*ones(n,1);
    end
    
    % Check whether to get the conditional choice probability only
    if probonly
        % Calculate only the exponential ratio
        I = (exp(b.*p(cidx) + X(cidx)) ...
            ./ sum(exp((b*ones(1,J)).*p + X),2));
    else
        % Calculate choice probability
        I = (1/sqrt(2*pi*sigma2)) ...
            * (exp(b.*p(cidx) + X(cidx) - ((b-beta_bar).^2)/(2*sigma2)) ...
            ./ sum(exp((b*ones(1,J)).*p + X),2));
    end
    
    % Set NaN values to zero
    I(isnan(I)) = 0;
end

% Calculate log-likelihood
if strcmp(method,'direct')
    % Calculate log-likelihood
    L = -sum(log(integral(@(b)CP(b,beta_bar,sigma2,0),-Inf,Inf, ...
        'ArrayValued',true,'RelTol',0,'AbsTol',tol)));
    
    % If this is (erroneously) evaluated to be infinite, set it to zero
    if L == Inf
        L = 0;
    end
elseif strcmp(method,'points')
    % Get number of quadrature points
    K = size(qp,2);
    
    % Adjust N(0,1) quadrature points to N(beta_bar,sigma2) variable
    qp = (qp * sqrt(sigma2) + beta_bar);
    
    % If quadrature points are the same across individuals, make them into
    % a matrix where each row contains the quadrature points
    if size(qp,1) == 1
        qp = ones(n,1)*qp;
    end
    
    % Evaluate exponential ratio at each point, for all individuals, which
    % will return a 1 x K cell array, where each cell contains an n x 1
    % vector of evaluated quadrature points
    L = arrayfun(@(i)CP(qp(:,i),beta_bar,sigma2,1),(1:K), ...
        'UniformOutput',false);
    
    % Convert the cell array into an n x 500 matrix using cell2mat, take
    % the weighted mean within rows using quadrature weights (this gives
    % the expected choice probabilities), then take the log and sum up
    L = -sum(log(sum(cell2mat(L).*(ones(n,1)*w),2)));
else
    % Display an error message
    disp('Method unknown')
    
    % Set L to NaN
    L = NaN;
    
    % Return to main program
    return
end
end