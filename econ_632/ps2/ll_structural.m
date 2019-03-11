function L = ll_structural(mu_beta, sigma, alpha, gamma, X, sit_id, ...
    cidx, qp, qw)
% Calculates the log-likelihood of a simple model of insurance plan choice,
% allowing for switching cost, as well as correlated tastes for plan
% coverage and service quality
%
% Inputs
% mu_beta: [d,1] vector, expected value for random coefficients on plan
%          premium, coverage, and service quality
% sigma: [d^2,1] vector, components of the covariance matrix for those
%        random coefficients
% alpha: [2,1] vector, coefficients on switching cost and switching cost
%        interacted with access to a comparison tool
% gamma: [k,1] vector, coefficients on demographics interacted with plan
%        premium
% X: [N,d+2+k] matrix, data on plan premium, coverage, and service quality,
%    a plan retention indicator, the plan retention indicator interacted
%    with a tool access indicator, and demographics interacted with plan
%    premium, in that order
% sit_id: [nsit,1] vector, choice situation ID
% cidx: [nsit,1] vector, indicates chosen plans

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Part 1: Get covariance matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Start by getting diagonal elements
Sigma = diag(sigma(1:length(mu_beta)));

% Fill in off-diagonal elements
%
% Set up a counter for the elements in the sigma vector
k = 1;

% Go through all but the last rows of the covariance matrix
for i = 1:length(mu_beta)-1
    % Go through all elements of that row past the diagonal
    for j = i+1:length(mu_beta)
        % Replace the i,j element with the covariance element
        Sigma(i,j) = sigma(length(mu_beta) + k);
        
        % Since this is symmetric, replace the j,i element as well
        Sigma(j,i) = sigma(length(mu_beta) + k);
        
        % Increase the element counter
        k = k + 1;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Part 2: Define conditional choice probability function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Make sure alpha is a column vector
if size(alpha, 1) == 1
    alpha = alpha.';
end

% Make sure gamma is a column vector
if size(gamma, 1) == 1
    gamma = gamma.';
end

% Set up a function to calculate the weighted conditional choice
% probability for a given vector beta = b and quadrature weight w
function ccp = cond_cp(b)
    % Get vector of all parameters
    theta = [b; alpha; gamma];

    % Get maximum value of parts inside the exponential for each choice
    % situation. This is needed to make an overflow adjustment.
    A = accumarray(sit_id, X * theta, [], @max);

    % Calculate numerator, using only chosen quantities, subtracting the
    % maximum within choice situation. This prevents the expression being
    % evaluated as Inf
    ccp_num = exp(X(cidx, :) * theta - A);

    % Calculate parts of the denominator, using all quantities, subtracting
    % the maximum within choice situation (the indexing works out such that
    % the maximum is repeated exactly as many times as needed). Again, this
    % prevents infinity
    ccp_dnm = exp(X * theta - A(sit_id));

    % Add up values within choice situation to get actual denominator
    ccp_dnm = accumarray(sit_id, ccp_dnm);

    % Divide numerator by denominator
    ccp = (ccp_num ./ ccp_dnm);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Part 3: Perform Monte Carlo integration, get log likelihood
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Use the chol() function's second output to check whether Sigma is
% positive definite
[C,p] = chol(Sigma);

% Check whether the second output is zero, indicating that Sigma is not
% positive definite, which will make it impossible to draw from the implied
% distribution
if p > 0
    % If so, set the likelihood to zero
    L = 0;
else
    % Make sure mu_beta is a column vector
    if size(mu_beta,1) == 1
        mu_beta = mu_beta.';
    end

    % Make sure the quadrature points are an array of column vectors
    if size(qp,1) > size(qp,2)
        qp = qp.';
    end

    % Get number of quadrature points
    np = size(qp, 2);
    
    % Make sure the quadrature weights are np x 1
    if size(qw,1) < size(qw,2)
        qw = qw.';
    end
    
    % Scale quadrature points
    qp = mu_beta + C * qp;
    
    %for i=1:np
    %    qp(:,i) = mu_beta + C * qp(:,i);
    %end
        
    % Get weighted conditional choice probabilities for all quadrature
    % points, which will create an nsit X np cell array, where nsit is the
    % number of choice situations
    L = arrayfun(@(i)cond_cp(qp(:,i)), (1:np), 'UniformOutput', false);
    
    % Perform the sparse grids integration, by getting the weighted average
    % within each row of the array. This converts the array to a matrix
    % before doing the integration, since those are a bit easier to work
    % with than cell arrays.
    L = cell2mat(L) * qw;
    
    % Get negative log likelihood by taking the log, summing up across
    % choice situations, and multiplying by -1
    L = -sum(log(L));
end
end