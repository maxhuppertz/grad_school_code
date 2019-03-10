function L = ll_structural(mu_beta, sigma, alpha, gamma, X, sit_id, ...
    cidx, qp, qw)

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
%%% Part 2: Scale quadrature points
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Make sure mu_beta is a column vector
if size(mu_beta, 1) == 1
    mu_beta = mu_beta.';
end

% Get the number of quadrature points
[~, np] = size(qp);

% The first element of beta will be integrated over last, and just uses its
% marginal distribution
qp(1,:) = qp(1,:) * Sigma(1,1) + mu_beta(1);

% The second element will be integrated over second to last, and has to use
% its conditional distribution, conditioning on the first element
%
% Calculate conditional mean
condmu2 = mu_beta(2) + (Sigma(2,1) / Sigma(1,1)) * (qp(1,:) - mu_beta(1));

% Calculate conditional variance
condV2 = Sigma(2,2) - (Sigma(2,1)/Sigma(2,2))*Sigma(1,2);

% Scale quadrature points
qp(2,:) = qp(2,:) * condV2 + condmu2;

% The third element will be integrated over first, and has to use its
% conditional distribution, conditioning on the first and second elements
%
% Stack scaled quadrature points for the other two elements
stackedqp12 = [qp(1,:);qp(2,:)];

% Calculate conditional mean
condmu3 = mu_beta(3) + (Sigma(3,1:2) / Sigma(1:2,1:2)) ...
    * (stackedqp12 - mu_beta(1:2) * ones(1,np));

% Calculate conditional variance
condV3 = Sigma(3,3) - (Sigma(3,1:2) / Sigma(1:2,1:2)) * Sigma(1:2,3);

% Scale quadrature points
qp(3,:) = qp(3,:) * condV3 + condmu3;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Part 3: Conditional choice probability function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get number of choice situations
nsit = sum(cidx);

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
function ccp = cond_cp(b, w)
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
    
    % Divide numerator by denominator, multiply by weight
    ccp = (ccp_num ./ ccp_dnm) * w;
    
    % Set NaN values to zero
    ccp(isnan(ccp)) = 0;
    
    % Set Inf values to zero
    ccp(isinf(ccp)) = 0;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Part 4: Perform integration, get log likelihood
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get weighted conditional choice probabilities for all quadrature points
% and weights, which will create an nsit X np cell array, where nsit is the
% number of choice situations
L = arrayfun(@(i)cond_cp(qp(:,i), qw(i)), (1:np), 'UniformOutput', false);

% Perform the sparse grids integration, by summing up within the rows of
% the array (also converts to a matrix, since that's easier to handle)
L = sum(cell2mat(L), 2);

% Get negative log likelihood by taking the log, and summing up across
% choice situations
L = -sum(log(L));

% Get only the real part of the log likelihood
L = real(L);

% Check whether the likelihood is NaN or Inf
if isnan(L) || isinf(L)
    % If so, set it to zero
    L = 0;
end
end