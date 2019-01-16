function L = ll_multilogit_rc(beta_bar,sigma2,xi,p,c,method)
% Set some hard coded parameters
tol = 10^(-14);  % Tolerance for direct integration
D = 500;  % Number of draws for Monte Carlo integration
k = 4;  % Precision for sparse grids integration

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
if strcmp(method,'integral')
    % Calculate log-likelihood
    L = -sum(log(integral(@(b)CP(b,beta_bar,sigma2,0),-Inf,Inf, ...
        'ArrayValued',true,'RelTol',0,'AbsTol',tol)));
    
    % If this is (erroneously) evaluated to be infinite, set it to zero
    if L == Inf
        L = 0;
    end
elseif strcmp(method,'monte_carlo')
    % Draw a matrix of i.i.d. N(beta_bar, sigma2) random variables
    L = randn(n,D) * sqrt(sigma2) + beta_bar;
    
    % Apply the choice probability function to each column of the random
    % numbers, which will return a 1 x D cell array where each cell
    % contains an n x 1 vector of conditional choice probabilities
    L = arrayfun(@(i)CP(L(:,i),beta_bar,sigma2,1),(1:D), ...
        'UniformOutput',false);
    
    % Convert the cell array into an n x D matrix using cell2mat, take the
    % mean within rows (this gives the choice probabilities), then take the
    % log and sum up
    L = -sum(log(mean(cell2mat(L),2)));
elseif strcmp(method,'sparse')
    % Get quadrature points for N(0,1) variable
    [b,w] = nwspgr('KPN',1,k);
    
    % Adjust quadrature points for N(beta_bar,sigma2) variable
    % Does this have any theoretical justification? I'm not sure.
    b = (b + beta_bar)*sqrt(sigma2)';
    
    % Set up matrix where each row contains quadrature points for each
    % individual
    L = ones(n,1)*b';
    
    % Evaluate exponential ratio at each point, for all individuals, which
    % will return a 1 x size(b,1) cell array, where each cell contains a
    % vector of evaluated quadrature points
    L = arrayfun(@(i)CP(L(:,i),beta_bar,sigma2,1),(1:size(b,1)), ...
        'UniformOutput',false);
    
    % Convert the cell array into an n x 500 matrix using cell2mat, take
    % the weighted mean within rows using quadrature weights (this gives
    % the expected choice probabilities), then take the log and sum up
    L = -sum(log(sum(cell2mat(L).*(ones(n,1)*w'),2)));
else
    % Display an error message
    disp('Method unknown')
    
    % Set L to NaN
    L = NaN;
    
    % Return to main program
    return
end
end