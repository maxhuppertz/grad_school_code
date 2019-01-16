function L = ll_multilogit_rc(beta_bar,sigma2,xi,p,c,method)
% Determine number of individuals and options
[n,J] = size(p);

% Make an index of observed choices
cidx = sub2ind([n,J],(1:n)',c);

% Make a matrix where column j contains xi(j)
X = ones(n,1) * xi;

% Define the weighted normal PDF this is all based on as a function, which
% is needed for all methods of integration used in this MLE
function I = wnormpdf(b,beta_bar,sigma2)
    % If b is provided as a scalar, make it into a vector
    if size(b) == [1,1]
        b = b*ones(n,1);
    end
    
    % Calculate weighted normal PDF value
    I = (1/sqrt(2*pi*sigma2)) ...
        * (exp(b.*p(cidx) + X(cidx) - ((b-beta_bar).^2)/(2*sigma2)) ...
        ./ sum(exp((b*ones(1,J)).*p + X),2));
    
    % Set NaN values to zero
    I(isnan(I)) = 0;
end

% Calculate log-likelihood
if method == 'integral'
    % Calculate log-likelihood
    L = -sum(log(integral(@(b)wnormpdf(b,beta_bar,sigma2),-Inf,Inf, ...
        'ArrayValued',true,'RelTol',0,'AbsTol',1e-14)));
    
    % If this is (erroneously) evaluated to be infinite, set it to zero
    if L == Inf
        L = 0;
    end
elseif method == 'monte_carlo'
    % Draw a matrix of i.i.d. N(beta_bar, sigma2) random variables
    L = randn(n,5000) * sqrt(sigma2) + beta_bar;
    
    % Convert to a cell array, where each cell contains an n x 1 vector
    % of i.i.d. draws
    L = num2cell(L,1);
    
    % Calculate the weighted normal PDF for all vectors
    L = cellfun(@(b)wnormpdf(b,beta_bar,sigma2),L,'UniformOutput',false);
    
    % Convert the vectors into a n x 500 matrix, take the mean within rows
    % (this gives the choice probabilities), then take the log and sum up
    L = -sum(log(mean(cell2mat(L),2)));
end
end