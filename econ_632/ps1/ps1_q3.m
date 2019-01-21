% Clear everything
clear

% Set random number generator's seed
rng(632)

% Set number of people and products
n = 5000;
J = 3;

% Set beta
beta = -.2;

% Set mean and variance for the price distribution
mu = 1;
sigma2 = 1;

% Draw prices as lognormal(mu,sigma) i.i.d. random variables
p = lognrnd(mu,sqrt(sigma2),n,J);

% Set up xi, where the jth element of this row vector equals xi_j
xi = [1,2,0] + 10;

% Draw epsilon as Gumbel(0,1) i.i.d. random variables
eps = evrnd(0,1,n,J);

% Construct utility
u = beta*p + ones(n,1)*xi + eps;

% Get vector of chosen goods, using MATLAB's max() function (the 2 makes
% sure it returns the row maximum); the second value it returns is the
% index of the maximum (which is the index of the chosen good, i.e. the
% choice I'm looing for)
[~,c] = max(u,[],2);

% Set initial values for MLE. To be able to identify the xi later, I'll
% normalize xi_J = 0. Therefore, the initial values for xi only include the
% first J-1 elements of the vector
beta0 = 0;
xi0 = zeros(1,J-1);

% Set optimization options
options = optimset('GradObj','on','HessFcn','on','Display','off', ...
    'TolFun',1e-6,'TolX',1e-6); 

% Get MLE estimate of theta = [beta, xi], as well as the Hessian of the log
% likelihood function, which is the same as the (sample) Fisher information
% for the estimator
[theta_hat,~,~,~,~,I] = fminunc( ...
    @(theta)ll_multilogit_fc(theta(1),[theta(2:J),0],p,c,1), ...
    [beta0,xi0],options);

% Get analytic standard errors, based on properties of correctly specified
% MLE (variance is the negative inverse of Fisher information, estimate
% this using sample analogue)
V = inv(I);
SE_a = sqrt(diag(V));

% Specify number of bootstrap iterations
B = 4999;

% Set up matrix of bootstrap estimates
T = zeros(B,J);

% Go through all bootstrap iterations
% Using a pairs bootstrap is obviously pretty inefficient, and a wild
% bootstrap would probably work better. But I've never used that with MLE,
% and I'm not sure whether there are any big caveats I should know about
% when doing that, so I stuck with the simpler (albeit less efficient)
% choice.
for b=1:B
    % Draw bootstrap sample
    i = randi([1,n],n,1);
    
    % Get prices and choices for the bootstrap sample
    pstar = p(i,:);
    cstar = c(i,:);
    
    % Run MLE
    T(b,:) = fminunc( ...
        @(theta)ll_multilogit_fc(theta(1),[theta(2:J),0],pstar,cstar,1),...
        [beta0,xi0],options);
end

% Get the boostrapped standard errors
SE_b = sqrt(sum((T - ones(B,1) * theta_hat).^2) / B);

% Display the results
D = cell(J+1,4);
D(1,:) = {'theta', 'theta_hat', 'SE_a', 'SE_b'};
D(2:J+1,:) = num2cell([[beta, xi(1,1:J-1)]', theta_hat', SE_a, SE_b']);
disp(D)