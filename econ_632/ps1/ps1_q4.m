% Clear everything
clear

% Set random number generator's seed
rng(632)

% Set number of people and products
n = 5000;
J = 3;

% Set mean and variance for price coefficient distribution
mu_beta = -.2;
sigma2_beta = 1;

% Draw price coefficients
beta = randn(n,1) * sqrt(sigma2_beta) + mu_beta;

% Set mean and variance for the price distribution
mu_p = 10;
sigma2_p = 10;

% Draw prices as N(10,10) i.i.d. random variables
p = randn(n,J) * sqrt(sigma2_p) + mu_p;

% Set up xi, where the jth element of this row vector equals xi_j
xi = [1,2,0];

% Draw epsilon as Gumbel(0,1) i.i.d. random variables
eps = evrnd(0,1,n,J);

% Construct utility
u = beta.*p + ones(n,1)*xi + eps;

% Get vector of chosen goods, using MATLAB's max() function (the 2 makes
% sure it returns the row maximum); the second value it returns is the
% index of the maximum (which is the index of the chosen good, i.e. the
% choice I'm looing for)
[~,c] = max(u,[],2);

beta_bar0 = 0;
sigma2_beta0 = 1.2;
xi0 = xi(1,1:J-1) + randn(size(xi(1,1:J-1)));

% Set optimization options
options = optimset('GradObj','off','HessFcn','off','Display','off', ...
    'TolFun',1e-6,'TolX',1e-6); 

[theta_hat,~,~,~,~,I] = fminunc( ...
    @(theta)ll_multilogit_rc(theta(1),theta(2),[theta(3:J+1),0],p,c,'integral'), ...
    [beta_bar0,sigma2_beta0,xi0],options);

% Get analytic standard errors, based on properties of correctly specified
% MLE (variance is the negative inverse of Fisher information, estimate
% this using sample analogue)
V = inv(I);
SE_a = sqrt(diag(V));