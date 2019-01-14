% Clear everything
clear

% Set random number generator's seed
rng(632)

% Set number of people and products
n = 500000;
J = 3;

% Set beta
beta = -.2;

% Set mean and variance for the price distribution
mu = 10;
sigma = 10;

% Draw prices as N(10,10) i.i.d. random variables
p = randn(n,J) * sqrt(sigma) + 10;

% Set up xi, where the jth element of this row vector equals xi_j
xi = [-1,2,0];

% Draw epsilon as Gumbel(0,1) i.i.d. random variables
eps = evrnd(0,1,n,J);

% Construct utility
u = beta*p + ones(n,1)*xi + eps;

% Get vector of chosen goods, using MATLAB's max() function (the 2 makes
% sure it returns the row maximum); the second value it returns is the
% index of the maximum (which is the index of the chosen good, i.e. the
% choice I'm looing for)
[~,c] = max(u,[],2);

% Set initial values
beta0 = beta + randn();
xi0 = xi + randn(size(xi));

% Set optimization options
options = optimset('GradObj','on','TolFun',1e-10,'TolX',1e-10); 
[theta_hat,~,~,~,Gradient,Hessian] = fminunc(@(theta)ll_multilogit_fc(theta(1),theta(2:4),p,c),[beta0,xi0],options);
disp(theta_hat)