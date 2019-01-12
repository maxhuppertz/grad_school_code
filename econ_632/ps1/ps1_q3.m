% Set random number generator's seed
rng(632)

% Set number of people and products
n = 5000;
J = 3;

% Set beta
beta = .2;

% Set mean and variance for the price distribution
mu = 10;
sigma = 10;

% Draw prices as N(10,10) i.i.d. random variables
p = randn(n,J) * sqrt(sigma) + 10;

% Set up xi, where the jth element of this row vector equals xi_j
xi = [1,1.5,.8];

% Draw epsilon as Gumbel(0,1) i.i.d. random variables
eps = evrnd(0,1,n,J);

% Construct utility
u = beta*p + ones(n,1)*xi + eps;

% Get vector of chosen goods, using the fact that MATLAB's max() function
% returns the row maximum when applied to a matrix, and returns its index
% as a second output argument (this index being the choice I'm looking for)
[~,c] = max(u,[],2);