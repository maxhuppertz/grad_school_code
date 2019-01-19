% Clear everything
clear

% Set random number generator's seed
rng(632)

% Set number of people n, number of and products per market J, and number
% of markets M
n = 10000;
J = 3;
M = 100;

% Get m = m(i) for each individual
m = ceil((1:n)' * (M/n));

% Set up xi, where the [m,j] element of this matrix equals xi_{mj}
mu_xi = [1,2,0];  % Means of the quality distribution for each alternative
sigma_xi = 1;  % Scale of the quality distribution

% Draw xi as Gumbel(0,sigma_xi) + mu_xi, use kron() to repeat that vector
% as many times as needed for each market
xi = kron(evrnd(0,sigma_xi,M,J) + mu_xi,ones(n/M,1));
return
% Set mean and variance for price coefficient distribution across markets
mu_beta = -.2;
sigma2_beta = .5;

% Set up Z, where the mth element of this column vector equals Z_m


% Draw price coefficients
beta = randn(n,1) * sqrt(sigma2_beta) + mu_beta;

% Set mean and variance for the price distribution
mu_p = 10;
sigma2_p = 10;

% Draw prices as N(10,10) i.i.d. random variables
p = randn(n,J) * sqrt(sigma2_p) + mu_p;

% Draw epsilon as Gumbel(0,1) i.i.d. random variables
eps = evrnd(0,1,n,J);

% Construct utility
u = beta.*p + ones(n,1)*xi + eps;

% Get vector of chosen goods, using MATLAB's max() function (the 2 makes
% sure it returns the row maximum); the second value it returns is the
% index of the maximum (which is the index of the chosen good, i.e. the
% choice I'm looing for)
[~,c] = max(u,[],2);