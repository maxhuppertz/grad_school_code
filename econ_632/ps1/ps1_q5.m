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

% Draw xi as Gumbel(0,sigma_xi) + mu_xi
xi = evrnd(0,sigma_xi,M,J) + mu_xi;

% Set mean and variance for price coefficient distribution across markets
mu_beta = -.2;
sigma2_beta = .5;

% Set up Z, where the mth element of this column vector equals Z_m
mu_Z = .3;  % Mean of base distribution for Z
sigma_Z = .15;  % Standard deviation of base distribution for Z

% Draw Z as lognormal random variable
Z = lognrnd(mu_Z,sigma_Z,M,J);

% Rescale Z to make sure it's below 1 (think e.g. tax rates)
Z = Z./(1.2*ones(M,1)*max(Z,[],1));

% Set coefficient for pricing equation
gamma_Z = .1;

% Get prices
p = xi + gamma_Z*Z;

% Draw epsilon as Gumbel(0,1) i.i.d. random variables
eps = evrnd(0,1,n,J);

% Set price coefficient for utility function
beta = -.2;

% Construct utility as u_{ij} = beta*p_{ij} + xi_{mj} + eps_{ij}
% The Kronecker product repeats the [1,J] vectors p_j and xi_j exactly n/M
% times for each market, i.e. exactly as many times as there are people in
% the market
u = beta*kron(p+xi,ones(n/M,1)) + eps;

% Get [n,J] indicator matrix of chosen goods, where the [i,J] element is 1
% if individual i chooses good J, and zero otherwise
C = (u == max(u,[],2));

% Get market shares using accumarray on each option
S = zeros(M,J);
for i=1:J
    S(:,i) = accumarray(m,C(:,i),[],@mean);
end

disp([(1:M)',S,p,Z])