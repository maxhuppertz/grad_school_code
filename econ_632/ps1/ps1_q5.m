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

% Set up xi, where the [m,j] element of this vector equals xi_{mj} = xi_j
%mu_xi = [10,20,15];  % Means of the quality distribution for each alternative
%sigma2_xi = 10;  % Variance of the quality distribution
mu_xi = [20,10,15];
xi = ones(M,1) * mu_xi;

% Draw xi as N(mu_xi,sigma_xi)
%xi = randn(M,J) * sqrt(sigma2_xi) + ones(M,1)*mu_xi;

% Set mean and variance for price coefficient distribution across markets
mu_beta = -.2;
sigma2_beta = .5;

% Set up Z, where the mth element of this column vector equals Z_m
mu_Z = .3;  % Mean of base distribution for Z
sigma_Z = .15;  % Standard deviation of base distribution for Z

% Draw Z as lognormal random variable
Z = lognrnd(mu_Z,sigma_Z,M,J);

% Rescale Z to make sure it's below 1 (think e.g. tax rates)
Z = ( Z./(1.2*ones(M,1)*max(Z,[],1)) ) * 100;

% Set coefficient for pricing equation
gamma_Z = .5;

% Get prices as quality plus price times price coefficient plus disturbance
p = xi + gamma_Z*Z + randn(M,J)*sqrt(10);

% Draw epsilon as Gumbel(0,1) i.i.d. random variables
eps = evrnd(0,1,n,J);

% Set price coefficient for utility function
beta = -.2;

% Construct utility as u_{ij} = beta*p_{ij} + xi_{mj} + eps_{ij}
% The Kronecker product repeats the [1,J] vectors p_j and xi_j exactly n/M
% times for each market, i.e. exactly as many times as there are people in
% the market
u = kron(beta*p+xi,ones(n/M,1)) + eps;

% Get [n,J] indicator matrix of chosen goods, where the [i,J] element is 1
% if individual i chooses good J, and zero otherwise
C = (u == max(u,[],2));

% Get market shares using accumarray on each option
S = zeros(M,J);
for i=1:J
    S(:,i) = accumarray(m,C(:,i),[],@mean);
end

% Set optimization options
options = optimset('GradObj','off','HessFcn','off','Display','off', ...
    'TolFun',1e-6,'TolX',1e-6); 

delta_hat = zeros(M,J);
for i=1:M
   delta_hat(i,:) = ...
       fminunc(@(delta)nls_shares(S(i,:),delta),[0,0,0],options); 
end

theta_hat = ivreg(delta_hat(:,1),p(:,1),Z(:,1),1);
disp(theta_hat)