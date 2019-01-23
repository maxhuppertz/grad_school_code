% Clear everything
clear

% Set random number generator's seed
rng(632)

% Set number of people n, number of and products per market J, and number
% of markets M
n = 10000;
J = 4;
M = 100;

% Get m = m(i) for each individual
m = ceil((1:n)' * (M/n));

% Set up xi, where the [m,j] element of this vector equals xi_{mj}
mu_xi = [11,12,10,11.5];  % Mean of the quality for each product
sigma2_xi = [1,.5,1.2,.7];  % Variance for each product
xi = randn(M,J) .* (ones(M,1) * sqrt(sigma2_xi)) + ones(M,1) * mu_xi;

% Set up Z, where the [m,j] element of this [M,J] matrix equals Z_{mj}
mu_Z = 0;  % Mean Z
sigma2_Z = 2;  % Variance of Z

% Draw Z as N(mu_Z,sigma2_Z) random variable
Z = randn(M,J)*sqrt(sigma2_Z) + mu_Z;

% Set coefficient for pricing equation (how the instrument affects price)
gamma_Z = 1;

% Get prices, which have an additional disturbance built in
sigma2_p = .2;
p = xi + gamma_Z*Z + randn(M,J)*sqrt(sigma2_p);

% Draw epsilon as Gumbel(0,1) i.i.d. random variables
eps = evrnd(0,1,n,J);

% Set price coefficient for utility function
beta = -.2;

% Construct utility as u_{ij} = beta*p_{ij} + xi_{mj} + eps_{ij}
% The Kronecker product repeats the [1,J] vectors p_j and xi_j exactly n/M
% times for each market, i.e. exactly as many times as there are people in
% the market
u = kron(beta*p + xi,ones(n/M,1)) + eps;

% Get [n,J] indicator matrix of chosen goods, where the [i,J] element is 1
% if individual i chooses good J, and zero otherwise
C = (u == max(u,[],2));

% Get market shares by using accumarray on the choices for each option
S = zeros(M,J);
for i=1:J
    S(:,i) = accumarray(m,C(:,i),[],@mean);
end

% Calculate log shares
lnS = log(S);

% Markets with zero share for any good will cause a problem, since they
% will cause the IV estimation to return NaNs. Select whether to add a tiny
% amount to the share of the good in question for that market
addtozeros = 1;

% Add to the shares of the goods in question, if desired
if addtozeros
    lnS(lnS==-Inf) = 10^(-14);
    
    % Display a message that this happened
    fprintf('\nZero shares set to 10^(-14)\n')
end

% Get IV sample (markets with non-zero shares for all goods)
ivsamp = sum(lnS~=-Inf,2) == J;

% Get Effective number of markets (i.e. those with non-zero shares)
Mivsamp = sum(ivsamp);

% Display effective number of markets
fprintf(['\nEffective number of markets: ',num2str(Mivsamp),'\n'])

% Set up cell array to display results
D = cell(3,4);
D(1,:) = {'Parameter', 'True value', 'Estimate', 'SE_a'};
D(2,1) = {'Intercept'};
D(2,2) = {'n/a'};
D(3,1) = {'beta'};
D(3,2) = num2cell(beta);

% Created vector of log share differences (option J is the outside good)
DlnSflat = ...
    reshape(lnS(ivsamp,1:J-1),[Mivsamp*(J-1),1]) ...
    - reshape(lnS(ivsamp,J)*ones(1,J-1),[Mivsamp*(J-1),1]);

% Create flattened version of price and instrument data
pflat = reshape(p(ivsamp,1:J-1),[Mivsamp*(J-1),1]);
Zflat = reshape(Z(ivsamp,1:J-1),[Mivsamp*(J-1),1]);

% Run 2SLS on the flattened (pooled) data, including an intercept
[theta_hat,Sigma_hat] = ivreg(DlnSflat, [ones((J-1)*Mivsamp,1),pflat], ...
    [ones((J-1)*Mivsamp,1),Zflat]);

% Calculate standard errors
SE_a = sqrt(diag(Sigma_hat));

% Display the results
D(2:3,3:4) = num2cell([theta_hat, SE_a]);
fprintf('\n2SLS estimates\n')
disp(D)