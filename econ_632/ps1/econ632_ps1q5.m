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
mu_xi = [1,2,0,1.5];  % Mean of the quality for each product
sigma2_xi = [.5,.2,.7,1];  % Variance for each product
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
addtozeros = 0;

% Add to the shares of the goods in question, if desired
if addtozeros == 1
    % Add to the shares
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

% Created vector of log share differences (option J is the outside good)
DlnSflat = ...
    reshape(lnS(ivsamp,1:J-1),[Mivsamp*(J-1),1]) ...
    - reshape(lnS(ivsamp,J)*ones(1,J-1),[Mivsamp*(J-1),1]);

% Create flattened version of price and instrument data
pflat = reshape(p(ivsamp,1:J-1),[Mivsamp*(J-1),1]);
Zflat = reshape(Z(ivsamp,1:J-1),[Mivsamp*(J-1),1]);

% Make a flattened version of the product identifier
pidflat = kron(ones(Mivsamp,1),(1:J-1)');

% Create a dummy version of this
DP = zeros(Mivsamp*(J-1),J-1);

% Go through all products but the outside option
for i=1:J-1
    % Replace the dummy in question
    DP(:,i) = (pidflat == i);
end

% Specify whether to use simple intercept. Otherwise, uses a simple
% intercept, plus a dummy for each option other than the outside one and
% the second to last one
simpleicept = 0;

% Implement the chosen intercept
if simpleicept == 1
    icept = ones(Mivsamp*(J-1),1);
else
    icept = DP(:,1:J-1);
end

% Run 2SLS on the flattened (pooled) data, including the intercept
[theta_hat,Sigma_hat,xi_hat] = ivreg(DlnSflat,[icept,pflat],[icept,Zflat]);

% Calculate standard errors
SE_a = sqrt(diag(Sigma_hat));

% Set up cell array to display results
D1 = cell(3,4);
D1(1,:) = {'Parameter', 'True value', 'Estimate', 'SE_a'};
D1(length(theta_hat)+1,1) = {'beta'};
D1(length(theta_hat)+1,2) = num2cell(beta);

% Fill in estimate labels for the parts of the intercept
for i=1:length(theta_hat)-1
    D1(i+1,1) = {strcat('D',num2str(i))};
    D1(i+1,2) = {'n/a'};
end

% Display the results
D1(2:length(theta_hat)+1,3:4) = num2cell([theta_hat, SE_a]);
fprintf('\n2SLS estimates\n')
disp(D1)
xi_hat_orig = mean(reshape(xi_hat,[Mivsamp,J-1]),1);

% Select number of bootstrap iterations
B = 4999;

% Set up vector of bootstrap estimates
T = zeros(B,J-1);

% Go through all bootstrap samples
parfor b=1:B
    % Set random number generator seed (necessary if this is run in
    % parallel)
    rng(b)
    
    % Draw index for bootstrap sample (note this is at the market level)
    i = randi([1,Mivsamp],Mivsamp,1);
    
    % Convert to an index to select from flattened arrays. Get the
    % initially chosen market, but the corresponding next two rows in the
    % flattened arrays
    add = kron(ones(Mivsamp,1),(0:2)');  % Gets next to rows
    I = kron(i,ones(J-1,1)) + add;
    
    % Estimate on the bootstrap sample
    [~,~,xi_hat] = ...
        ivreg(DlnSflat(I,:),[icept(I,:),pflat],[icept(I,:),Zflat]);
    
    % Add mean of estimates to the vector of bootstrap estimates
    T(b,:) = mean(reshape(xi_hat,[Mivsamp,J-1]),1);
end

% Set up array to display results
D2 = cell(4,4);
D2(1,1) = {'Parameter'};
D2(1,2) = {'True value'};
D2(1,3) = {'Estimate'};
D2(1,4) = {'SE_b'};

% Add rows for each xi_hat
for i=1:J-1
    D2(i+1,1) = {strcat('xi_hat ',num2str(i))};
    D2(i+1,2) = {mu_xi(i)};
end

% Add estimates and bootstrapped standard errors
D2(2:J,3:4) = num2cell([xi_hat_orig', std(T,1)']);

% Display the results
disp('Recovered mean quality')
disp(D2)