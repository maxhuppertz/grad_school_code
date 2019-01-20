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
mu_xi = [11,12,10];
xi = ones(M,1) * mu_xi;

% Set up Z, where the mth element of this column vector equals Z_m
mu_Z = 0;  % Mean of base distribution for Z
sigma2_Z = 1;  % Variance of base distribution for Z

% Draw Z as lognormal random variable
Z = randn(M,J) * sigma2_Z + mu_Z;

% Set coefficient for pricing equation
gamma_Z = 2;

% Get prices as quality plus price times price coefficient plus disturbance
sigma2_p = 10;  % Variance for additional price disturbances
p = xi + gamma_Z*Z + randn(M,1)*sqrt(sigma2_p);

% Draw epsilon as Gumbel(0,1) i.i.d. random variables
eps = evrnd(0,1,n,J);

% Set price coefficient for utility function
beta = -.5;

% Construct utility as u_{ij} = beta*p_{ij} + xi_{mj} + eps_{ij}
% The Kronecker product repeats the [1,J] vectors p_j and xi_j exactly n/M
% times for each market, i.e. exactly as many times as there are people in
% the market
u = kron(p*beta+xi,ones(n/M,1)) + eps;

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

% Set them to a very small number if the computer evaluates them as
% negative infinity
lnS(lnS==-Inf) = 10^(-14);

% Set up matrices to store parameter estimates and standard errors
theta_hat = zeros(J-1,2);
SE_a = zeros(J-1,2);

% Set up cell array which will be used to display results
D = cell((J-1)*2+1,3);

% Go through all goods but the last, which is the outside good
for j=1:J-1
    % Run 2SLS estimation on the different in log shares, store estimated
    % coefficients
    [theta_hat(j,:),V_hat,~,~] = ivreg(lnS(:,j)-lnS(:,J), ...
        [ones(M,1),p(:,j)],[ones(M,1),Z(:,j)]);
    
    % Store standard errors
    SE_a(j,:) = sqrt(diag(V_hat));
    
    % Store labels for results
    D(j + 1,1) = cellstr(strcat('xi',{' '},num2str(j)));
    D(j + J,1) = cellstr(strcat('beta',{' '},num2str(j)));
end

% Display the results
D(1,:) = {'Parameter', 'Estimate', 'SE'};
D(2:(J-1)*2+1,2:3) = num2cell(reshape([theta_hat, SE_a],[4,2]));
disp(D)