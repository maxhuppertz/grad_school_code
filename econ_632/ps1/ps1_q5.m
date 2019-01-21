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
mu_xi = [0,1,2];
xi = ones(M,1) * mu_xi;

% Set up Z, where the mth element of this column vector equals Z_m
mu_Z = 0;  % Mean of base distribution for Z
sigma2_Z = 2;  % Variance of base distribution for Z

% Draw Z as lognormal random variable
Z = randn(M,J) * sigma2_Z + mu_Z;

% Set coefficient for pricing equation (how the instrument affects price)
gamma_Z = .25;

% Get prices as quality plus price times price coefficient plus disturbance
sigma2_p = 2;  % Variance for additional price disturbances
p = xi + gamma_Z*Z;

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

% Set them to a very small number if the computer evaluates them as
% negative infinity
lnS(lnS==-Inf) = 10^(-14);

% Set up cell array to display results
D = cell(J+1,4);
D(1,:) = {'', 'True value', 'Estimate', 'SE_a'};
D(2,1) = {'xi_1'};
D(J+1,1) = {'beta'};

% Created vector of log share differences (option J is the outside good)
DlnSflat = ...
    reshape(lnS(:,1:J-1),[M*(J-1),1]) ...
    - reshape(lnS(:,J)*ones(1,J-1),[M*(J-1),1]);

% Create dummies for all options other than J
Dxi = zeros((J-1)*M,J-2);  % Matrix of dummies

% Go through all such options
for j=1:J-1
    % Check whether this is the first option
    if j==1
        % If it is, make an intercept
        d = ones(J-1,1);
    else
        % Otherwise, start with a vector of zeros
        d = zeros(J-1,1);

        % Make the jth element equal to one
        d(j) = 1;
    end
    
    % Put it in the matrix
    Dxi(:,j) = repmat(d,M,1);
    
    % Add a label to the results matrix
    D(j+1,1) = cellstr(strcat('xi_',num2str(j)));
end

% Create flattened version of price and instrument data
pflat = reshape(p(:,1:J-1),[M*(J-1),1]);
Zflat = reshape(Z(:,1:J-1),[M*(J-1),1]);

% Run 2SLS on the flattened (pooled) data
[theta_hat,Sigma_hat] = ivreg(DlnSflat, [Dxi,pflat], ...
    [Dxi,Zflat]);

% Calculate standard errors
SE_a = sqrt(diag(Sigma_hat));

% Display the results
D(2:J+1,2:4) = num2cell([[mu_xi(1:J-1), beta]', theta_hat, SE_a]);
fprintf('\n2SLS estimates\n')
disp(D)