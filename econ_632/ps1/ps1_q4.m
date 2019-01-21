% Clear everything
clear

% Set random number generator's seed
rng(632)

% Set number of people and products
n = 5000;
J = 3;

% Set mean and variance for price coefficient distribution
mu_beta = -.2;
sigma2_beta = .5;

% Draw price coefficients
beta = randn(n,1) * sqrt(sigma2_beta) + mu_beta;

% Set mean and variance for the price distribution
mu_p = 1;
sigma2_p = 1;

% Draw prices as lognormal(mu_p,sigma_p) i.i.d. random variables
p = lognrnd(mu_p,sqrt(sigma2_p),n,J);

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

% Set initial values for MLE
beta_bar0 = mu_beta * randn();
sigma2_beta0 = sigma2_beta * randn();
xi0 = xi(1,1:J-1) + randn(size(xi(1,1:J-1)));

% Set optimization options
options = optimset('GradObj','off','HessFcn','off','Display','off', ...
    'TolFun',1e-6,'TolX',1e-6); 

% Set tolerance for direct integration
tol = 10^(-14);

% Get the MLE using direct integration
tic
[theta_hat,~,~,~,~,I] = fminunc( ...
    @(theta)ll_multilogit_rc(theta(1),theta(2),[theta(3:J+1),0],p,c, ...
    'direct',tol,[],[]),[beta_bar0,sigma2_beta0,xi0],options);
time = toc;

% Get analytic standard errors, based on properties of correctly specified
% MLE (variance is the negative inverse of Fisher information, estimate
% this using sample analogue)
V = inv(I);
SE_a = sqrt(diag(V));

% Display the results
D = cell(J+2,3);
D(1,:) = {'theta', 'theta_hat', 'SE_a'};
D(2:J+2,:) = num2cell([[mu_beta, sigma2_beta, xi(1,1:J-1)]', ...
    theta_hat', SE_a]);
fprintf('\nDirect integration\n\n')
disp(D)
disp(['Time elapsed: ', num2str(time), ' seconds'])

% Number of draws for Monte Carlo integration
K = 500;  

% Generate Monte Carlo quadrature points as N(0,1) random variables
mcqp = randn(n,K);

% Generate Monte Carlo quadrature weights, which are simply 1/K
mcqw = ones(1,K)/K;

% Get the MLE using Monte Carlo integration
tic
[theta_hat,~,~,~,~,I] = fminunc( ...
    @(theta)ll_multilogit_rc(theta(1),theta(2),[theta(3:J+1),0],p,c, ...
    'points',[],mcqp,mcqw),[beta_bar0,sigma2_beta0,xi0],options);
time = toc;

% Get analytic standard errors, based on properties of correctly specified
% MLE (variance is the negative inverse of Fisher information, estimate
% this using sample analogue)
V = inv(I);
SE_a = sqrt(diag(V));

% Display the results
D = cell(J+2,3);
D(1,:) = {'theta', 'theta_hat', 'SE_a'};
D(2:J+2,:) = num2cell([[mu_beta, sigma2_beta, xi(1,1:J-1)]', ...
    theta_hat', SE_a]);
fprintf('\nMonte Carlo\n')
disp(D)
disp(['Time elapsed: ', num2str(time), ' seconds'])

% Set precision for sparse grids integration
k = 6;

% Get sparse grid quadrature points L and weights w for N(0,1) variable
[sgqp,sgqw] = nwspgr('KPN',1,k);

% Get the MLE using sparse grids (this uses sgqp', since ll_multilogit_rc
% expectes a row vector of quadrature points)
tic
[theta_hat,~,~,~,~,I] = fminunc( ...
    @(theta)ll_multilogit_rc(theta(1),theta(2),[theta(3:J+1),0],p,c, ...
    'points',[],sgqp',sgqw'),[beta_bar0,sigma2_beta0,xi0],options);
time = toc;

% Get analytic standard errors, based on properties of correctly specified
% MLE (variance is the negative inverse of Fisher information, estimate
% this using sample analogue)
V = inv(I);
SE_a = sqrt(diag(V));

% Display the results
D = cell(J+2,3);
D(1,:) = {'theta', 'theta_hat', 'SE_a'};
D(2:J+2,:) = num2cell([[mu_beta, sigma2_beta, xi(1,1:J-1)]', ...
    theta_hat', SE_a]);
fprintf('\nSparse grids\n')
disp(D)
disp(['Time elapsed: ', num2str(time), ' seconds'])