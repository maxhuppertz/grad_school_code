function [beta_hat,Sigma_hat,eps_hat] = ivreg(y,X,Z)
% Estimates two stage least squares instrumental variables
%
% Inputs
% y: [n,1] vector, outcome variable for the second stage
% X: [n,k] matrix, contains all endogenous variables, as well as all
%                  exogenous variables which aren't used as instruments in
%                  the first stage
% Z: [n,k] matrix, contains all instruments, as well as all exogenous
%                  variables which aren't used as instruments in the first
%                  stage
%
% Outputs
% beta_hat: [k,1] vector, estimated coefficients from the second stage
% Sigma_hat: [k,k] matrix, variance/covariance matrix for the coefficients
%
% Note
% Endogenous second stage regressors and their instruments need to appear
% in the same column in X and Z. The same goes for exogenous regressors not
% used as instruments. Suppose I want to estimate the model
%
% y = beta0 + X1*beta1 + X2beta2 + X3beta3 + epsilon
%
% where I assume that X3 is exogenous, but X1 and X2 are endogenous (i.e.
% correlated with epsilon). To solve this issue, I use an instrument Z1 for
% X1, and another instrument Z2 for X2. To estimate beta0, I just use an
% intercept, i.e. an nx1 vector of ones. To use this routine to estimate
% the model, I would need to specify X and Z as
%
% X = [ones(n,1), X1, X2, X3]
% Z = [ones(n,1), Z1, Z2, X3]

% Get number of observations and number of coefficients
[n,k] = size(Z);

% Calculate projector matrix of Z
Pz = (Z/(Z'*Z))*Z';

% Calculate first stage predicted values
X_hat = Pz*X;

% Calcuate (X_hat'X_hat)^(-1) (useful because it's necessary for the
% coefficient estimate, and it also provides the bread for sandwich
% variance estimator below
XXinv = (X_hat'*X_hat) \ eye(k);

% Estimate second stage coefficient
beta_hat = (X_hat'*X_hat)\(X_hat'*y);

% Get second stage residuals (it's important to use X, not X_hat)
eps_hat = y - X*beta_hat;

% Calculate rescaled residuals (useful for variance/covariance estimator)
u = (eps_hat*ones(1,k)).*X_hat;

% Get heteroskedasticity-robust variance/covariance estimator
V = u'*u;  % Filling for the sandwich
Sigma_hat = (n/(n-k)) * XXinv * V * XXinv;  % Putting the sandwich together
end