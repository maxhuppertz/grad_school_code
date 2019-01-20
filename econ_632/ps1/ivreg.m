function [beta_hat,Sigma_hat,eps_hat] = ivreg(y,X,Z)
[n,k] = size(Z);

% Get projector matrix of Z
Pz = (Z/(Z'*Z))*Z';

% Get predicted values
X_hat = Pz*X;

% Estimate second stage
beta_hat = (X_hat'*X_hat)\(X_hat'*y);

% Get residuals
eps_hat = y - X*beta_hat;

% Get rescaled residuals (useful for variance/covariance estimator)
u = eps_hat.*X_hat;

% Get variance/covariance estimator
H = ((X'*Z)/(Z'*Z))*(Z'*X);  % Bread for the sandwich
V = u'*u;  % Filling for the sandwich
Sigma_hat = (n/(n-k)) * H\V/H;  % Putting the sandwich together
end