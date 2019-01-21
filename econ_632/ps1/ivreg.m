function [beta_hat,Sigma_hat] = ivreg(y,X,Z)
% Get number of observations and number of coefficients
[n,k] = size(Z);

% Get projector matrix of Z
Pz = (Z/(Z'*Z))*Z';

% Get first stage predicted values
X_hat = Pz*X;

% Estimate second stage
beta_hat = (X_hat'*X_hat)\(X_hat'*y);

% Get second stage residuals
eps_hat = y - X*beta_hat;

% Calculate rescaled residuals (useful for variance/covariance estimator)
u = (eps_hat*ones(1,k)).*X_hat;

% Get variance/covariance estimator
H = (X'*Pz*X);  % Bread for the sandwich
V = u'*u;  % Filling for the sandwich
Sigma_hat = (n/(n-k)) * H\V/H;  % Putting the sandwich together
end