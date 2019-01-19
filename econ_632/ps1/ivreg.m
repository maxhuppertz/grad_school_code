function [beta_hat] = ivreg(y,X,Z)
% Get number of observations
n = size(y,1);

% Get projector matrix of Z
Pz = Z*((Z'*Z)\Z');

X_hat = [ones(n,1), Pz*X];

% Estimate second stage
beta_hat = (X_hat'*X_hat)\(X_hat'*y);
end