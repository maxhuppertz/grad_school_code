function [beta_hat] = ivreg(y,X,Z,cons)
% Get number of observations
n = size(y,1);

% Get projector matrix of Z
Pz = Z*((Z'*Z)\Z');

% Get predicted values
X_hat = Pz*X;

% Check whether to include a constant in the second stage
if cons
    X_hat = [ones(n,1), X_hat];
end

% Estimate second stage
beta_hat = (X_hat'*X_hat)\(X_hat'*y);
end