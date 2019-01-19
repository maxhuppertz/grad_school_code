function [beta_hat] = ivreg(y,X,Z)
% Get projector matrix of Z
Pz = Z*inv(Z'Z)*Z';

% Estimate first stage
X_hat = Pz*X;

% Estimate second stage
beta_hat = inv(X'*Pz*X)*(X'*P*y);
end