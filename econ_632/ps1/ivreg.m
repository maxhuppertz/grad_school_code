function [beta_hat] = ivreg(y,X,Z)
% Get projector matrix of Z
Pz = Z*((Z'*Z)\Z');

% Estimate second stage
beta_hat = (X'*Pz*X)\(X'*P*y);
end