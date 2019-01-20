function [beta_hat,Sigma_hat] = ivreg(y,X,Z)
% Get number of observationsn and coefficients k
[n,k] = size(Z);

% Get projector matrix of Z
Pz = Z*((Z'*Z)\Z');

% Get predicted values
X_hat = Pz*X;

% Estimate second stage
beta_hat = (X_hat'*X_hat)\(X_hat'*y);

% Get residuals
eps_hat = y - X*beta_hat;

Qzz_hat = (Z'*Z)/n;
Qxz_hat = (X'*Z)/n;
Qzx_hat = (Z'*X)/n;
Ze = Z.*(eps_hat*ones(1,k));
Omega_hat = (Ze'*Ze)/n;

Sigma_hat = (Qxz_hat*(Qzz_hat\Qzx_hat))\ ...
    ((Qxz_hat\Qzz_hat)*Omega_hat*(Qzz_hat\Qzx_hat)) ...
    /(Qxz_hat*(Qzz_hat\Qzx_hat));
end