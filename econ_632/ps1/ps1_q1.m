% Finds the highest and lowest values x such that log(exp(x)) = x; denote
% these as H and L, respectively

% Set up a 2 by 2 matrix containing upper and lower bounds for H and L,
% respectively (the first row will be used to pin down H, the second to pin
% down L)
bounds = ones(2,2);
bounds(2,1) = -bounds(2,1);

% Set an adjustment parameter (used only to adjust the initial upper bound
% on the overflow and lower bound on the underflow value)
alpha = 1.2;

% Go through both bounds
for i=1:2
    % While the upper bound on H (i=1) and the lower bound on L (i=2) are
    % inside of the machine precision range, keep increasing / decreasing
    % them
    while log(exp(bounds(i,3-i))) == bounds(i,3-i)
        bounds(i,3-i) = bounds(i,3-i) * alpha;
    end
end

% Set up a vector of indicators for whether H and L converged (the first
% element indicates whether H has converged, and the second element does
% the same for L)
conv = zeros(2,1);

% Specify tolerance for convergence
tol = 10^(-12);

% As long as at least one value hasn't converged...
while conv(1,1) * conv(2,1) == 0
    % Go through bounds on H and L
    for i=1:2
        % Check whether the given bound has converged
        if conv(i,1) == 0
            % Calculate midpoint between upper and lower bound
            mp = (bounds(i,2) + bounds(i,1)) / 2;
            
            % Adjust bounds accordingly
            if log(exp(mp)) == mp
                % If the midpoint does not suffer from underflow, use it as
                % the new lower bound on H (i=1), or use it as the new
                % upper bound on L (i=2)
                bounds(i,i) = mp;
            else
                % Otherwise, replace the other bound, i.e. the upper bound
                % on H (i=1) or the lower bound on L (i=2)
                bounds(i,3-i) = mp;
            end
            
            % If bounds are within tolerance, change convergence indicator
            if (bounds(i,2) - bounds(i,1)) / tol <= 1
                conv(i,1) = 1;
            end
        end
    end
end

% Display the results (I could also use a midpoint here, but since the two
% values are within tolerance it really doesn't matter, unless tolerance is
% crazy low)
fprintf('\nResults of machine precision check:\n')
disp(['H: ', num2str(bounds(1,1))])
disp(['L: ', num2str(bounds(2,1))])

% Test out the underflow safe logit inclusive value function
% Set up an X ~ chi2(0,1) vector or random variables
X = chi2rnd(5,1000,1);

% Display results
fprintf('\nUnderflow check, chi2(5) random variable:\n')
disp(['Naive LIV: ', num2str(log(sum(exp(X))))])
disp(['Underflow safe LIV: ', num2str(logsumexp_safe(X))])

% Multiply the vector by a large positive number
Xhi = X * exp(600);

% Display results
fprintf('\nUnderflow check, chi2(5) * exp(600) random variable:\n')
disp(['Naive LIV: ', num2str(log(sum(exp(Xhi))))])
disp(['Underflow safe LIV: ', num2str(logsumexp_safe(Xhi))])

% Multiply the vector by a very large negative number
Xlo = -Xhi;

% Display the results
fprintf('\nUnderflow check, chi2(5) * [-exp(600)] random variable:\n')
disp(['Naive LIV: ', num2str(log(sum(exp(Xlo))))])
disp(['Underflow safe LIV: ', num2str(logsumexp_safe(Xlo))])