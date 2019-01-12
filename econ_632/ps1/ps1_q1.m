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
disp(['H: ', num2str(bounds(1,1))])
disp(['L: ', num2str(bounds(2,1))])