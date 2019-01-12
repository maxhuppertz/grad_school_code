% Set up a 2 by 2 matrix containing upper and lower bounds for the
% overflow and underflow values, respectively (the first row will be used
% to pin down the overflow, the second to pin down the underflow value)
bounds = ones(2,2);

% Set an adjustment parameter
alpha = 1.2;

% Increase the upper bound on the overflow value until machine precision
% isn't perfect anymore
while log(exp(bounds(1,2))) == bounds(1,2)
    bounds(1,2) = bounds(1,2) * alpha;
end

% Decrease the lower bound on the underflow value until machine precision
% isn't perfect anymore
while log(exp(bounds(2,1))) == bounds(2,1)
    bounds(2,1) = bounds(2,1) / alpha;
end

% Set up a vector of indicators for whether overflow and underflow values
% converged (the first element indicates whether the overflow value has
% converged, and the second element does the same for the underflow value)
conv = zeros(2,1);

% Specify tolerance for bound convergence
tol = 10^(-10);

% As long as at least one bound hasn't converged...
while conv(1,1) * conv(2,1) == 0
    % Go through overflow and underflow bounds
    for i=1:2
        % Check whether the given bound has converged
        if conv(i,1) == 0
            % Calculate midpoint between upper and lower bound
            mp = (bounds(i,2) + bounds(i,1)) / 2;
            
            % Adjust bounds accordingly
            if log(exp(mp)) == mp
                % If the midpoint is not in the overflow or underflow
                % region, use is the new lower bound on the overflow value
                % (i.e. when i=1), or use it as the new upper bound on the
                % underflow value (i.e. when i=2)
                bounds(i,i) = mp;
            else
                % Otherwise, replace the other bound, i.e. the upper bound
                % on the overflow value (i=1) or the lower bound on the
                % underflow value (i=2)
                bounds(i,3-i) = mp;
            end

            % If bounds are within tolerance, change convergence indicator
            if bounds(i,2) - bounds(i,1) <= tol
                conv(i,1) = 1;
            end
        end
    end
end

% Display the results (I could also use a midpoint here, but since the two
% values are within tolerance it really doesn't matter, unless tolerance is
% crazy low)
disp(['Overflow value: ', num2str(bounds(1,1))])
disp(['Underflow value: ', num2str(bounds(2,1))])