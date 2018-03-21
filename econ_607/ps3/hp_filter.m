function [y_dt, T] = hp_filter(y, mu)
    % This detrends a time series using the Hodrick-Prescott filter and
    % returns both the estimated trend and detrended series
    
    % Make sure y is a column vector
    szy = size(y);
    if szy(1) ~= max(size(y))
        y = y.';
    end
    
    % Setting up the HP filter
    % Create the coefficient matrix (based on first order conditions for
    % the HP filter) needed to compute the filtering values; except for the
    % first two and last two rows, it's basically just the shifted vector
    % [1 -4 6 -4 1] * mu + [0 0 1 0 0], although the mu bit and the other
    % added vector are going to follow later on
    M = zeros(length(y), length(y)) ...
        + diag(ones(length(y) - 2, 1), -2) ...
        + diag(ones(length(y) - 2, 1), 2) ...
        + diag(ones(length(y) - 1, 1) * -4, -1) + ...
        + diag(ones(length(y) - 1, 1) * -4, 1)...
        + diag(ones(length(y), 1) * 6);
    
    % For the first two and last two rows, manually replace some elements
    % of the matrix
    for it = 1:2
        for jt = 1:2
            if it ~= jt
                M(it,jt) = -2;
                M(length(y) + 1 - it, length(y) + 1 - jt) = -2;
            elseif it == 1
                M(it,it) = 1;
                M(length(y), length(y)) = 1;
            else
                M(it,it) = 5;
                M(length(y) - 1, length(y) - 1) = 5;
            end
        end
    end
    
    % Mutiply by mu and add the plus one on the main diagonal
    M = M * mu + eye(length(y));
    
    % Calculate HP filter trend
    T = (M \ y);
    
    % Generate detrended series
    y_dt = y - T;