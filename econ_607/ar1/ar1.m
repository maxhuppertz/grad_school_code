function y = ar1(a, r, s2, T, y0)
    % This function generates a time series of length T taken from an AR(1)
    % process with initial value y0, offset a, persitence parameter r, and
    % N(0, s2) innovations
    
    % Set up y matrix and innovations
    y = ones(T,1)*y0;
    e = randn(T,1)*sqrt(s2);
    
    % Generate AR(1) data
    for t=2:T
        y(t,1) = a + r*y(t-1,1) + e(t,1);
    end