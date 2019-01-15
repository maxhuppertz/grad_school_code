function A = loop_accumarray(subs,val,sz,fun)

% Figure out the size of the subscript array
[n,m] = size(subs);

% Abort if this tries to create anything with more than 2 dimensions
if m>2
    disp('Dimensionality error: subs has more than two columns, which')
    disp('suggests you are trying to create an output array with more')
    disp('than two dimensions. This function cannot handle objects of')
    disp('that dimensionality.')
    
    % Set output array to NaM
    A = NaN;
    
    % Abort
    return
end

% If subs is a vector, make it into a matrix where the second column is
% just a column of ones
if m == 1
    subs = [subs ones(n,1)];
end

% Set up output array as zeros
A = zeros(max(max(subs(:,1),sz(1))), max(max(subs(:,2)),sz(2)));

% Go through all subscripts
for i=1:n
    % Add corresponding input array value to output array value
    A(subs(i,1), subs(i,2)) = ...
            fun([A(subs(i,1), subs(i,2)), val(i)]);
end
end