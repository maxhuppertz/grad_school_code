function A = loop_accumarray(subs,val,sz,fun,fillval,issparse)
% Figure out the size of the subscript array
[n,m] = size(subs);

% Abort if this tries to create anything with more than 2 dimensions
if m>2
    disp('Dimensionality error: subs has more than two columns, which')
    disp('suggests you are trying to create an output array with more')
    disp('than two dimensions. This function cannot handle objects of')
    disp('that dimensionality.')
    
    % Set output array to NaN
    A = NaN;
    
    % Abort
    return
end

% If subs is a vector, make it into a matrix where the second column is
% just a column of ones
if m == 1
    subs = [subs ones(n,1)];
end

% Set up output array B as zeros, and set up an intermediate cell array A
% which will collect elements of val as a vector so they can be accumulated
% using fun
if ~(size(sz,1)==0 && size(sz,2) ==0)
    A = cell(max(max(subs(:,1),sz(1))), max(max(subs(:,2)),sz(2)));
else
    A = cell(max(subs(:,1)), max(subs(:,2)));
end

% Go through all subscripts, add elements of val to the corresponding cell
% in A, which collects them as a vector
for i=1:n
    A{subs(i,1), subs(i,2)} = [A{subs(i,1), subs(i,2)}, val(i)];
end

% Apply function to all elements of A
A = cellfun(fun,A);

% Fill in NaN values as fillval
A(isnan(A)) = fillval;

% Make B sparse if desired
if issparse == 1
    A = sparse(A);
end
end