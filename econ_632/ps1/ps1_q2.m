% Clear everything
clear

% Set up a vector of subscripts and a vector of values
subs = [1,2; 1,2; 4,1; 5,1; 5,2];
vals = (101:105)';

% Compare results of loop-based accumarray and built-in function; the inner
% all makes sure that within each column, the elements are the same, and
% the outer all combines them across columns
fagree = all(all(loop_accumarray(subs,vals,@max) ...
    == accumarray(subs,vals,[],@max)));

% Display whether the two agree
fprintf('\nResult of function agreement check:\n')
if fagree == 1
    % If they agree, display that
    disp('Loop-based and built-in accumarray functions agree')
else
    % If they don't, display that
    disp('Loop-based and built-in accumarray functions do not agree')
end