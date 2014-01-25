function [z] = rescale(X)
    % also known as z score
    if isempty(X)
        z = X;
    else
        stdev = std(X);
        if stdev == 0
            z = zeros(size(X));
        else    
            z = (X - mean(X)) / std(X);
        end
    end
end