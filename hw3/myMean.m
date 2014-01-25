function [a] = myMean(v)

    a = mean(v);

end

function [y] = meanCenter(v)
       
    y = v - mean(v);

end