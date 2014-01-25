function [score] = meanTest()
    score = 0;
    totalPoints = 3;
    
    % simple test
    check = epsilonAway(myMean([0 1 2 3]), 1.5);
    feedback(1, check);
    score = score + check;

    check = epsilonAway(myMean([-2 -1 0 1]), -.5);
    feedback(2, check);
    score = score + check;

    % this might take awhile
    check = epsilonAway(myMean(2:100000000), 50000001);
    feedback(3, check);
    score = score + check;

    fprintf('Total score: %i/%i', score, totalPoints);
    score = score / totalPoints;
end
