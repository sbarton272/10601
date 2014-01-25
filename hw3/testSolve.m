function [score] = testSolve()
    score = 0;
    totalPoints = 2;
    
    check = all(solve([1 3; 2 4],[5;6]) == [-1;2]);
    feedback(1, check);
    score = score + check;

    check = all(solve([1 3 0; 2 -1 1; -2 6 2], [-1;-1;4]) == [-1;0;1]);
    feedback(2, check);
    score = score + check;

    fprintf('Total score: %i/%i', score, totalPoints);
    score = score / totalPoints;
end
