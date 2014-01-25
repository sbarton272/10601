function [score] = testRescale()
    score = 0;
    totalPoints = 3;
    precision = 0.001;
    
    check = ( norm(rescale([1 2 3 4 5]) - [-1.264911064067352  -0.632455532033676                   0   0.632455532033676   1.264911064067352
] ) <  precision);
    feedback(1, check);
    score = score + check;

    check = ( norm(rescale([-1 -2 5 2 2]) - [ -0.792824967172092  -1.153199952250316   1.369424943297249   0.288299988062579   0.288299988062579] ) <  precision);
    feedback(1, check);
    score = score + check;
    
    check = ( norm(rescale([2 2 2 2 2]) - [0 0 0 0 0] ) <  precision);
    feedback(1, check);
    score = score + check;

    fprintf('Total score: %i/%i', score, totalPoints);
    score = score / totalPoints;
end
