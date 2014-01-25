function [score] = dotpTest()
	score = 0;
	totalPoints = 3;

    check = (dotp([3 4], [3 4]) == 25);
    feedback(1, check);
	score = score + check;

	check = (dotp([1 -1], [-1 -1]) == 0);
    feedback(2, check);
	score = score + check;

	check = (dotp([3 4], [0 0]) == 0);
    feedback(3, check);
	score = score + check;

    fprintf('Total score: %i/%i', score, totalPoints);
	score = score / totalPoints;
end
