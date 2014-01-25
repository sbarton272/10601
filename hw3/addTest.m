function [score] = addTest()
	score = 0;
	totalPoints = 3;

	% can it add two floats? (1 point)
	x = rand();
	y = rand();
	score = score + epsilonAway(add(x,y), x+y);

	% can it add two vectors? (1 point)
	x = rand(2,1);
	y = rand(2,1);
	score = score + epsilonAway(add(x,y), x+y);

	% can it add two matrices? (1 point)
	x = rand(2);
	y = rand(2);
	score = score + epsilonAway(add(x,y), x+y);

	fprintf('Total score: %i/%i', score, totalPoints);
    score = score / totalPoints;
    
end
