function [z] = epsilonAway(x,y)
	EPSILON = 1e-6;
	z = max(abs(x(:) - y(:))) < EPSILON;
