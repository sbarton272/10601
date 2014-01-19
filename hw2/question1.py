# Spencer Barton, sebarton
# 10-601 S14 HW2
# question 1

import sys

fileName = sys.argv[1]
with open(fileName, 'r') as f:
	line = f.readline()
# split on space, join with comma and strp any possible whitepace
sys.stdout.write(','.join(line.split(' ')).strip())
