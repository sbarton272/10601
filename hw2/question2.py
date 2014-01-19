# Spencer Barton, sebarton
# 10-601 S14 HW2
# question 2

import sys

fileName = sys.argv[1]
with open(fileName, 'r') as f:
	line = f.readline()

# parse
D = dict()
for word in line.split(' '):
	if word in D:
		D[word]+=1;
	else:
		D[word] = 1;

# output
output = ""
for (k,v) in D.iteritems():
	output = output + k + ':' + str(v) + ','

# remove final comma
sys.stdout.write(output[:-1])