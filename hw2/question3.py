# Spencer Barton, sebarton
# 10-601 S14 HW2
# question 3

import sys
from string import lower

inFileName = sys.argv[1]
stopwordsFileName = sys.argv[2]

# generate stopwords
with open(stopwordsFileName, 'r') as stop_file:
    content = stop_file.read()
    words = content.split('\n')
    stopwords  = set( map(lower, words) )

with open(inFileName, 'r') as f:
	line = f.readline()

# parse
D = dict()
for word in line.split(' '):
	if word.lower() not in stopwords:
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