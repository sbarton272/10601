# Spencer Barton, sebarton
# 10-601 S14 HW2
# question 3

import sys, re
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
words = re.findall('(\S+)',line)
words = map(lower,words)
for word in words:
	if word not in stopwords:
		if word in D:
			D[word] +=1;
		else:
			D[word] = 1;

# output
output = ""
for (k,v) in sorted(D.items()):
	output = output + k + ':' + str(v) + ','

# remove final comma
sys.stdout.write(output[:-1])