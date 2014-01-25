# Spencer Barton, sebarton
# 10-601 S14 HW2
# question 1

import sys, re
from string import lower

fileName = sys.argv[1]
with open(fileName, 'r') as f:
	line = f.readline()

# pull non-whitespace
words = re.findall('(\S+)',line)
words = map(lower,words)
words = sorted(set(words)) # remove duplicates
sys.stdout.write(','.join(words))
