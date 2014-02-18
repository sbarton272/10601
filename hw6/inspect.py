# Spencer Barton
# 10-601 Spring 2011
# HW 6 2/19/14
# inspect.py

import sys, csv, math

# arg 1 is file
fileName = sys.argv[1] 

# label settings
classificationOpts = set(['hit', 'grade'])

# vars
labelCounts = {}

with open(fileName, 'r') as f:

    # assume header on 1st row
    reader = csv.DictReader(f)

    for r in reader:
        keys = r.keys()
        key = list(set(keys).intersection( classificationOpts ))[0]

        label = r[key]
        if label not in labelCounts:
            labelCounts[label] = 0
        labelCounts[label] += 1

assert( len(labelCounts.keys()) == 2 )
numPlus  = float(max(labelCounts.values()))
numMinus = float(min(labelCounts.values()))

# output entropy
probP = numPlus / (numPlus + numMinus)
probM = numMinus / (numPlus + numMinus)
H = -probP * math.log(probP, 2) - probM * math.log(probM, 2)
print "entropy:", round(H,3)

# output error: prob minus 
print "error:", round(probM,3)