# Spencer Barton
# 10-601 Spring 2011
# HW 6 2/19/14
# inspect.py

import sys, csv, math

# arg 1 is file
fileName = sys.argv[1] 

# label settings
plusLabel  = 'yes'
minusLabel = 'no'
classificationAttr = 'hit'

# vars
numPlus  = 0.0
numMinus = 0.0

with open(fileName, 'r') as f:

    # assume header on 1st row
    reader = csv.DictReader(f)

    for r in reader:
        if r[classificationAttr] == plusLabel:
            numPlus += 1
        else:
            numMinus += 1

# output entropy
probP = numPlus / (numPlus + numMinus)
probM = numMinus / (numPlus + numMinus)
H = -probP * math.log(probP, 2) - probM * math.log(probM, 2)
print "entropy: ", round(H,3)

# output error: prob minus 
print "error: ", round(probM,3)