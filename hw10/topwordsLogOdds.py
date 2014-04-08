# Spencer Barton
# 10-601
# Naive Bayes Classifier

from NaiveBayesClassifier import NaiveBayesClassifier
from math import log
import sys

#===============================================
# Script
#===============================================

TRAIN_FILE_NAME = sys.argv[1]

NB = NaiveBayesClassifier()
NB.train(TRAIN_FILE_NAME)
topWords = NB.getSortedWords()

# sort words
libWords = sorted( topWords['lib'] ,key=lambda x: x[0], reverse=True) 
conWords = sorted( topWords['con'] ,key=lambda x: x[0], reverse=True) 

# calculate log prob per word
topLibWords = []
for i in xrange(0,len(libWords)):
	libWord = libWords[i]
	conWord = conWords[i]
	topLibWords.append( ( libWord[0], log(libWord[1]/conWord[1]) ) )
topLibWords = sorted( topLibWords ,key=lambda x: x[1], reverse=True) 

topConWords = []
for i in xrange(0,len(libWords)):
	libWord = libWords[i]
	conWord = conWords[i]
	topConWords.append( ( libWord[0], log(conWord[1]/libWord[1]) ) )
topConWords = sorted( topConWords ,key=lambda x: x[1], reverse=True) 

# print top 20 results
N = 20
for i in xrange(0,N):
	pair = topLibWords[i]
	print pair[0], round(pair[1],4)

print

for i in xrange(0,N):
	pair = topConWords[i]
	print pair[0], round(pair[1],4)
