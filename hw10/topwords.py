# Spencer Barton
# 10-601
# Naive Bayes Classifier

from NaiveBayesClassifier import NaiveBayesClassifier

#===============================================
# Script
#===============================================

TRAIN_FILE_NAME = 'split.train'  #sys.argv[1]

NB = NaiveBayesClassifier()
NB.train(TRAIN_FILE_NAME)
topWords = NB.getSortedWords()

N = 20
for i in xrange(0,N):
	pair = topWords['lib'][i]
	print pair[0], round(pair[1],4)

print

for i in xrange(0,N):
	pair = topWords['con'][i]
	print pair[0], round(pair[1],4)
