# Spencer Barton
# 10-601
# Naive Bayes Classifier

from NaiveBayesClassifier import NaiveBayesClassifier

#===============================================
# Script
#===============================================

# NB_TEST = NaiveBayesClassifier('train.txt', 'test.txt', ['foo', 'bar'])
# print NB_TEST.categoryProb
# for (k,v) in NB_TEST.wordProb.iteritems():
# 	print k, v

TRAIN_FILE_NAME = 'split.train'  #sys.argv[1]
TEST_FILE_NAME  = 'split.test'  #sys.argv[2]

NB = NaiveBayesClassifier()
NB.train(TRAIN_FILE_NAME)
NB.test(TEST_FILE_NAME)