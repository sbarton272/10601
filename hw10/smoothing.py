# Spencer Barton
# 10-601
# Naive Bayes Classifier

from NaiveBayesClassifier import NaiveBayesClassifier

#===============================================
# Script
#===============================================

TRAIN_FILE_NAME = 'split.train'  #sys.argv[1]
TEST_FILE_NAME  = 'split.test'  #sys.argv[2]
Q 				= 5 #sys.argv[2]

NB = NaiveBayesClassifier(smoothingConst = Q)
NB.train(TRAIN_FILE_NAME)
NB.test(TEST_FILE_NAME)