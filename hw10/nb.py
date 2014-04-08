# Spencer Barton
# 10-601
# Naive Bayes Classifier

from NaiveBayesClassifier import NaiveBayesClassifier
import sys

#===============================================
# Script
#===============================================

TRAIN_FILE_NAME = sys.argv[1]
TEST_FILE_NAME  = sys.argv[2]

NB = NaiveBayesClassifier()
NB.train(TRAIN_FILE_NAME)
NB.test(TEST_FILE_NAME)