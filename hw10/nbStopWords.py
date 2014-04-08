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
N_STOP_WORDS    = int(sys.argv[3])

NB = NaiveBayesClassifier(nStopWords = N_STOP_WORDS)
NB.train(TRAIN_FILE_NAME)
NB.test(TEST_FILE_NAME)