# Spencer Barton
# 10-601 
# HW 11

import sys, HMM, math

DEBUG = True

STATES = ['PR', 'VB', 'RB', 'NN', 'PC', 'JJ', 'DT', 'OT']

#===============================================
# Script
#===============================================

HMM = HMM.HiddenMarkovModel()

TRAIN_FILE_NAME = sys.argv[1]

# place training data in list of sentence lists
trainingData = list() # list of lists
vocabulary = set() # vocabulary contains unique words
with open(TRAIN_FILE_NAME) as FID:
	for line in FID:
		data = line.strip().split(' ')
		trainingData.append( data )
		vocabulary.update(set(data))

# random prob assignment or use provided files
if len(sys.argv) == 5:
	# initialization files
	TRANS_FILE_NAME = sys.argv[2]
	EMIT_FILE_NAME = sys.argv[3]
	PRIOR_FILE_NAME = sys.argv[4]
	HMM.initHMM(TRANS_FILE_NAME, EMIT_FILE_NAME, PRIOR_FILE_NAME)
else:
	# init topology with standard files
	HMM.initHMMRand(STATES, vocabulary)

# for debug
if DEBUG:
	print 'prior', HMM.hmmPrior
	print 'trans', HMM.hmmTrans
	print 'emit', HMM.hmmEmit
	print 'states', HMM.getStates()
	print 'observables', HMM.getObservables()

HMM.baumWelchAlg(trainingData)

# for debug
if DEBUG:
	print 'prior', HMM.hmmPrior
	print 'trans', HMM.hmmTrans
	print 'emit', HMM.hmmEmit
	print 'states', HMM.getStates()
	print 'observables', HMM.getObservables()
