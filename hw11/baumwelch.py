# Spencer Barton
# 10-601 
# HW 11

import sys, HMM, math

DEBUG = False

#===============================================
# Script
#===============================================

HMM = HMM.HiddenMarkovModel()

TRAIN_FILE_NAME = sys.argv[1]
if len(sys.argv) == 5:
	TRANS_FILE_NAME = sys.argv[2]
	EMIT_FILE_NAME = sys.argv[3]
	PRIOR_FILE_NAME = sys.argv[4]
	HMM.initHMM(TRANS_FILE_NAME, EMIT_FILE_NAME, PRIOR_FILE_NAME)

# for debug
if DEBUG:
	print 'prior', HMM.hmmPrior
	print 'trans', HMM.hmmTrans
	print 'emit', HMM.hmmEmit
	print 'states', HMM.getStates()
	print 'observables', HMM.getObservables()