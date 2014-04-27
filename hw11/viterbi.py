# Spencer Barton
# 10-601 
# HW 11

import sys, HMM

DEBUG = True

#===============================================
# Script
#===============================================

DEV_FILE_NAME = sys.argv[1]
TRANS_FILE_NAME = sys.argv[2]
EMIT_FILE_NAME = sys.argv[3]
PRIOR_FILE_NAME = sys.argv[4]

HMM = HMM.HiddenMarkovModel()

HMM.initHMM(TRANS_FILE_NAME, EMIT_FILE_NAME, PRIOR_FILE_NAME)

# for debug
if DEBUG:
	print 'prior', HMM.hmmPrior
	print 'trans', HMM.hmmTrans
	print 'emit', HMM.hmmEmit
	print 'states', HMM.getStates()
	print 'observables', HMM.getObservables()

# actual output
delim = ' '
with open(DEV_FILE_NAME) as FID: 
	for line in FID:
		vObserved = line.strip().split(delim)
		path = HMM.viterbiAlg(vObserved)

		# generate format output: observed_State
		combined = []
		for i in xrange(0, len(path)):
			combined.append( vObserved[i] + '_' + path[i] )

		print ' '.join(combined)

		# debug
		if DEBUG: print 'VP:', vObserved, HMM._getVP(vObserved)

