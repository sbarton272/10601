# Spencer Barton
# 10-601 
# HW 11

import sys, HMM

#===============================================
# Script
#===============================================

DEV_FILE_NAME = sys.argv[1]
TRANS_FILE_NAME = sys.argv[2]
EMIT_FILE_NAME = sys.argv[3]
PRIOR_FILE_NAME = sys.argv[4]

HMM = HMM.HiddenMarkovModel()

HMM.initHMM(TRANS_FILE_NAME, EMIT_FILE_NAME, PRIOR_FILE_NAME)

print HMM.hmmPrior
print HMM.hmmTrans
print HMM.hmmEmit
print HMM.getStates()
print HMM.getObservables()