# Spencer Barton
# 10-601 
# HW 11

import subprocess

def runCmd(cmd):
	process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
	out, err = process.communicate()
	print(out)
	return out


#===============================================
# Forward Backward
#===============================================

# test with test data
# print 'Fwd Test Data'
# alpha = runCmd(['python','alpha.py', './test/test-dev.txt', './test/test-trans.txt', './test/test-emit.txt', './test/test-prior.txt'])

# print 'Bckwd Test Data'
# beta = runCmd(['python','beta.py', './test/test-dev.txt', './test/test-trans.txt', './test/test-emit.txt', './test/test-prior.txt'])

# print 'Viterbi Test Data'
# VP = runCmd(['python','viterbi.py', './test/test-dev.txt', './test/test-trans.txt', './test/test-emit.txt', './test/test-prior.txt'])
# runCmd(['python','viterbi.py', 'test/test3-dev.txt', 'test/test3-trans.txt', 'test/test3-emit.txt', 'test/test3-prior.txt'])

print 'Baum Welch Test Data'
runCmd(['python','baumwelch.py', './test/test4-train.txt', './test/test4-trans.txt', './test/test4-emit.txt', './test/test4-prior.txt'])
runCmd(['python','baumwelch.py', './test/test2-dev.txt'])

# print 'Correctness:'
# print 'Fwd', alpha == beta
# print 'Bckwd', alpha == beta
# print 'Viterbi', VP.strip() == 'A_I A_I B_F'


# print '-----------------------'
# print 'Fwd Real Data'
# fwdRes = runCmd(['python','alpha.py', 'dev.txt', 'hmm-trans.txt', 'hmm-emit.txt', 'hmm-prior.txt'])

# print 'Bckwd Real Data'
# bckwdRes = runCmd(['python','beta.py', 'dev.txt', 'hmm-trans.txt', 'hmm-emit.txt', 'hmm-prior.txt'])

# print 'Viterbi Real Data'
# outFile = 'my-hmm-tag.txt'
# paths = runCmd(['python','viterbi.py', 'dev.txt', 'hmm-trans.txt', 'hmm-emit.txt', 'hmm-prior.txt'])

# with open(outFile, 'w') as FID:
# 	FID.write(paths)

print 'Baum Welch Real Data'
runCmd(['python','baumwelch.py', 'train.txt', 'hmm-trans.txt', 'hmm-emit.txt', 'hmm-prior.txt'])
runCmd(['python','baumwelch.py', 'train.txt'])


# print 'Correctness:'
# print 'Fwd-Bckwd', fwdRes == bckwdRes
# runCmd(['python','eval.py','dev-tag.txt',outFile])

