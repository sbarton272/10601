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
print 'Fwd Test Data'
runCmd(['python','alpha.py', './test/test-dev.txt', './test/test-trans.txt', './test/test-emit.txt', './test/test-prior.txt'])

print 'Bckwd Test Data'
runCmd(['python','beta.py', './test/test-dev.txt', './test/test-trans.txt', './test/test-emit.txt', './test/test-prior.txt'])

print
print 'Fwd Real Data'
fwdRes = runCmd(['python','alpha.py', 'dev.txt', 'hmm-trans.txt', 'hmm-emit.txt', 'hmm-prior.txt'])

print 'Bckwd Real Data'
bckwdRes = runCmd(['python','beta.py', 'dev.txt', 'hmm-trans.txt', 'hmm-emit.txt', 'hmm-prior.txt'])

#===============================================
# Script
#===============================================
