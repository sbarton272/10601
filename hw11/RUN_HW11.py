# Spencer Barton
# 10-601 
# HW 11

import subprocess

def runCmd(cmd):
	process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
	out, err = process.communicate()
	print(out)

# test with test data
runCmd(['python','alpha.py', './test/test-dev.txt', './test/test-trans.txt', './test/test-emit.txt', './test/test-prior.txt'])
