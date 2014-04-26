# Spencer Barton
# 10-601 
# HW 11

class HMM(object):
	"""HMM - hidden markov model
		4 algorithms are implemented:
		Evaluation
		- Forward
		- Backward
		Decoding
		- Viterbi
		Learning
		- Baum-Welch (Forward-Backward)
	"""
	def __init__(self):
		# 5 elements define the HMM
		self.hmmStates = set();
		self.hmmObservables = set();
		self.hmmPrior = dict();
		self.hmmTrans = dict();
		self.hmmOutput = dict();