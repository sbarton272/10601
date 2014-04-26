# Spencer Barton
# 10-601 
# HW 11

import re

class HiddenMarkovModel(object):
	"""HiddenMarkovModel
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
		# 3 elements define the HMM
		self.hmmPrior = dict();
		self.hmmTrans = dict();
		self.hmmEmit = dict();

#===============================================
# Initialization from files
#===============================================

	def initHMM(self, transFileName, emitFileName, priorFileName, delim = ' '):
		""" Given the HMM data in files read the files and set-up the 
			HMM variables.
		"""
		_initHMMTrans(transFileName, delim)
		_initHMMEmit(emitFileName, delim)
		_initHMMPrior(priorFileName, delim)

	def _initHMMTrans(self, transFileName, delim):
		""" given file fill in data structure, data split by demiliter
			Format is:
			S1 S1:A11 S2:A12
			S2 S1:A21 S2:A22
		"""
		with open(transFileName) as FID:
			for line in FID:
				tokens = delim.split(line)
				Si = tokens[0]
				# update transition probabilities
				self.hmmTrans[Si] = dict()
				for tok in tokens[1:]:
					m = re.match(r'(\w+):([\d\.]+)', tok)
					Sj = m.group(0)
					Aij = m.group(1)
					self.hmmTrans[Si][Sj] = Aij

	def _initHMMEmit(self, emitFileName, delim):
		""" given file fill in data structure, data split by demiliter 
			Format is:
			S1 V1:B11 V2:B12
			S2 V1:B21 V2:B22
		"""
		with open(emitFileName) as FID:
			for line in FID:
				tokens = delim.split(line)
				Si = tokens[0]
				# update emission probabilities
				self.hmmEmit = dict()
				for tok in tokens[1:]:
					m = re.match(r'(\w+):([\d\.]+)', tok)
					Vk = m.group(0)
					Bij = m.group(1)
					self.hmmTrans[Si][Vk] = Bij

	def _initHMMPrior(self, priorFileName, delim):
		""" given file fill in data structure, data split by demiliter 
			Format is:
			S1 P1
			S2 P2
		"""
		with open(priorFileName) as FID:
			for line in FID:
				tokens = delim.split(line)
				Si = tokens[0]
				Pi = tokens[1]
				# update prior probabilities
				self.hmmPrior[Si] = Pi

#===============================================
# Getters
#===============================================

	def getStates(self):
		return set(self.emit.keys())

	def getObservables(self):
		return set(self.emit.values())