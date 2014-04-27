# Spencer Barton
# 10-601 
# HW 11

# TODO functional version instead of for loops

import re, operator, copy

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
		self._initHMMTrans(transFileName, delim)
		self._initHMMEmit(emitFileName, delim)
		self._initHMMPrior(priorFileName, delim)

	def _initHMMTrans(self, transFileName, delim):
		""" given file fill in data structure, data split by demiliter
			Format is:
			S1 S1:A11 S2:A12
			S2 S1:A21 S2:A22
		"""
		with open(transFileName) as FID:
			for line in FID:
				tokens = line.strip().split(delim)
				Si = tokens[0]
				# update transition probabilities
				self.hmmTrans[Si] = dict()

				# go token by token to update prob
				for tok in tokens[1:]:
					m = re.match(r'(\S+):(\S+)', tok)
					Sj = m.group(1)
					Aij = float(m.group(2))
					self.hmmTrans[Si][Sj] = Aij

	def _initHMMEmit(self, emitFileName, delim):
		""" given file fill in data structure, data split by demiliter 
			Format is:
			S1 V1:B11 V2:B12
			S2 V1:B21 V2:B22
		"""
		with open(emitFileName) as FID:
			for line in FID:
				tokens = line.strip().split(delim)
				Si = tokens[0]
				# update emission probabilities
				self.hmmEmit[Si] = dict()
				for tok in tokens[1:]:
					m = re.match(r'(\S+):(\S+)', tok)
					Vk = m.group(1)
					Bij = float(m.group(2))
					self.hmmEmit[Si][Vk] = Bij

	def _initHMMPrior(self, priorFileName, delim):
		""" given file fill in data structure, data split by demiliter 
			Format is:
			S1 P1
			S2 P2
		"""
		with open(priorFileName) as FID:
			for line in FID:
				tokens = line.strip().split(delim)
				Si = tokens[0]
				Pi = float(tokens[1])
				# update prior probabilities
				self.hmmPrior[Si] = Pi

#===============================================
# Forward Algorithm
#===============================================

	def forwardAlg(self, vObserved):
		""" Assuming trained HMM return alpha value for given 
			vObserved (observed vector)
		"""

		alpha = self._getAlpha(vObserved)
		return sum( alpha[-1].values() )

	def _getAlpha(self, vObserved):
		""" Generate alpha values
			Returns list ordered by time of dicts with state probabilities:
			[ {S1:p1, S2:p2}, {S1:p3, S2:p4} ]
		"""
		T = len(vObserved)
		alpha = [None]*T

		# first alpha values per state Si
		t = 0
		alpha[t] = dict()
		o1 = vObserved[t]
		for Si in self.getStates():
			alpha[t][Si] = self.hmmPrior[Si] * self.hmmEmit[Si][o1]

		# subsequent alpha based on prior alpha
		for t in xrange(1,T):
			# iterate through alphas over time
			ot = vObserved[t]
			alpha[t] = dict()

			for Si in self.getStates():
				# iterate through states per alpha
				bi = self.hmmEmit[Si][ot]
				tmpSum = 0.0

				for Sj in self.getStates(): 
					# iterate through prior states to get transition prob
					tmpSum += alpha[t-1][Sj] * self.hmmTrans[Sj][Si]

				# update alpha
				alpha[t][Si] = tmpSum * bi

		return alpha

#===============================================
# Backward Algorithm
#===============================================

	def backwardAlg(self, vObserved):
		""" Assuming trained HMM return beta value for given 
			vObserved (observed vector)
		"""

		beta = self._getBeta(vObserved)
		rtrnVal = 0.0
		o1 = vObserved[0]
		for Si in self.getStates():
			rtrnVal += self.hmmPrior[Si] * self.hmmEmit[Si][o1]	* beta[0][Si]		
		return rtrnVal

	def _getBeta(self, vObserved, finalStateProb = 1.0):
		""" Generate beta values
			Returns list ordered by time of dicts with state probabilities:
			[ {S1:p1, S2:p2}, {S1:p3, S2:p4} ]
		"""
		T = len(vObserved)
		beta = [None]*T

		# last beta values per state Si
		t = T-1
		beta[t] = dict()
		for Si in self.getStates():
			beta[t][Si] = finalStateProb

		# subsequent beta based on prior beta
		for t in xrange(T-2,-1,-1):
			# iterate through betas over time
			ot = vObserved[t+1]
			beta[t] = dict()

			for Si in self.getStates():
				# iterate through states per beta
				tmpSum = 0.0

				for Sj in self.getStates(): 
					# iterate through next states to get transition prob
					tmpSum += beta[t+1][Sj] * self.hmmTrans[Si][Sj] * self.hmmEmit[Sj][ot]

				# update beta
				beta[t][Si] = tmpSum
		return beta

#===============================================
# Viterbi Algorithm
#===============================================

	def viterbiAlg(self, vObserved):
		""" Assuming trained HMM return path value for given 
			vObserved (observed vector)
		"""
		return self._getPath( vObserved )

	def _getPath(self, vObserved):
		path, VP = self._getPathVP(vObserved)
		return path

	def _getVP(self, vObserved):
		path, VP = self._getPathVP(vObserved)
		return VP

	def _getPathVP(self, vObserved):
		""" Generate VP and path values
			TODO Returns list ordered by time of dicts with state probabilities:
			[ {S1:p1, S2:p2}, {S1:p3, S2:p4} ]
		"""
		T = len(vObserved)
		VP = [None]*T # list of dicts
		paths = dict.fromkeys(self.getStates()) # dict of lists

		# first VP values per state Si
		t = 0
		VP[t] = dict()
		o1 = vObserved[t]
		for Si in self.getStates():
			VP[t][Si] = self.hmmPrior[Si] * self.hmmEmit[Si][o1]
			paths[Si] = []

		# subsequent VP based on prior VP
		for t in xrange(1,T):
			# iterate through VPs over time
			ot = vObserved[t]
			VP[t] = dict()

			for Si in self.getStates():
				# iterate through states per VP
				bi = self.hmmEmit[Si][ot]
				maxVP = 0.0

				for Sj in self.getStates(): 
					# iterate through prior states to get max VP and which state it came from
					tmpVP = VP[t-1][Sj] * self.hmmTrans[Sj][Si] * bi
					if tmpVP > maxVP:
						maxVP = tmpVP
						maxSj = Sj

				# update VP with max of possibilities
				VP[t][Si] = maxVP

				# update paths
				paths[Si].append( maxSj )

		# final state is max of VP_T
		S_T = max(VP[T-1].iteritems(), key=operator.itemgetter(1))[0]
		# Knowing final state, path is optimal path to that state
		# Pull out path for state S_T over time, final state in path is S_T
		path = copy.copy( paths[S_T] )
		path.append(S_T)

		print S_T, paths
		print VP

		return path, VP

#===============================================
# Getters
#===============================================

	def getStates(self):
		return set(self.hmmEmit.keys())

	def getObservables(self):
		return set(self.hmmEmit.values()[0].keys())