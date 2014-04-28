# Spencer Barton
# 10-601 
# HW 11

# TODO functional version instead of for loops

import re, operator, copy, random, math

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
		self.hmmPrior = dict()
		self.hmmTrans = dict()
		self.hmmEmit = dict()
		self.bInitializedTopology = False

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
		self.bInitializedTopology = True

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


	def initHMMRand(self, states, vocabulary):
		""" Create random probabilities for HMM. Probabilities are normalized
			to add up to 1 as appropriate.
			Vocabulary is used so that HMMEmit include the full training 
			vocabulary.
			Requires that the HMM be initialized with file based probabilities
			so as to have the correct topology in place.
		"""
		self._initHMMPriorRand(states)
		self._initHMMEmitRand(states, vocabulary)
		self._initHMMTransRand(states)
		self.bInitializedTopology = True

	def _initHMMPriorRand(self, states):
		""" Assigns random prior per state 
			Assumes that topology has already been initialized
		"""
		self.hmmPrior = dict.fromkeys(states)
		sumP = 0.0		
		for Si in states:
			p = random.random() # [0,1)
			
			if p <= 0.00001:
				p = .5 # p=0 is problematic
			
			self.hmmPrior[Si] = p
			sumP += p
		
		# normalize so total prob is 1
		for Si in states:
			self.hmmPrior[Si] = self.hmmPrior[Si] / sumP
	
	def _initHMMEmitRand(self, states, vocabulary):
		""" Assigns random emission per state 
			vocabulary is a set
			Assumes that topology has already been initialized
		"""
		self.hmmEmit = dict.fromkeys(states)

		for Si in states:
			self.hmmEmit[Si] = dict.fromkeys(vocabulary)

			sumP = 0.0
			for word in vocabulary:

				p = random.random() # [0,1)
				if p <= 0.00001:
					p = .5 # p=0 is problematic

				self.hmmEmit[Si][word] = p
				sumP += p

			# normalize so total prob per state is 1
			for word in vocabulary:
				self.hmmEmit[Si][word] = self.hmmEmit[Si][word] / sumP

	def _initHMMTransRand(self, states):
		""" Assigns random transition prob per state 
			Assumes that topology has already been initialized
		"""
		self.hmmTrans = dict.fromkeys(states)

		for Si in states:

			self.hmmTrans[Si] = dict.fromkeys(states)
			sumP = 0.0
			for Sj in states:

				p = random.random() # [0,1)
				if p <= 0.00001:
					p = .5 # p=0 is problematic

				self.hmmTrans[Si][Sj] = p
				sumP += p

			# normalize so total prob per state is 1
			for Sj in states:
				self.hmmTrans[Si][Sj] = self.hmmTrans[Si][Sj] / sumP

#===============================================
# Forward Algorithm
#===============================================

	def forwardAlg(self, vObserved):
		""" Assuming trained HMM return alpha value for given 
			vObserved (observed vector)
		"""
		if not self.bInitializedTopology:
			raise("Cannot evaluate without initialized topology")

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
		if not self.bInitializedTopology:
			raise("Cannot evaluate without initialized topology")

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
		if not self.bInitializedTopology:
			raise("Cannot decode without initialized topology")
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
				maxSj = None

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
		St = max(VP[T-1].iteritems(), key=operator.itemgetter(1))[0]
		# Backtrace through paths from t = T-1 to 1 (in this case -1 due to 0 indexing)
		path = []
		for t in xrange(T-2,-1,-1):
			path.append(St) # fill in reverse order
			St = paths[St][t]
		path.append(St)
		path.reverse()

		return path, VP

#===============================================
# Baum-Welch Algorithm
#===============================================

	def baumWelchAlg(self, trainingData, printAvgLL = True , minAvgLLDelta = 0.1,	
					 maxNIter = 20):
		""" Takes in training data and generates HMM model. Starting HMM parameters
			can be specified through the assorted files. Training convergence 
			parameters can also be specified.
			trainingData is a list of sentence lists.
		"""
		if not self.bInitializedTopology:
			raise("Cannot train without initialized topology")

		# iterate on algorithm until avg LL converges or the maximum number 
		#  of iterations are made
		nIter = 0
		avgLLDelta = minAvgLLDelta
		avgCurLL   = self._calcAvgLL(trainingData)
		avgPastLL  = None

		M = len(trainingData)
		while ( (nIter < maxNIter) and (avgLLDelta >= minAvgLLDelta) ):
			# clear every iteration
			xi = [None]*M
			gamma = [None]*M

			for m in xrange(0,M):
				# iterate through all training data to get values xi and gamma
				# per observed vector
				# xi is the expected probability of transitioning state i->j
				#  xi = list(vObserved)< list(time)< dict<Si, dict<Sj,prob> > > >
				# gamma is the expected probability of passing through state i 
				#  gamma = list(vObserved)< list(time)< dict<Si, prob> > >
				
				vObserved = trainingData[m]

				# get alpha and beta values for current HMM configuration
				alpha = self._getAlpha(vObserved)
				beta  = self._getBeta(vObserved)
				
				# DEBUG
				# if m == 0:
				# 	alpha = [{'s': 0.34, 't': 0.08}, {'s': 0.066, 't': 0.155}, {'s': 0.02118, 't': 0.09285}, {'s': 0.00625, 't': 0.04919}]
				# 	beta = [{'s': 0.13315, 't': 0.12729}, {'s': 0.2561, 't': 0.2487}, {'s': 0.47, 't': 0.49}, {'s': 1.0, 't': 1.0}]
				# else:
				# 	alpha = [{'s': 0.51, 't': 0.08}, {'s': 0.0644, 't': 0.2145}, {'s': 0.0209, 't': 0.119}]
				# 	beta = [{'s': 0.2421, 't': 0.2507}, {'s': 0.53, 't': 0.51}, {'s': 1.0, 't': 1.0}]

				# print '----------'
				# print vObserved, self.forwardAlg(vObserved)

				# calculate xi and gamma
				# rtrn matrix over time for Si->Sj
				xi[m] = self._getXiM(alpha, beta, vObserved)
				# rtrn vector over time for Si
				gamma[m] = self._getGammaM(alpha, beta)

				# print 'alpha', alpha
				# print 'beta', beta
				# print 'xi'
				for x in xi[m]:
					# print x
					pass
				# print 'gamma'
				for g in gamma[m]:
					# print g
					pass
				# print '----------'


			# xi[0][0]['s']['s'] = 0.28271
			# xi[0][0]['s']['t'] = 0.53383
			# xi[0][0]['t']['s'] = 0.02217
			# xi[0][0]['t']['t'] = 0.16149
			# xi[0][1]['s']['s'] = 0.10071
			# xi[0][1]['s']['t'] = 0.20417
			# xi[0][1]['t']['s'] = 0.07884
			# xi[0][1]['t']['t'] = 0.61648
			# xi[0][2]['s']['s'] = 0.04584
			# xi[0][2]['s']['t'] = 0.13371
			# xi[0][2]['t']['s'] = 0.06699
			# xi[0][2]['t']['t'] = 0.75365
			# xi[1][0]['s']['s'] = 0.23185
			# xi[1][0]['s']['t'] = 0.65071
			# xi[1][0]['t']['s'] = 0.01212
			# xi[1][0]['t']['t'] = 0.13124
			# xi[1][1]['s']['s'] = 0.08286
			# xi[1][1]['s']['t'] = 0.16112
			# xi[1][1]['t']['s'] = 0.09199
			# xi[1][1]['t']['t'] = 0.68996

			# gamma[0][0]['s'] = 0.81654
			# gamma[0][0]['t'] = 0.18366
			# gamma[0][1]['s'] = 0.30488
			# gamma[0][1]['t'] = 0.69532
			# gamma[0][2]['s'] = 0.17955
			# gamma[0][2]['t'] = 0.82064
			# gamma[0][3]['s'] = 0.11273
			# gamma[0][3]['t'] = 0.88727
			# gamma[1][0]['s'] = 0.88256
			# gamma[1][0]['t'] = 0.14336
			# gamma[1][1]['s'] = 0.24398
			# gamma[1][1]['t'] = 0.78195
			# gamma[1][2]['s'] = 0.14939
			# gamma[1][2]['t'] = 0.85061

			# update HMM after looking at all training data
			self._updateHmmPrior(gamma)
			self._updateHmmTrans(xi)
			self._updateHmmEmit(gamma,trainingData)

			# caclulate average log likelihood
			avgPastLL = avgCurLL
			avgCurLL = self._calcAvgLL(trainingData)
			avgLLDelta = avgCurLL - avgPastLL
			nIter += 1

			if printAvgLL:
				print avgCurLL

		return avgCurLL

	def _calcAvgLL(self, trainingData):
		""" caclulate average log likelihood over training data given HMM """
		sumLL = 0.0
		M = len(trainingData)
		for m in xrange(0,M):
			sumLL += math.log( self.forwardAlg( trainingData[m] ) )
		return sumLL / M

	def _updateHmmPrior(self, gamma):
		""" update HMM model prior
			prior is the average prob of transitioning through the given state 
			@ t=1
		"""
		M = len(gamma)
		for Si in self.getStates():
			# calculate hmmPrior[Si]

			probSum = 0.0
			for m in xrange(0,M):
				# sum all prob over samples of going though state Si at t=0
				probSum += gamma[m][0][Si]
				# print 'pi', Si, m, gamma[m][0][Si]

			self.hmmPrior[Si] = probSum / M

	def _updateHmmTrans(self, xi):
		""" update HMM model transition probabilities
			Prob is based on average flow through transition i->j
		"""
		M = len(xi)
		for Si in self.getStates():

			denominator = 0.0
			for Sj in self.getStates():
				# calculate hmmTrans[Si][Sj]

				numerator = 0.0
				for m in xrange(0,M):
					# calculate numerator, sum xi over M
					T = len(xi[m])
					for t in xrange(0, T):
						# sum xi over T
						prob = xi[m][t][Si][Sj]
						numerator += prob
						denominator += prob

						# print 'aij', Si, Sj, m, t, T, prob 

				self.hmmTrans[Si][Sj] = numerator

			# normalize probabilities
			for Sj in self.getStates():
				self.hmmTrans[Si][Sj] = self.hmmTrans[Si][Sj] / denominator

	def _updateHmmEmit(self, gamma,trainingData):
		""" update HMM model """

		M = len(trainingData)

		for Si in self.getStates():
			for vk in self.getObservables():
				# update hmmEmit[Si][vk]

				denominator = 0.0
				numerator = 0.0
				for m in xrange(0,M):
					# iterate over all training data
					vObserved = trainingData[m]
					T = len(vObserved)
					for t in xrange(0,T):
						# iterate over all time
						ot = vObserved[t]
						prob = gamma[m][t][Si]
						denominator += prob
						if ot == vk:
							# count only instance where in state Si and output
							# of training data is vk
							numerator += prob

						# print 'bivk', Si, vk, m, t, prob, ot==vk

				self.hmmEmit[Si][vk] = numerator / denominator

	def _getXiM(self, alpha, beta, vObserved):
		""" Return matrix over time for Si->Sj
			Given alpha, beta and vObserved which are all over time
		"""	
		T = len(alpha)
		xi = [None] * (T-1)

		# iterate over time and state i and then state j to calculate the 
		# probability of transitioning i->j @ t.
		# Note that iterate up to T-1
		for t in xrange(0,T-1):

			totalProb = 0.0
			ot_1 = vObserved[t+1]
			xi[t] = dict.fromkeys(self.getStates())

			# iterate state by state @ t
			for Si in self.getStates():

				totalProb += alpha[t][Si] * beta[t][Si]

				xi[t][Si] = dict.fromkeys(self.getStates())

				# iterate over states that Si can transition to @ t
				for Sj  in self.getStates():

					# print 'xi:', t, Si, Sj
					# print '\t', alpha[t][Si], self.hmmTrans[Si][Sj], self.hmmEmit[Sj][ot_1], beta[t+1][Sj],

					xi[t][Si][Sj] = alpha[t][Si] * self.hmmTrans[Si][Sj] * self.hmmEmit[Sj][ot_1] * beta[t+1][Sj]

					# print xi[t][Si][Sj]

			# print '\t>', totalProb
			# normalize probabilities
			for Si in self.getStates():
				for Sj  in self.getStates():
					xi[t][Si][Sj] = xi[t][Si][Sj] / totalProb
		return xi


	def _getGammaM(self, alpha, beta):
		""" Return matrix over time for Si
			Given alpha and beta which are all over time
		"""	
		T = len(alpha)
		gamma = [None] * T

		# iterate over time and state to calulate probability of being at that
		# state and that time. Also keep track of total prob at given times to
		# normalized by at the end.
		for t in xrange(0,T):
			
			totalProb = 0.0
			gamma[t] = dict.fromkeys(self.getStates())

			# calculate probability
			for Si in self.getStates():
				prob = alpha[t][Si] * beta[t][Si]
				gamma[t][Si] = prob
				totalProb += prob

			# normalized probability
			for Si in self.getStates():
				gamma[t][Si] = gamma[t][Si] / totalProb

		return gamma


#===============================================
# Getters
#===============================================

	def getStates(self):
		return set(self.hmmEmit.keys())

	def getObservables(self):
		return set(self.hmmEmit.values()[0].keys())