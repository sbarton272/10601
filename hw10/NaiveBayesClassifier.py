# Spencer Barton
# 10-601
# Naive Bayes Classifier

import re
from math import log as ln
from math import exp

class NaiveBayesClassifier(object):
	"""NaiveBayesClassifier"""
	def __init__(self, docCategories = ['lib', 'con'], nStopWords = 0, smoothingConst = 1):
		self.smoothingConst = smoothingConst
		self.docCategories = docCategories
		self.nStopWords = nStopWords

	#===============================================
	# Train
	#===============================================

	def train(self, trainFile):

		self.countCategoryProb(trainFile)

		# calc word prob then remove stop words and recalculate
		self.countWords(trainFile)
		print len(self.wordCount)
		if self.nStopWords > 0:
			self.stopWords = self.getStopWords(self.nStopWords)
			self.countWords(trainFile, self.stopWords)

	def countCategoryProb(self, trainFile):
		""" determine probability for each file type category """
		self.categoryProb = dict.fromkeys(self.docCategories,0)

		with open(trainFile) as trainFID:
			for docName in trainFID:

				# read in file names of blog documents
				docCategory = re.match('([a-z]+)', docName).group(0)

				# store doc count in categoryProb var
				self.categoryProb[docCategory] += 1 # if key error will want to see it

		# normalize categoryProb to prob instead of count
		total = float(sum(self.categoryProb.values()))
		# dict comprehension used for efficiency, simply dividing each value by total
		self.categoryProb = {k: v/total for k, v in self.categoryProb.iteritems()}

	def countWords(self, trainFile, stopWords = set()):
		"""For each doc in the training file:
			1) count word occurance
			2) record unique words
			3) update category prob
			4) update word prob conditioned on category """
		self.wordCount = dict(); # stores unique word prob
		self.wordCountsPerCat = dict.fromkeys(self.docCategories,0) # count words by category

		with open(trainFile) as trainFID:
			for docName in trainFID:

				# read in file names of blog documents
				docCategory = re.match('([a-z]+)', docName).group(0)

				with open(docName.strip()) as docFID:
					# add each document's words to the vocabulary
					
					for word in docFID:
						word = self.preprocessWord(word)
						if word not in stopWords:
							# count unique words by doc type
							self.updateWordCount(word, docCategory) 
							self.wordCountsPerCat[docCategory] += 1

	def updateWordCount(self, word, docCategory):
		"""Init/update word count. Data structure is of form {word: {cat1: count1, cat2: count2}}"""

		if word not in self.wordCount:
			self.wordCount[word] = dict.fromkeys(self.docCategories,0) 
		self.wordCount[word][docCategory] += 1 

	def preprocessWord(self, word):
		return word.lower().strip()

	def getWordProb(self, cat, word):
		""" normalize word probability from count """
		vocabLen = len(self.wordCount)
		q = self.smoothingConst

		# prob: (count + q) / (nWordsByCat + vocabLength)
		n  = self.wordCountsPerCat[cat]
		nk = self.wordCount[word][cat]
		return (nk + q) / float(n + q*vocabLen)

	#===============================================
	# Test
	#===============================================

	def test(self, testFile):
		""" label docs in test file """

		# for reporting correctness 
		nCorrect = 0.0
		nTotal = 0.0

		with open(testFile) as testFID:
			for docName in testFID:
				# classify all docs

				# read in file names of blog documents
				docCategory = re.match('([a-z]+)', docName).group(0)
				classification = self.classify(docName)

				if classification == docCategory:
					nCorrect += 1
				nTotal += 1

				self.printClassification(classification)
		
		print 'Accuracy:', round(nCorrect / nTotal,4)

	def classify(self, docName):
		""" classify the given document. Utilize the naive bayes assumption with log probability."""		
		maxProb = float('-inf') # lowest number for comparison purposes
		maxLabel = ''

		for cat in self.docCategories:
			# calculate probabilities of various classifications
			prob = ln(self.categoryProb[cat])

			with open(docName.strip()) as docFID:
				# add each document's words to the vocabulary
				
				for word in docFID:
					word = self.preprocessWord(word)
					if ((word in self.wordCount) and
					   (word not in self.stopWords)):
						# ignore words not in vocabulary or stopWords
						prob += ln( self.getWordProb(cat,word) )

			# update max prob and label
			if maxProb < prob:
				maxProb = prob
				maxLabel = cat

		return maxLabel

	def printClassification(self, classification):
		print classification[0].upper()


	#===============================================
	# Common/Stop words
	#===============================================

	def getSortedWords(self):
		"""return words sorted by probability in a dictionay with category keys:
			{cat: [(word, prob),...]}"""

		sortedWords = dict.fromkeys(self.docCategories,None)

		# reorganize (word,prob) dicts into lists
		for word,counts in self.wordCount.iteritems():

			for cat in self.docCategories:
				prob = self.getWordProb(cat, word)
				if (sortedWords[cat] == None): sortedWords[cat] = [] # deal with first case
				sortedWords[cat].append( (word,prob) ) # store as tuple

		# sort lists within categories
		for cat in self.docCategories:
			sortedWords[cat] = sorted( sortedWords[cat] ,key=lambda x: x[1], reverse=True) 

		return sortedWords

	def getStopWords(self, nStopWords):
		""" Set top N words as stop words """
		stopWords = set([])
		if nStopWords == 0:
			return stopWords

		sortedWords = self.getSortedWords()
		stopWordPairs = []
		for cat,words in sortedWords.iteritems():
			# take top N words from each category to find top N overall
			# not the most efficient but simple
			stopWordPairs += words[0:nStopWords]

		stopWordPairs = sorted(stopWordPairs, key=lambda x: x[1], reverse=True) 		
		
		# take only words not prob and remove duplicates
		stopWordPairs = map(lambda x: x[0], stopWordPairs)
		for w in stopWordPairs:
			# add top N unique words to stopWords set
			stopWords.add(w)
			if len(stopWords) >= nStopWords: break

		return stopWords