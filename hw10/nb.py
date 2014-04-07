# Spencer Barton
# 10-601
# Naive Bayes Classifier

import re
from math import log as ln

class NaiveBayesClassifier(object):
	"""NaiveBayesClassifier"""
	def __init__(self, trainFile, testFile, docCategories, smoothingConst = 1):
		self.trainFile = trainFile
		self.testFile  = testFile
		self.smoothingConst = smoothingConst

		self.docCategories = docCategories
		self.categoryProb = dict.fromkeys(self.docCategories,0)

		self.wordProb = dict(); # stores unique word prob

		self.train(self.trainFile)
		self.test(self.testFile)


	#===============================================
	# Train
	#===============================================

	def train(self, trainFile):
		"""For each doc in the training file:
			1) count word occurance
			2) record unique words
			3) update category prob
			4) update word prob conditioned on category """
		wordCountsByCat = dict.fromkeys(self.docCategories,0) # count words by category

		with open(trainFile) as trainFID:
			for docName in trainFID:

				# read in file names of blog documents
				docCategory = re.match('([a-z]+)', docName).group(0)

				# store doc count in categoryProb var
				self.categoryProb[docCategory] += 1 # if key error will want to see it

				with open(docName.strip()) as docFID:
					# add each document's words to the vocabulary
					
					for word in docFID:
						word = self.preprocessWord(word)
						# count unique words by doc type
						self.updateWordProb(word, docCategory) 
						wordCountsByCat[docCategory] += 1

		# normalize categoryProb to prob instead of count
		total = float(sum(self.categoryProb.values()))
		# dict comprehension used for efficiency, simply dividing each value by total
		self.categoryProb = {k: v/total for k, v in self.categoryProb.iteritems()}

		# normalize prob per category
		self.normalizeWordProb(wordCountsByCat)

	def normalizeWordProb(self, wordCountsByCat):
		"""normalize word probability from count"""
		vocabLen = len(self.wordProb)
		q = self.smoothingConst

		# updating each count to be a prob: (count + q) / (nWordsByCat + vocabLength)
		for (word,counts) in self.wordProb.iteritems():
			for (cat, nk) in counts.iteritems():
				n = wordCountsByCat[cat]
				self.wordProb[word][cat] = (nk + q) / float(n + q*vocabLen)

	def updateWordProb(self, word, docCategory):
		"""Init/update word count. Data structure is of form {word: {cat1: count1, cat2: count2}}"""

		if word not in self.wordProb:
			self.wordProb[word] = dict.fromkeys(self.docCategories,0) 
		self.wordProb[word][docCategory] += 1 

	def preprocessWord(self, word):
		return word.lower()

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
					if word in self.wordProb:
						# ignore words not in vocabulary
						prob += ln(self.wordProb[word][cat])

			# update max prob and label
			if maxProb < prob:
				maxProb = prob
				maxLabel = cat

		return maxLabel

	def printClassification(self, classification):
		print classification[0].upper()


#===============================================
# Script
#===============================================

# NB_TEST = NaiveBayesClassifier('train.txt', 'test.txt', ['foo', 'bar'])
# print NB_TEST.categoryProb
# for (k,v) in NB_TEST.wordProb.iteritems():
# 	print k, v

TRAIN_FILE_NAME = 'split.train'  #sys.argv[1]
TEST_FILE_NAME  = 'split.test'  #sys.argv[2]

NB = NaiveBayesClassifier(TRAIN_FILE_NAME, TEST_FILE_NAME, ['lib', 'con'])