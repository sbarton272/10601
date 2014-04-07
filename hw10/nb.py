# Spencer Barton
# 10-601
# Naive Bayes Classifier

import re

#TEST_FILE_NAME = sys.argv[1] 
#TRAIN_FILE_NAME = sys.argv[2] 

class NaiveBayesClassifier(object):
	"""NaiveBayesClassifier"""
	def __init__(self, trainFile, testFile, docCategories, smoothingConst = 1):
		self.trainFile = trainFile
		self.testFile  = testFile
		self.smoothingConst = smoothingConst

		self.docCategories = docCategories
		self.categoryProb = dict.fromkeys(self.docCategories,0)

		self.wordProb = dict(); # stores unique word prob

		# 1) Gather vocabulary found in training files
		# 2) Determine document category probability
		self.parseDocs(self.trainFile)

	def parseDocs(self, trainFile):
		"""get all unique words in training file
			also records counts of doc categories"""
		wordCountsByCat = dict.fromkeys(self.docCategories,0) # count words by category

		with open(trainFile) as trainFID:
			for docName in trainFID:

				# read in file names of blog documents
				#docCategory = re.match('([a-z]+)', docName).group(0)
				docCategory = re.match('testData/([a-z]+)', docName).group(1)

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
			for (cat, n) in counts.iteritems():
				nk = wordCountsByCat[cat]
				self.wordProb[word][cat] = (n + q) / float(nk + q*vocabLen)

	def updateWordProb(self, word, docCategory):
		"""Init/update word count. Data structure is of form {word: {cat1: count1, cat2: count2}}"""

		if word not in self.wordProb:
			self.wordProb[word] = dict.fromkeys(self.docCategories,0) 
		self.wordProb[word][docCategory] += 1 

	def preprocessWord(self, word):
		return word.strip().lower()

NB_TEST = NaiveBayesClassifier('testData/test.txt', '', ['lib', 'con'])
print NB_TEST.categoryProb
for (k,v) in NB_TEST.wordProb.iteritems():
	print k, v