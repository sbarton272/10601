# Spencer Barton
# 10-601
# Naive Bayes Classifier

import re

TEST_FILE_NAME = sys.argv[1] 
TRAIN_FILE_NAME = sys.argv[2] 

class NaiveBayesClassifier(object):
	"""NaiveBayesClassifier"""
	def __init__(self, trainFile, testFile, categories):
		self.trainFile = trainFile
		self.testFile  = testFile

		self.categories = categories
		self.categoryProb = dict.fromkeys(self.categories,0)

		# 1) Gather vocabulary found in training files
		self.vocabulary = self.getVocab(self.trainFile)
		self.vocabSize = len(self.vocab)

		# 2) Determine word prob.
		#self.wordProb

	def getVocab(self, trainFile):
		"""get all unique words in training file
			also records counts of doc categories"""
		vocabulary = set(); # stores unique words

		with open(trainFile) as trainFID:
			for docName in trainFID:
				# read in file names of blog documents
				docType = re.match('([a-z]+)', docName).group(0)
				
				# store doc count in categoryProb var
				self.categoryProb[docType] += 1 # if key error will want to see it

				with open(docName) as docFID:
					# add each document's words to the vocabulary
					
					for
						# set 
						vocabulary.add(word)

		# normalize categoryProb to prob instead of count
		total = float(sum(self.categoryProb.values()))
		# dict comprehension used for efficiency, simply dividing each value by total
		self.categoryProb = {k: v/total for k, v in self.categoryProb.iteritems()}

		return vocabulary


