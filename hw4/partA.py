# Spencer Barton
# 10-601 Spring 2014
# HW 4

import sys, math
from FileParser import FileParser
from Hypothesis import ConjHypothesis

class FindS(object):
	"""FindS algorithm implementation"""
	def __init__(self):
		self.dataAttr = ["Gender", "Age", "Student?", "PreviouslyDeclined?", 
			"HairLength", "Employed?", "TypeOfColateral", "FirstLoan", "LifeInsurance"]
		self.dataClasif = "Risk"
		self.posClassification = "high"
		self.negClassification = "low"
		self.matchAllChar = '?'
		self.hypothesis = ConjHypothesis(self.dataAttr, self.matchAllChar)
		self.printEvery = 30

		# files
		self.testFileName = "hw4Data/9Cat-Test.labeled" #sys.argv[1]
		self.devFileName  = "hw4Data/9Cat-Dev.labeled"
		self.trainFileName = "hw4Data/9Cat-Train.labeled"
		self.outFileName = "partA4.txt"

		self.trainingData = FileParser(self.trainFileName, self.dataAttr, self.dataClasif).getOutputData()
		self.developmentData = FileParser(self.devFileName, self.dataAttr, self.dataClasif).getOutputData()
		self.testData = FileParser(self.testFileName, self.dataAttr, self.dataClasif, False).getOutputData()

		# run program
		self.printInitialization()
		with open(self.outFileName, 'w') as f:
			self.runTraining(f, self.trainingData)
		self.runTrial(self.developmentData)
		self.runClassification(self.testData)


	def sizeInputSpace(self):
		# num choices per attr is two
		return 2**len(self.dataAttr)

	def numDigitsConceptSpace(self):
		# 2^x ~= 10^(x/3.3)
		return int(math.ceil(3.0 * self.sizeInputSpace() / 10)) + 1

	def sizeHypotSpace(self):
		# num choices per attr is two, including ?
		# includes null hypot
		return 3**len(self.dataAttr) + 1

 	def printInitialization(self):
 		print self.sizeInputSpace()
 		print self.numDigitsConceptSpace()
 		print self.sizeHypotSpace()

	def runTraining(self, outFile, data):
		i = 0 # iterator for printing
		for dataClassified in data:
			i += 1
			if dataClassified["classification"] == self.posClassification:
				self.hypothesis.updateHypothesis( dataClassified["dataVect"] )

			# print every once and awhile
			if( i % self.printEvery == 0 ):
				outFile.write( str(self.hypothesis) + '\n' )

	def runTrial(self, data):
		# print misclassification rate
		totalIncorrect = 0.0

		for dataClassified in data:
			classification = self.hypothesis.classify( dataClassified["dataVect"], self.posClassification, self.negClassification )
			
			if classification != dataClassified["classification"]:
				totalIncorrect += 1

		print round( totalIncorrect/len(data), 2)

	def runClassification(self, data):
		# print classification
		for dataVect in data:
			print self.hypothesis.classify( dataVect, self.posClassification, self.negClassification )

partA = FindS()
	

