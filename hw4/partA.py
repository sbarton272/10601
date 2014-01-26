# Spencer Barton
# 10-601 Spring 2014
# HW 4

import sys, math
from FileParser import FileParser
from Hypothesis import Hypothesis

class FindS(object):
	"""FindS algorithm implementation"""
	def __init__(self):
		self.dataAttr = ["Gender", "Age", "Student?", "PreviouslyDeclined?", 
			"HairLength", "Employed?", "TypeOfColateral", "FirstLoan", "LifeInsurance"]
		self.dataClasif = "Risk"
		self.posClassification = "high"
		self.negClassification = "low"
		self.matchAllChar = '?'
		self.hypothesis = Hypothesis(self.dataAttr, self.matchAllChar)

		# files
		#self.testFileName = sys.argv[2]
		self.devFileName  = "hw4Data/9Cat-Dev.labeled"
		self.trainFileName = "hw4Data/9Cat-Train.labeled"
		self.outFileName = "partA4.txt"

		self.trainingData = FileParser(self.devFileName, self.dataAttr, self.dataClasif).getOutputData()
		self.developmentData = FileParser(self.devFileName, self.dataAttr, self.dataClasif).getOutputData()

		self.printInitialization()
		with open(self.outFileName, 'w') as f:
			self.runTraining(f, self.developmentData)

	def sizeInputSpace(self):
		# num choices per attr is two
		return 2**len(self.dataAttr)

	def numDigitsConceptSpace(self):
		# 2^x ~= 10^(x/3.3)
		return int(math.ceil(10.0 * self.sizeInputSpace() / 3))

	def sizeHypotSpace(self):
		# num choices per attr is two, including ?
		# includes null hypot
		return 3**len(self.dataAttr) + 1

 	def printInitialization(self):
 		print self.sizeInputSpace()
 		print self.numDigitsConceptSpace()
 		print self.sizeHypotSpace()

	def runTraining(self, f, data):
		for dataClasif in data:
			if dataClasif["classification"] == self.posClassification:
				self.hypothesis.updateHypothesis( dataClasif["dataVect"] )

				v = dataClasif["dataVect"]
				f.write( '=' + '\t'.join(v.values()) + '\n')
				f.write( str(self.hypothesis.printHypothesis(v.keys())) + '\n' )


partA = FindS()
	

