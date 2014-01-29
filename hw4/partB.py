# Spencer Barton
# 10-601 Spring 2014
# HW 4

import sys, itertools
from FileParser import FileParser

class LTE(object):
	"""FindS algorithm implementation"""
	def __init__(self):
		self.dataAttr = ["Gender", "Age", "Student?", "PreviouslyDeclined?"]
		self.dataAttrOpt = {"Gender": ["Male", "Female"], "Age": ["Young", "Old"], "Student?": ["Yes", "No"] , "PreviouslyDeclined?": ["Yes", "No"]}
		self.dataClasif = "Risk"
		self.posClassification = "high"
		self.negClassification = "low"
		self.versionSpace = self.populateVS()
		self.printEvery = 30

		# files
		self.testFileName = "hw4Data/4Cat-Dev.labeled" #sys.argv[1]
		self.devFileName  = "hw4Data/4Cat-Dev.labeled"
		self.trainFileName = "hw4Data/4Cat-Train.labeled"
		self.outFileName = "partB5.txt"

		self.trainingData = FileParser(self.trainFileName, self.dataAttr, self.dataClasif).getOutputData()
		self.developmentData = FileParser(self.devFileName, self.dataAttr, self.dataClasif).getOutputData()
		self.testData = FileParser(self.testFileName, self.dataAttr, self.dataClasif).getOutputData()

		# run program
		self.printInitialization()
		# with open(self.outFileName, 'w') as f:
		# 	self.runTraining(f, self.trainingData)
		# self.runTrial(self.developmentData)
		# self.runClassification(self.testData)


	def sizeInputSpace(self):
		# num choices per attr is two
		return 2**len(self.dataAttr)

	def sizeConceptSpace(self):
		return 2**self.sizeInputSpace()

 	def printInitialization(self):
 		print self.sizeInputSpace()
 		print self.sizeConceptSpace()

 	def populateVS(self):
 		# populate VS using a binary representation of the data
 		# generate all binary strings using a trick with product
		return ["".join(prod) for prod in itertools.product("01", repeat = self.sizeInputSpace())]

	def matchHypot(self, hypot, data):
		# return True if the data matches the hypothesis, False otherwise
		dataIndex = self.getDataIndex(data) # basically a hash to determine which input it is

		# compare classification to hypot's classification, 1 is pos for hypot 
		if data["classification"] == self.posClassification:
			return hypot[dataIndex] == 1
		else:
			return hypot[dataIndex] == 0

	def getDataIndex(self, data):
		# compute index by encoding values to binary
		i = 0
		for attr in self.dataAttr:
			valIndex = self.dataAttrOpt[attr].find(data[attr]) # value mapped to 0/1
			i = (i << 1) + valIndex # create index bit by bit
		return i

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
		for dataClassified in data:
			print self.hypothesis.classify( dataClassified["dataVect"], self.posClassification, self.negClassification )

partA = LTE()
