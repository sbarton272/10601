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
		self.versionSpace = self.populateVS() # does not include NULL hypot
		self.printEvery = 30

		# files
		self.testFileName = "hw4Data/4Cat-Test.labeled" #sys.argv[1]
		self.trainFileName = "hw4Data/4Cat-train.labeled"

		self.trainingData = FileParser(self.trainFileName, self.dataAttr, self.dataClasif).getOutputData()
		self.testData = FileParser(self.testFileName, self.dataAttr, self.dataClasif, False).getOutputData()

		# run program
		self.printInitialization()
		self.runTraining(self.trainingData) # update VS
		self.runClassification(self.testData)


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

	def getDataIndex(self, data):
		# compute index by encoding values to binary
		i = 0
		for attr in self.dataAttr:
			valIndex = self.dataAttrOpt[attr].index(data[attr]) # value mapped to 0/1
			i = (i << 1) + valIndex # create index bit by bit
		return i

	def matchHypot(self, hypot, dataClassified):
		# return True if the data matches the hypothesis, False otherwise
		dataIndex = self.getDataIndex( dataClassified["dataVect"] ) # basically a hash to determine which input it is

		# compare classification to hypot's classification, 1 is pos for hypot 
		if dataClassified["classification"] == self.posClassification:
			return hypot[dataIndex] == '1'
		else:
			return hypot[dataIndex] == '0'

	def classifyPos(self, hypot, dataVect):
		# return pos or neg classification
		dataIndex = self.getDataIndex(dataVect)
		return hypot[dataIndex] == '1'

	def runTraining(self, data):
		# iter through all hypotheses and eliminate those that do not match the training data
		# modified VS
		newVS = []
		for hypot in self.versionSpace:
			valid = True
			for dataClassified in data:
				# move to next if no match
				if not self.matchHypot(hypot, dataClassified):
					valid = False
					break
			# include hypot in VS if matched every piece of training data
			if valid:
				newVS.append(hypot)
		self.versionSpace = newVS
		print len(newVS)

	def runClassification(self, data):
		# print classification vote with current VS
		for dataVect in data:
			votePos = 0
			voteNeg = 0
			for hypot in self.versionSpace:
				if self.classifyPos(hypot, dataVect):
					votePos += 1
				else:
					voteNeg += 1
			print str(votePos) + ' ' + str(voteNeg)

partA = LTE()
