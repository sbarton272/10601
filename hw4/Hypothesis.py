# Spencer Barton
# 10-601 Spring 2014
# HW 4

class ConjHypothesis(object):
	"""Conjunction Hypothesis instance"""
	def __init__(self, dataAttr, matchAllChar = '?', dataVect = None):
		self.dataAttr = dataAttr
		self.matchAllChar = matchAllChar
		if dataVect:
			self.dataVect = dataVect
		else:
			self.dataVect = dict.fromkeys(dataAttr, None)

	def isNull(self):
		return None in self.dataVect.values()

	def updateHypothesis(self, dataVect):
		# for the Find-S alg.
		# iterate through key
		# if match do nothing
		# if not match update hypothesis
		for attr in self.dataAttr:
			hypotVal = self.dataVect[attr]

			if hypotVal == None:
				self.dataVect[attr] = dataVect[attr]
			
			elif hypotVal != dataVect[attr]:
				self.dataVect[attr] = self.matchAllChar


	def match(self, dataVect):
		# check if given dataVect matches hypot
		if self.isNull():
			return False

		for attr in self.dataAttr:
			if (dataVect[attr] != self.dataVect[attr]) and (self.dataVect[attr] != self.matchAllChar):
				return False
		return True

	def classify(self, dataVect, pos, neg):
		# classify as pos/neg
		if self.match(dataVect):
			return pos
		return neg

	def __str__(self):
		return self.printHypothesis(self.dataAttr)

	def printHypothesis(self, dataAttr):
		if self.isNull():
			return "<null>"
		outStr = "<"
		for attr in dataAttr:
			outStr += self.dataVect[attr] + "\t"
		outStr = outStr[:-1] # remove last tab 
		outStr += ">"
		return outStr