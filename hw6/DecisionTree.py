# Spencer Barton
# 10-601 Spring 2011
# HW 6 2/19/14
# DecisionTree.py

import math, sys, csv

class DecisionTree(object):
	"""DecisionTree for performing algorithm"""
	def __init__(self, arg):
		super(DecisionTree, self).__init__()
		self.arg = arg
		
#=================================================================
# Binary Node
#=================================================================


class Node(object):
	"""Nodes that make-up tree. Has children or is a leaf."""
	def __init__(self, depth, examples, targetPlus, targetMinus, minEntropy, 
				 targetKey = "Target", attrKey = "Attrs"):
		# constructor args
		self.depth 			= depth 		# int
		self.examples 		= examples 		# list of dict: {Attrs: {Attr: val, ... } , Target: (+,-) }
		self.targetPlus 	= targetPlus 	# label
		self.targetMinus 	= targetMinus 	# label
		self.minEntropy		= minEntropy 	# float, lowest Entropy to branch on
		self.targetKey		= targetKey		# for access into examples
		self.attrKey		= attrKey		# for access into examples
		
		# examples meta data
		self.nPlus  = 0
		self.nMinus = 0
		self.label  = 0 # TODO vote nPlus vs nMinus

		# generateChildren can return all None if no children generated
		(splitAttr, children) = self._generateChildren()
		self.splitAttr		  = splitAttr 	# attribute for this node's split,
		self.children 		  = children	# dict mapping attr val to correct branch
											# children are also Node objs

	#=================================================================
	# Generating children and helpers
	#=================================================================

	def _entropy(self, examples):
		"""calculate entropy of examples: H(Y)
			returns float"""
		nElems = float(len(examples))
		nPlusElems  = sum(1 for e in examples if (e[self.targetKey] == self.targetPlus) )
		nMinusElems = nElems - nPlusElems
		probP = nPlusElems / nElems
		probM = nMinusElems / nElems
		return -probP * math.log(probP, 2) - probM * math.log(probM, 2)

	def _condEntropy(self, attr):
		"""calculate conditional entropy of examples: H(Y|A)
			returns float"""
		nElems = float(len(self.examples))

		splitExamples = self._split(attr)

		# calculate entropy conditioned on attr values
		condEntropy = 0.0
		for val in splitExamples:
			probVal = len(splitExamples[val]) / nElems
			entropyCondOnVal = self._entropy(splitExamples[val])
			condEntropy += probVal * entropyCondOnVal

		return condEntropy

	def _mutualInfo(self, attr):
		"""determine the mutual information between the examples and the given attribute.
			I(Y;A) = H(Y) - H(Y|A) where Y is examples (data) and A is attribute. 
			returns float"""
		return self._entropy(self.examples) - self.condEntropy(attr)

	def _split(self, attr):
		"""split the examples on an attribute's values (v1, v2, ...)
			returns {v1 : examples_v1, v2 : examples_v2, ...} """
		# split examples by attr values
		splitExamples = {}
		for ex in self.examples:
			val = ex[self.attrKey][attr]
			# add val as key in split examples
			if val not in splitExamples:
				splitExamples[val] = []
			# add example to split
			splitExamples[val].append(ex)

		return splitExamples

	def _generateChildren(self):
		"""generate child nodes based on maximizing information gain.
			Returns (splitAttr, children) whose values can be None if no split"""
		# no children generated if at max depth or no more training examples or no more attributes
		if ( (depth == 0) or (len(self.examples) == 0) or 
			 (len(self.examples[0][self.attrKey]) == 0) ):
			return (None, None)

		# find attribute to branch on
		maxMutualInfo = 0.0 # mutual info is pos for these cases
		maxAttr = None
		for attr in self.examples[0][self.attrKey]:
			mutualInfo = self._mutualInfo(attr)
			# update maximum
			if mutualInfo >= maxMutualInfo:
				maxMutualInfo = mutualInfo
				maxAttr = attr
		splitAttr = maxAttr

		# if max entropy is not larger then minEntropy then don't branch
		if maxMutualInfo < self.minEntropy:
			return (None, None)

		# split on attr values and return new examples
		examplesByVal = self._split(splitAttr)
		# generate children
		children = {}
		for val in examplesByVal:
			examples = examplesByVal[val]
			children[val] = Node(self.depth - 1, examples, self.targetPlus, 
								 self.targetMinus, self.minEntropy)
		
		return (splitAttr, children)

	#=================================================================
	# Other methods
	#=================================================================

	def isLeaf(self):
		"isLeaf if children are non-existant"
		return (self.lChild == None) and (self.rChild == None)

	def classify(self, testExample):
		"""return the pos/neg classification if leaf, 
			otherwise branch to correct child for classification"""
		pass

	def printTree(self):
		"nice print option for displaying whole tree"
		pass

	def __str__(self):
		"print label"
		return self.label
