# Spencer Barton
# 10-601 Spring 2011
# HW 6 2/19/14
# DecisionTree.py

import math, sys, csv

FLOAT_EPSILON = 0.0001

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
	def __init__(self, maxDepth, examples, targetPlus, targetMinus, minEntropy, 
				 depth = 0, nodeAttr = None, nodeVal = None, 
				 targetKey = "Target", attrKey = "Attrs"):
		# constructor args
		self.depth 			= depth 		# int
		self.maxDepth 		= maxDepth 		# int
		self.examples 		= examples 		# list of dict: {Attrs: {Attr: val, ... } , Target: (+,-) }
		self.targetPlus 	= targetPlus 	# label
		self.targetMinus 	= targetMinus 	# label
		self.minEntropy		= minEntropy 	# float, lowest Entropy to branch on
		self.nodeAttr		= nodeAttr 		# attribute of this node
		self.nodeVal 		= nodeVal 		# value of attr for branch
		self.targetKey		= targetKey		# for access into examples
		self.attrKey		= attrKey		# for access into examples
		
		# examples meta data
		nElems = len(examples)
		nPlusElems  = sum([ 1 for e in examples if (e[self.targetKey] == self.targetPlus) ])
		nMinusElems = nElems - nPlusElems
		
		# determine classification
		self.nPlus  = nPlusElems
		self.nMinus = nMinusElems
		if nPlusElems >= nMinusElems:
			self.classification = self.targetPlus
		else:
			self.classification = self.targetMinus

		# generateChildren can return all None if no children generated
		(branchAttr, children) = self._generateChildren()
		self.branchAttr		   = branchAttr # attribute for this node's split,
		self.children 		   = children	# dict mapping attr val to correct branch
											# children are also Node objs

	#=================================================================
	# Generating children and helpers
	#=================================================================

	def _entropy(self, examples):
		"""calculate entropy of examples: H(Y)
			returns float"""
		nElems = float(len(examples))
		nPlusElems  = sum( [1 for e in examples if (e[self.targetKey] == self.targetPlus)] )
		nMinusElems = nElems - nPlusElems
		probP = nPlusElems / nElems
		probM = nMinusElems / nElems
		# dealing with case of zero prob
		# entropy is 0 as one prob is 0 then the other is 1
		# and log2(1) = 0
		if (probM < FLOAT_EPSILON) or (probP< FLOAT_EPSILON):
			return 0.0
		return -probP * math.log(probP, 2) - probM * math.log(probM, 2)

	def _condEntropy(self, examples, attr):
		"""calculate conditional entropy of examples: H(Y|A)
			returns float"""
		nElems = float(len(examples))

		splitExamples = self._split(examples, attr)

		# calculate entropy conditioned on attr values
		condEntropy = 0.0
		for val in splitExamples:
			probVal = len(splitExamples[val]) / nElems
			entropyCondOnVal = self._entropy(splitExamples[val])
			condEntropy += probVal * entropyCondOnVal

		return condEntropy

	def _mutualInfo(self, examples, attr):
		"""determine the mutual information between the examples and the given attribute.
			I(Y;A) = H(Y) - H(Y|A) where Y is examples (data) and A is attribute. 
			returns float"""
		return self._entropy(examples) - self._condEntropy(examples, attr)

	def _split(self, examples, attr):
		"""split the examples on an attribute's values (v1, v2, ...)
			returns {v1 : examples_v1, v2 : examples_v2, ...} """
		# split examples by attr values
		splitExamples = {}
		for ex in examples:
			val = ex[self.attrKey][attr]
			# add val as key in split examples
			if val not in splitExamples:
				splitExamples[val] = []
			# add example to split by attr val
			splitExamples[val].append(ex)

		return splitExamples

	def _generateChildren(self):
		"""generate child nodes based on maximizing information gain.
			Returns (branchAttr, children) whose values can be None if no split"""
		# no children generated if at max depth or no more training examples or no more attributes
		if ( (self.depth == self.maxDepth) or (len(self.examples) == 0) or 
			 (len(self.examples[0][self.attrKey]) == 0) ):
			return (None, None)

		# find attribute to branch on
		maxMutualInfo = 0.0 # mutual info is pos for these cases
		maxAttr = None
		# know at least one example from above conditional
		for attr in self.examples[0][self.attrKey]:
			mutualInfo = self._mutualInfo(self.examples, attr)
			# update maximum
			if mutualInfo >= maxMutualInfo:
				maxMutualInfo = mutualInfo
				maxAttr = attr
		branchAttr = maxAttr

		# if max entropy is not larger then minEntropy then don't branch
		if maxMutualInfo < self.minEntropy:
			return (None, None)

		# split on attr values and return new examples
		examplesByVal = self._split(self.examples, branchAttr)
		# generate children
		children = {}
		for val in examplesByVal:
			examples = examplesByVal[val]

			children[val] = Node( self.maxDepth, examples, self.targetPlus, 
								  self.targetMinus, self.minEntropy, 
								  depth = self.depth + 1, nodeAttr = branchAttr,
								  nodeVal = val)

		return (branchAttr, children)

	#=================================================================
	# Other methods
	#=================================================================

	def isLeaf(self):
		"isLeaf if children are non-existant"
		return (self.lChild == None) and (self.rChild == None)

	def isRoot(self):
		"isRoot if not nodeAttr"
		return (self.nodeAttr == None) or (self.nodeVal == None)

	def classify(self, testExample):
		"""return the pos/neg classification if leaf, 
			otherwise branch to correct child for classification"""
		# base case at leaf
		if self.isLeaf():
			return self.classification
		
		# branch based on branch attr value
		val = testExample[self.attrKey][self.branchAttr]
		branch = self.children[ val ]
		return branch.classify(testExample)

	def printTree(self):
		"nice print option for displaying whole tree"
		if self.isRoot():
			print "[" + str(self.nPlus) + "+/" + str(self.nMinus) + "-]"
		else:
			# depth > 0 if not root
			strn = ""
			strn += "| " * (self.depth - 1)
			strn += self.nodeAttr
			strn += " = "
			strn += self.nodeVal
			strn += ": [" + str(self.nPlus) + "+/" + str(self.nMinus) + "-]"
			print strn
		# recurse on branches
		if self.children:
			for child in self.children.values():
				child.printTree()

	def __str__(self):
		"print label"
		return self.label


#=======================================
# Run program
#=======================================

# TESTING
with open("example1.csv", 'r') as f:
	# assume header on 1st row
	reader = csv.DictReader(f)

	examples = []
	for ln in reader:
		ex = {}
		ex["Target"] = ln.pop("hit") # destructive
		ex["Attrs"]   = ln
		examples.append(ex)

test = Node(2, examples, "yes", "no", 1.0)
test.printTree()