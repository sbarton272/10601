# Spencer Barton
# 10-601 Spring 2011
# HW 6 2/19/14
# DecisionTree.py

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
	def __init__(self, depth, examples, targetPlus, targetMinus, minEntropy):
		# constructor args
		self.depth 			= depth 		# int
		self.examples 		= examples 		# dict
		self.targetPlus 	= targetPlus 	# label
		self.targetMinus 	= targetMinus 	# label
		self.minEntropy		= minEntropy 	# float, lowest Entropy to branch on
		
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

	def _entropy(self):
		"""calculate entropy of examples: H(Y)
			returns float"""
		pass

	def _condEntropy(self, attr):
		"""calculate conditional entropy of examples: H(Y|A)
			returns float"""
		pass


	def _mutualInfo(self, attr):
		"""determine the mutual information between the examples and the given attribute.
			I(Y;A) = H(Y) - H(Y|A) where Y is examples (data) and A is attribute. 
			returns float"""
		pass

	def _split(self, attr):
		"""split the examples on an attribute's values (v1, v2, ...)
			returns {v1 : examples_v1, v2 : examples_v2, ...} """
		pass

	def _generateChildren(self):
		"generate child nodes based on maximizing information gain"
		pass

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
