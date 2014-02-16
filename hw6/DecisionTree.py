# Spencer Barton
# 10-601 Spring 2011
# HW 6 2/19/14
# DecisionTree.py

class DecisionTree(object):
	"""DecisionTree for performing algorithm"""
	def __init__(self, arg):
		super(DecisionTree, self).__init__()
		self.arg = arg
		

class BinaryNode(object):
	"""Binary Nodes that make-up tree. Has 2 children or is a leaf."""
	def __init__(self, depth, targetPlus, targetMinus, ):
		# constructor args
		self.depth 			= depth 		# int
		self.examples 		= examples 		# dict
		self.targetPlus 	= targetPlus 	# label
		self.targetMinus 	= targetMinus 	# label
		self.minEntropy		= minEntropy 	# float, lowest Entropy to branch on
		
		self.nPlus = 0
		self.nMinus = 0
