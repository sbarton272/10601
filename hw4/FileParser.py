# Spencer Barton
# 10-601 Spring 2014
# HW 4

import re

class FileParser(object):
	"""FileParser reads in input files and converts to dataVect"""
	def __init__(self, fileName, dataAttr, dataClasif, classified = True):
		self.fileName = fileName
		self.dataAttr = dataAttr
		self.dataClasif = dataClasif
		self.classified = classified

		self.outputData = []

		# open file and parse
		with open(fileName, 'r') as f:
			self.updateOuputData(f)

	def updateOuputData(self, f):
		for line in f:
			self.outputData.append( self.convertDataLine(line) )

	def convertDataLine(self, line):
		pairs = re.findall(r"(\S+) (\S+)", line)
		dataVect = dict(pairs)
		if( self.classified ):
			# data file included classification
			classifiedVect = dict()
			classifiedVect['classification'] = dataVect[self.dataClasif]
			del dataVect[self.dataClasif]
			classifiedVect['dataVect'] = dataVect
			return classifiedVect
		else: 
			return dataVect

	def getOutputData(self):
		return self.outputData