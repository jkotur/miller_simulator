
import re 

import numpy as np
import operator as op

class Parser :
	def __init__( self , path , off ) :
		self.pos = []
		self.off = off
		self.i = 0

		with open(path) as f :
			self.read_file(f)

	def read_file( self , f ) :
		for l in f :
			self.pos.append( map( op.add , map( np.float32 , re.match( "N\d+G01X(.*)Y(.*)Z(.*)" , l ).groups() ) , self.off ) )
			self.pos[-1][1] , self.pos[-1][2] = self.pos[-1][2] , self.pos[-1][1]

	def reset( self ) :
		self.i = 0

	def next( self ) :
		if self.i < len(self.pos) :
			self.i += 1
			return self.pos[self.i-1] 
		else :
			return None

	def curr( self ) :
		return self.pos[self.i] if self.i <= len(self.pos) else None
