
import re 

import numpy as np
import operator as op

class Parser :
	FLAT , ROUND , UNKNOWN = range(3)

	def __init__( self , path , off = (0,0,0) ) :
		self.pos = []
		self.i = 0

		self.set_off( off )

		self.get_type( path )
		with open(path) as f :
			self.read_file(f)

	def get_drill( self ) :
		return self.type , self.size

	def set_off( self , off ) :
		self.off = off

	def get_type( self , filename ) :
		try :
			s = re.match(".*\.([fk])([0-9]+)",filename)
			if s.group(1) == 'f' :
				self.type = Parser.FLAT
			elif s.group(1) == 'k' :
				self.type = Parser.ROUND 
			self.size = int( s.group(2) )
		except Exception as e :
			print 'Cannot prase file extension:' , e
			self.type = Parser.UNKNOWN
			self.size = 0

	def read_file( self , f ) :
		for l in f :
			self.pos.append( map( np.float32 , re.match( "N\d+G01X(.*)Y(.*)Z(.*)" , l ).groups() ) )
			self.pos[-1][1] , self.pos[-1][2] = self.pos[-1][2] , self.pos[-1][1]

	def reset( self ) :
		self.i = 0

	def next( self ) :
		if self.i < len(self.pos) :
			self.i += 1
			return map( op.sub , self.pos[self.i-1] , self.off )
		else :
			return None

	def curr( self ) :
		return map( op.sub , self.pos[self.i] , self.off ) if self.i <= len(self.pos) else None
