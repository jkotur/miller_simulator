
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

	def get_len( self ) :
		return len(self.pos)

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
			p = [ 0 ] * 3
			if not re.match( 'N\d+G01' , l ) :
				continue
			l = re.sub( 'N\d+G01' , '' , l )
			lst = -1 
			while len(l) > 2 :
				if   l[0] == 'X' : new = 0
				elif l[0] == 'Y' : new = 1
				elif l[0] == 'Z' : new = 2

				for i in range(lst+1,new) : p[i] = self.pos[-1][i]

				r = r'[XYZ]([\d\.-]*)'
#                print "l'%s' - %d" % ( l , len(l) )
				m = re.match( r , l ).groups()[0]
#                print "m'%s'" % m
				p[new] = np.float32( m )
				l = l[len(m)+1:]
				lst = new

			for i in range(lst+1,3) : p[i] = self.pos[-1][i]

			self.pos.append(p)

		for p in self.pos :
			p[0] = -p[0]
			p[1] , p[2] = p[2] , p[1]

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
