
import math as m
import numpy as np
import operator as op

from OpenGL.GL import *
from OpenGL.GLU import *

import pycuda.autoinit
import pycuda.driver as cuda_driver
import pycuda.gl as cuda_gl
import pycuda.gpuarray as cuda_garr

from pycuda.compiler import SourceModule

import shaders as sh

# FIXME: get this info from system
sizeOfFloat = 4

class Solid :
	def __init__( self , beg , end , prec ) :
		self.gdata = 0
		
		self.beg  = beg
		self.size = map( op.sub , end , beg )
		self.prec = prec

		print beg , end , self.size

		self.gen_hdata()

		self.set_round_drill( 16 )

	def gen_data( self ) :
		self.gen_hdata()
		self.gen_gdata()

	def set_size( self , size ) :
		self.size = size
		self.gen_data()

	def set_prec( self , prec ) :
		self.prec = prec
		self.gen_data()

	def get_triangles_count( self ) :
		return self.prec[0] * self.prec[1] * 3 * 2

	def get_buff_len( self ) :
		return self.get_triangles_count() * 3

	def get_buff_bytes( self ) :
		return self.get_buff_len() * sizeOfFloat 

	def get_scale( self ) :
		return float(self.size[0]) / self.prec[0] , float(self.size[1]) / self.prec[1]

	def gen_hdata( self ) :
		self.hdata = np.zeros( (2,self.prec[0], self.prec[1], 3, 3) , np.float32 )

		sx , sy = self.get_scale()

		for x in xrange(self.prec[0]-1) :
			for y in xrange(self.prec[1]-1) :
				self.hdata[0,x,y,0,0] = x * sx
				self.hdata[0,x,y,0,1] = self.size[2]
				self.hdata[0,x,y,0,2] = y * sy
				self.hdata[0,x,y,1,0] = (x+1) * sx
				self.hdata[0,x,y,1,1] = self.size[2]
				self.hdata[0,x,y,1,2] = y * sy
				self.hdata[0,x,y,2,0] = x * sx
				self.hdata[0,x,y,2,1] = self.size[2]
				self.hdata[0,x,y,2,2] = (y+1) * sy

		for x in xrange(1,self.prec[0]) :
			for y in xrange(1,self.prec[1]) :
				self.hdata[1,x-1,y-1,0,0] = x * sx
				self.hdata[1,x-1,y-1,0,1] = self.size[2]
				self.hdata[1,x-1,y-1,0,2] = y * sy
				self.hdata[1,x-1,y-1,1,0] = (x-1) * sx
				self.hdata[1,x-1,y-1,1,1] = self.size[2]
				self.hdata[1,x-1,y-1,1,2] = y * sy
				self.hdata[1,x-1,y-1,2,0] = x * sx
				self.hdata[1,x-1,y-1,2,1] = self.size[2]
				self.hdata[1,x-1,y-1,2,2] = (y-1) * sy

	def gen_gdata( self ) :
		if not self.gdata :
			self.gdata , self.normals = glGenBuffers(2)

		glBindBuffer( GL_ARRAY_BUFFER , self.gdata )
		glBufferData( GL_ARRAY_BUFFER , self.get_buff_bytes() , self.hdata , GL_STREAM_DRAW )
		glBindBuffer( GL_ARRAY_BUFFER , 0 )

		glBindBuffer( GL_ARRAY_BUFFER , self.normals )
		glBufferData( GL_ARRAY_BUFFER , self.get_buff_bytes() , np.array( [0.0,1.0,0.0] * self.get_buff_len() , np.float32 )  , GL_STREAM_DRAW )
		glBindBuffer( GL_ARRAY_BUFFER , 0 )

	def gfx_init( self ) :
		self.gen_gdata()
		self.cuda_init()

	def draw( self ) :
		if not self.gdata : return

#        glDisable( GL_CULL_FACE )

		glFrontFace(GL_CW);

		glColor3f( .5 , .5 , .5 )

		glEnableClientState( GL_VERTEX_ARRAY )
		glEnableClientState( GL_NORMAL_ARRAY )
#        glEnableClientState( GL_COLOR_ARRAY )

		glBindBuffer( GL_ARRAY_BUFFER , self.gdata )
		glVertexPointer( 3 , GL_FLOAT , 0 , None )
		glBindBuffer( GL_ARRAY_BUFFER , 0 )

		glBindBuffer( GL_ARRAY_BUFFER , self.normals )
		glNormalPointer(     GL_FLOAT , 0 , None )
#        glColorPointer( 3 , GL_FLOAT , 0 , None )
		glBindBuffer( GL_ARRAY_BUFFER , 0 )

		glDrawArrays( GL_TRIANGLES , 0 , self.get_triangles_count() )

		glDisableClientState( GL_VERTEX_ARRAY )
		glDisableClientState( GL_NORMAL_ARRAY )
#        glDisableClientState( GL_COLOR_ARRAY )

#        glEnable(GL_LIGHTING)

#        glEnable( GL_CULL_FACE )

	def cuda_init( self ) :
		mod = cuda_driver.module_from_file( 'solid_kernel.cubin' )

		self.cut = mod.get_function("cut_x")
		self.cut.prepare( "PPPifiiiii" )

	def set_cut( self , pos ) :
		self.pos = pos

	def next_cut( self , pos ) :
		if not self.gdata or not pos :
			self.pos = pos
			return

		sx , sy = self.get_scale()
		nx , ny = self.hdrill.shape

		grid = map( int , ( m.ceil(nx/22.0) , m.ceil(ny/22.0) ) )
		block = ( min(nx,22) , min(ny,22) , 1 )

		cdata = cuda_gl.BufferObject( long( self.gdata ) )
		norms = cuda_gl.BufferObject( long( self.normals) )
		hmap = cdata.map()
		nmap = norms.map()

		dx = pos[0] - self.pos[0]
		dz = pos[2] - self.pos[2]

		print np.array(self.pos) , ' -> ' , np.array(pos)
		print np.array(self.pos) / sx , ' -> ' , np.array(pos) / sy
		print ( dx , dz )

		#
		# perform one cut
		#
		if dx == 0.0 and dz == 0.0 :
			self.cut.prepared_call( grid , block ,
					hmap.device_ptr() ,
					nmap.device_ptr() ,
					self.cdrill ,
					np.int32( pos[0] / sx + .5 ) , np.float32(pos[1]) , np.int32( pos[2] / sy + .5 ) ,
					np.int32(self.prec[0]) , np.int32(self.prec[1]) ,
					np.int32(nx) , np.int32(ny) )

		#
		# cutting by x axis
		#
		elif m.fabs(dx) > m.fabs(dz) :

			x = np.  int32( float(self.pos[0]) / float(sx) )
			ex= np.  int32( float(     pos[0]) / float(sx) )
			y = np.float64( self.pos[1] )
			z = np.float64( self.pos[2] )

			dz = dz / float(ex - x)
			dy = (pos[1] - self.pos[1]) / float(dx)
			dx = m.copysign( 1 , dx )

			print 'c' , x , y , z
			print 'd' , dx , dy , dz 

			while True :
				if dx < 0 :
					if x <= ex : break
				else :
					if x >= ex : break

				print '1',x , y , z
				print '2',x * sx , y , z
				print '3',np.int32( x ) , np.float32(y) , np.int32( z / float(sy) + .5 )
				self.cut.prepared_call( grid , block ,
						hmap.device_ptr() ,
						nmap.device_ptr() ,
						self.cdrill ,
						np.int32(x) , np.float32(y) , np.int32( z / float(sy) + .5 ) ,
						np.int32(self.prec[0]) , np.int32(self.prec[1]) ,
						np.int32(nx) , np.int32(ny) )
				x += dx 
				y += dy
				z += dz

		#
		# cutting by z axis
		#
		else :
			x = np.float64( self.pos[0] )
			y = np.float64( self.pos[1] )
			z = np.  int32( float(self.pos[2]) / float(sy) )
			ez= np.  int32( float(     pos[2]) / float(sy) )

			dx = dx / float(ez - z)
			dy = (pos[1] - self.pos[1]) / float(dz)
			dz = m.copysign( 1 , dz )

			print 'c' , x , y , z
			print 'd' , dx , dy , dz 

			while True :
				if dz < 0 :
					if z <= ez : break
				else :
					if z >= ez : break

				self.cut.prepared_call( grid , block ,
						hmap.device_ptr() ,
						nmap.device_ptr() ,
						self.cdrill ,
						np.int32( x / float(sx) + .5 ) , np.float32(y) , np.int32( z ) ,
						np.int32(self.prec[0]) , np.int32(self.prec[1]) ,
						np.int32(nx) , np.int32(ny) )
				x += dx 
				y += dy
				z += dz

		self.pos = pos

		cuda_driver.Context.synchronize()

		hmap.unmap()
		nmap.unmap()
		cdata.unregister()
		norms.unregister()

	def set_flat_drill( self , size ) :
		sx , sy = self.get_scale()
		nx , ny = int(size / sx + .5) , int(size / sy + .5)

		print 'Setting flat drill:'
		print size
		print sx , sy 
		print nx , ny

		self.hdrill = np.zeros( (nx,ny) , np.float32 )
		self.cdrill = cuda_driver.mem_alloc( self.hdrill.nbytes )
		cuda_driver.memcpy_htod( self.cdrill , self.hdrill )

	def set_round_drill( self , size ) :
		sx , sy = self.get_scale()
		nx , ny = int(size / sx + .5) , int(size / sy + .5)

		print 'Setting round drill:'
		print size
		print sx , sy 
		print nx , ny

		self.hdrill = np.zeros( (nx,ny) , np.float32 )

		size /= 2.0

		for x in range(nx) :
			for y in range(ny) :
				fx = (x-int(nx/2+.5)) * sx
				fy = (y-int(nx/2+.5)) * sy 
				ts = size*size - fx*fx - fy*fy
				self.hdrill[x,y] = -m.sqrt( ts ) if ts > 0 else 0

		self.cdrill = cuda_driver.mem_alloc( self.hdrill.nbytes )
		cuda_driver.memcpy_htod( self.cdrill , self.hdrill )

