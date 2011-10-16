
import numpy as np

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
	def __init__( self , size , prec ) :
		self.gdata = 0
		
		self.size = size
		self.prec = prec

		self.gen_hdata()

		self.set_flat_drill( .4 )

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
				self.hdata[0,x,y,0,1] = self.size[2] * x * sx * y * sy
				self.hdata[0,x,y,0,2] = y * sy
				self.hdata[0,x,y,1,0] = (x+1) * sx
				self.hdata[0,x,y,1,1] = self.size[2] * (x+1) * sx * y * sy
				self.hdata[0,x,y,1,2] = y * sy
				self.hdata[0,x,y,2,0] = x * sx
				self.hdata[0,x,y,2,1] = self.size[2] * x * sx * (y+1) * sy
				self.hdata[0,x,y,2,2] = (y+1) * sy

		for x in xrange(1,self.prec[0]) :
			for y in xrange(1,self.prec[1]) :
				self.hdata[1,x-1,y-1,0,0] = x * sx
				self.hdata[1,x-1,y-1,0,1] = self.size[2] * x * sx * y * sy
				self.hdata[1,x-1,y-1,0,2] = y * sy
				self.hdata[1,x-1,y-1,1,0] = (x-1) * sx
				self.hdata[1,x-1,y-1,1,1] = self.size[2] * (x-1) * sx * y * sy
				self.hdata[1,x-1,y-1,1,2] = y * sy
				self.hdata[1,x-1,y-1,2,0] = x * sx
				self.hdata[1,x-1,y-1,2,1] = self.size[2] * x * sx * (y-1) * sy
				self.hdata[1,x-1,y-1,2,2] = (y-1) * sy

	def gen_gdata( self ) :
		if not self.gdata :
			self.gdata = glGenBuffers(1)

		glBindBuffer( GL_ARRAY_BUFFER , self.gdata )
		glBufferData( GL_ARRAY_BUFFER , self.get_buff_bytes() , self.hdata , GL_STREAM_DRAW )
		glBindBuffer( GL_ARRAY_BUFFER , 0 )

	def gfx_init( self ) :
		self.gen_gdata()
		self.cuda_init()

	def draw( self ) :
		if not self.gdata : return

		glDisable( GL_CULL_FACE )

		glColor4f( 1 , 1 , 1 , 1 )

		glEnableClientState( GL_VERTEX_ARRAY )

		glBindBuffer( GL_ARRAY_BUFFER , self.gdata )
		glVertexPointer( 3 , GL_FLOAT , 0 , None )
		glBindBuffer( GL_ARRAY_BUFFER , 0 )

		glDrawArrays( GL_TRIANGLES , 0 , self.get_triangles_count() )

		glDisableClientState( GL_VERTEX_ARRAY )

		glEnable( GL_CULL_FACE )

	def cuda_init( self ) :
		self.hbounds = np.array( [ 0 , 0 , 0 , 1 , 1 , 1 ] , np.float32 )
		self.gbounds = cuda_driver.mem_alloc( self.hbounds.nbytes )
		cuda_driver.memcpy_htod( self.gbounds , self.hbounds )

		mod = SourceModule(open('solid_kernel.cu').read())

		self.cut = mod.get_function("cut")
		self.cut.prepare( "PPP" )

	def next_cut( self ) :
		if not self.gdata : return

		nx , ny = self.hdrill.shape

		cdata = cuda_gl.BufferObject( long( self.gdata ) )
		cmapping = cdata.map()

		self.cut.prepared_call( (1,1) , (nx,ny,1) ,
				cmapping.device_ptr() ,
				self.cdrill ,
				self.gbounds )

		cuda_driver.Context.synchronize()

		cmapping.unmap()
		cdata.unregister()

	def set_flat_drill( self , size ) :
		sx , sy = self.get_scale()
		nx , ny = size / sx , size / sy

		print size
		print sx , sy 
		print nx , ny

		self.hdrill = np.zeros( (nx,ny) , np.float32 )
		self.hdrill[1,1] = .1;
		self.cdrill = cuda_driver.mem_alloc( self.hdrill.nbytes )
		cuda_driver.memcpy_htod( self.cdrill , self.hdrill )

	def set_round_drill( self , size ) :
		pass

