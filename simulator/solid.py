# -*- coding: utf-8 -*-

import math as m
import numpy as np
import operator as op

from OpenGL.GL import *
from OpenGL.GLU import *

import logging

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
		
		self.newbeg = beg
		self.newend = end
		self.newprec = self.prec = prec
		self.drilllen = 100

		self.size = map( op.sub , end , beg )

	def reset( self ) :
		self.pos = None

		self.prec = self.newprec
		self.size = map( op.sub , self.newend , self.newbeg )

		self.gen_data()

	def gen_data( self ) :
		self.gen_gdata()

	def set_size( self, beg , end ) :
		self.newbeg = beg
		self.newend = end

	def set_prec( self , prec ) :
		self.newprec = prec

	def set_drill_len( self , l ) :
		self.drilllen = l

	def get_triangles_count( self ) :
		return self.prec[0] * self.prec[1] * 3 * 2

	def get_buff_len( self ) :
		return self.get_triangles_count() * 3

	def get_buff_bytes( self ) :
		return self.get_buff_len() * sizeOfFloat 

	def get_prec( self ) :
		return self.prec[0] , self.prec[1]
	def get_scale( self ) :
		return float(self.size[0]) / self.prec[0] , float(self.size[2]) / self.prec[1]

	def gen_gdata( self ) :
		if not self.gdata :
			self.gdata , self.normals = glGenBuffers(2)

		px , py = self.get_prec()
		sx , sy = self.get_scale()

		grid  = map( int , ( m.ceil(px/22.0) , m.ceil(py/22.0) ) ) 
		block = ( min(px,22) , min(py,22) , 1 )

		glBindBuffer( GL_ARRAY_BUFFER , self.gdata )
		glBufferData( GL_ARRAY_BUFFER , self.get_buff_bytes() , None , GL_STREAM_DRAW )
		glBindBuffer( GL_ARRAY_BUFFER , 0 )

		print grid , block

		cdata = cuda_gl.BufferObject( long( self.gdata ) )
		hmap = cdata.map()
		self.fill_v.prepared_call( grid , block ,
				hmap.device_ptr() ,
				np.int32(px) , np.int32(py) ,
				np.float32(self.size[1]) ,
				np.float32(sx) , np.float32(sy) )
		hmap.unmap()
		cdata.unregister()

		glBindBuffer( GL_ARRAY_BUFFER , self.normals )
		glBufferData( GL_ARRAY_BUFFER , self.get_buff_bytes() , None  , GL_STREAM_DRAW )
		glBindBuffer( GL_ARRAY_BUFFER , 0 )

		ndata = cuda_gl.BufferObject( long( self.normals ) )
		nmap = ndata.map()
		self.fill_n.prepared_call( grid , block ,
				nmap.device_ptr() ,
				np.int32(px) , np.int32(py) ,
				np.float32(self.size[1]) ,
				np.float32(sx) , np.float32(sy) )
		nmap.unmap()
		ndata.unregister()

	def gfx_init( self ) :
		self.cuda_init()

	def draw( self ) :
		if not self.gdata : return

		glPushMatrix()

		glFrontFace(GL_CW);

		glColor3f( .5 , .5 , .5 )

		glTranslatef( *self.newbeg )

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

		glPopMatrix()

	def cuda_init( self ) :
		mod = cuda_driver.module_from_file( 'solid_kernel.cubin' )

		self.cerr = cuda_driver.mem_alloc( 4 )

		self.cut = mod.get_function("cut_x")
		self.cut.prepare( "PPPifiiiiiffP" )

		self.fill_v = mod.get_function("fill_v")
		self.fill_v.prepare( "Piifff")

		self.fill_n = mod.get_function("fill_n")
		self.fill_n.prepare( "Piifff")

	def set_cut( self , pos ) :
		self.pos = pos

	def next_cut( self , pos ) :
		if not pos :
			return
		if not self.gdata or not self.pos :
			self.pos = pos
			return

		log = logging.getLogger('miller')

		if pos[1] < self.newbeg[1] :
			log.warning('cutting under block')

		sx , sy = self.get_scale()
		nx , ny = self.hdrill.shape

		cdata = cuda_gl.BufferObject( long( self.gdata ) )
		norms = cuda_gl.BufferObject( long( self.normals) )
		hmap = cdata.map()
		nmap = norms.map()

		dx = pos[0] - self.pos[0]
		dz = pos[2] - self.pos[2]

#        print np.array(self.pos) , ' -> ' , np.array(pos)
#        print np.array(self.pos) / sx , ' -> ' , np.array(pos) / sy
#        print ( dx , dz )

		cuda_driver.memcpy_htod( self.cerr , np.int32(0) )
		
		#
		# perform one cut
		#
		if dx == 0.0 and dz == 0.0 :
#            print "0 cut!"
#            if pos[1] < pos[2] : log.warning( 'vertical mill')
			self.cut.prepared_call( self.grid , self.block ,
					hmap.device_ptr() ,
					nmap.device_ptr() ,
					self.cdrill ,
					np.int32( pos[0] / sx + .5 ) , np.float32(pos[1]) , np.int32( pos[2] / sy + .5 ) ,
					np.int32(self.prec[0]) , np.int32(self.prec[1]) ,
					np.int32(nx) , np.int32(ny) ,
					np.float32(self.drilllen) , np.float32(self.drillrad) ,
					self.cerr )

		#
		# cutting by x axis
		#
		if m.fabs(dx) > m.fabs(dz) :
#            print "x cut!"

			x = np.  int32( float(self.pos[0]) / float(sx) )
			ex= np.  int32( float(     pos[0]) / float(sx) )
			y = np.float64( self.pos[1] )
			z = np.float64( self.pos[2] )

#            print 'a' , ex-x , dx  , pos[1] - self.pos[1]

			dz = dz / m.fabs(float(ex - x))
			dy = (pos[1] - self.pos[1]) / m.fabs(float(ex-x))
			dx = m.copysign( 1 , dx )

#            print 'c' , x , y , z
#            print 'd' , dx , dy , dz 

			while True :
				if dx < 0 :
					if x <= ex : break
				else :
					if x >= ex : break

#                print '1',x , y , z
#                print '2',x * sx , y , z
#                print '3',np.int32( x ) , np.float32(y) , np.int32( z / float(sy) + .5 )
				self.cut.prepared_call( self.grid , self.block ,
						hmap.device_ptr() ,
						nmap.device_ptr() ,
						self.cdrill ,
						np.int32(x) , np.float32(y) , np.int32( z / float(sy) + .5 ) ,
						np.int32(self.prec[0]) , np.int32(self.prec[1]) ,
						np.int32(nx) , np.int32(ny) ,
						np.float32(self.drilllen) , np.float32(self.drillrad) ,
						self.cerr )
				x += dx 
				y += dy
				z += dz

		#
		# cutting by z axis
		#
		else :
#            print "z cut!"
			x = np.float64( self.pos[0] )
			y = np.float64( self.pos[1] )
			z = np.  int32( float(self.pos[2]) / float(sy) )
			ez= np.  int32( float(     pos[2]) / float(sy) )

			dx = dx / m.fabs(float(ez - z))
			dy = (pos[1] - self.pos[1]) / m.fabs(float(ez-z))
			dz = m.copysign( 1 , dz )

#            print 'c' , x , y , z
#            print 'd' , dx , dy , dz 

			while True :
				if dz < 0 :
					if z <= ez : break
				else :
					if z >= ez : break

#                print '1',x , y , z
#                print '2',x , y , z * sy
#                print '3',np.int32( x / float(sx) + .5 ) , np.float32(y) , np.int32( z )
				self.cut.prepared_call( self.grid , self.block ,
						hmap.device_ptr() ,
						nmap.device_ptr() ,
						self.cdrill ,
						np.int32( x / float(sx) + .5 ) , np.float32(y) , np.int32( z ) ,
						np.int32(self.prec[0]) , np.int32(self.prec[1]) ,
						np.int32(nx) , np.int32(ny) ,
						np.float32(self.drilllen) , np.float32(self.drillrad) ,
						self.cerr )
				x += dx 
				y += dy
				z += dz

		cuda_driver.Context.synchronize()

		hmap.unmap()
		nmap.unmap()
		cdata.unregister()
		norms.unregister()

		err = np.array( [0] , np.int32 )
		cuda_driver.memcpy_dtoh( err , self.cerr )
		self.parse_err( 0 , err[0] , (pos[1] < self.pos[1] and self.drillflat) )

		self.pos = pos

	def parse_err( self , n , err , flathole ) :
		log = logging.getLogger('miller')
		if err & 1 :
			log.warning(u"Skrawanie poniżej poziomu zero")
		if err & 2 :
			log.warning(u"Skrawanie częścią nieskrawającą")
		if err & 4 and flathole :
			log.warning(u"Wiercenie dziury frezem płaskim")

	def set_flat_drill( self , size ) :
		sx , sy = self.get_scale()
		nx , ny = int(size / sx + .5) , int(size / sy + .5)

		self.drillflat = True

		print 'Setting flat drill:'
		print size
		print sx , sy 
		print nx , ny

		self.hdrill = np.zeros( (nx,ny) , np.float32 )

		size /= 2.0
		for x in range(nx) :
			for y in range(ny) :
				fx = (x-int(nx/2+.5)) * sx
				fy = (y-int(ny/2+.5)) * sy 
				ts = size*size - fx*fx - fy*fy
				self.hdrill[x,y] = 0 if ts > 0 else size*2

		self.drillrad = size

		self.cdrill = cuda_driver.mem_alloc( self.hdrill.nbytes )
		cuda_driver.memcpy_htod( self.cdrill , self.hdrill )

		self.grid = map( int , ( m.ceil(nx/22.0) , m.ceil(ny/22.0) ) )
		self.block = ( min(nx,22) , min(ny,22) , 1 )

		print self.grid 
		print self.block

	def set_round_drill( self , size ) :
		sx , sy = self.get_scale()
		nx , ny = int(size / sx + .5) , int(size / sy + .5)

		self.drillflat = False

		print 'Setting round drill:'
		print size
		print sx , sy 
		print nx , ny

		self.hdrill = np.zeros( (nx,ny) , np.float32 )

		size /= 2.0
		for x in range(nx) :
			for y in range(ny) :
				fx = (x-int(nx/2+.5)) * sx
				fy = (y-int(ny/2+.5)) * sy 
				ts = size*size - fx*fx - fy*fy
				self.hdrill[x,y] = -m.sqrt( ts ) + size if ts > 0 else size*2

		self.drillrad = size

		print self.hdrill
		print self.drillrad

		self.cdrill = cuda_driver.mem_alloc( self.hdrill.nbytes )
		cuda_driver.memcpy_htod( self.cdrill , self.hdrill )

		self.grid = map( int , ( m.ceil(nx/22.0) , m.ceil(ny/22.0) ) )
		self.block = ( min(nx,22) , min(ny,22) , 1 )

		print self.grid 
		print self.block

