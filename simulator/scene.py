
import sys
import time

import numpy as np
import numpy.linalg as la
import transformations as tr

from OpenGL.GL import *
from OpenGL.GLU import *

import math as m

if sys.platform.startswith('win'):
    timer = time.clock
else:
    timer = time.time

from camera import Camera
from robot import Robot
from plane import Plane
from solid import Solid
from parser import Parser

class Scene :
	def __init__( self , fovy , ratio , near , far , robot_files ) :
		self.fovy = fovy
		self.near = near 
		self.far = far
		self.ratio = ratio

		self.running = False
		self.speed = .5

		self.camera = Camera( ( 150 , 550 , -50 ) , ( 150 , 300 , 100 ) , ( 0 , 1 , 0 ) )
		self.plane  = Plane( (2,2) )
		self.solid  = Solid( (-150,-150,-150) , (150,150,150) , (100,100) )
		self.robot  = Robot( robot_files )
		self.load_path( 'data/t1.k16' )

		self.solid.set_cut( self.parser.next() )

		self.x = 0.0

		self.last_time = timer()
		self.ntime = int(self.last_time) + 1

		self.plane_alpha = 65.0 / 180.0 * m.pi

		self._make_plane_matrix()

	def set_speed( self , s ) :
		self.speed = s

	def _make_plane_matrix( self ) :
		r = tr.rotation_matrix( self.plane_alpha , (0,0,1) )
		s = tr.scale_matrix( 1 )
		t = tr.translation_matrix( (-1.25,.7,.05) )

		self.m = np.dot( np.dot( t , s ) , r )
		self.im = la.inv( self.m )
		self.im[3] = [ 0 , 0 , 0 , 1 ]

	def gfx_init( self ) :
		self._update_proj()
		self._set_lights()

		glEnable( GL_DEPTH_TEST )
		glEnable( GL_NORMALIZE )
		glEnable( GL_CULL_FACE )
		glEnable( GL_COLOR_MATERIAL )
		glColorMaterial( GL_FRONT_AND_BACK , GL_AMBIENT_AND_DIFFUSE )

		self.solid.gfx_init()

	def draw( self ) :
		self.time = timer()

		dt = self.time - self.last_time

		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()

		self.camera.look()

		self._draw_scene()

		self.robot.update( dt )

		if self.running and self.time > self.ntime :
			self.solid.next_cut( self.parser.next() )
			self.ntime = self.time + self.speed

		self.x+=dt*.3

		self.last_time = self.time

	def _draw_scene( self ) :
		pos = np.dot( self.m , np.array( [ m.sin(self.x*7)*m.cos(self.x/3.0) , 0 , m.cos(self.x*5) , 1 ] ) )
		nrm = np.dot( self.m , np.array( [      0        ,-1 ,      0        , 0 ] ) )

#        self.robot.resolve( pos , nrm )

		glPushMatrix();
		glScalef(100,100,100)

		glClearStencil(0);
		glClear(GL_DEPTH_BUFFER_BIT|GL_STENCIL_BUFFER_BIT);

		glDisable(GL_DEPTH_TEST)
		glEnable(GL_STENCIL_TEST)
		glStencilFunc(GL_ALWAYS, 1, 1)
		glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE)

		glColorMask(0,0,0,0);
		glFrontFace(GL_CCW);
#        self.plane.draw( self.m )

		glEnable(GL_DEPTH_TEST)

		glColorMask(1,1,1,1);
		glStencilFunc(GL_EQUAL, 1, 1);
		glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);

		glPushMatrix()
		glMultTransposeMatrixf( self.m )
		glScalef(1,-1,1)
		glMultTransposeMatrixf( self.im )

		glFrontFace(GL_CW);
#        self.robot.draw()

		glPopMatrix();
		glFrontFace(GL_CCW);

		glDisable(GL_STENCIL_TEST)

		glEnable( GL_BLEND )
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

		glColor4f(.7,.7,.7,.85)

#        self.plane.draw( self.m )

		glDisable( GL_BLEND )

#        self.robot.draw()
		
		glPopMatrix()

		self.solid.draw()

	def _update_proj( self ) :
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		gluPerspective( self.fovy , self.ratio , self.near , self.far )
		glMatrixMode(GL_MODELVIEW)

	def _set_lights( self ) :
		glEnable(GL_LIGHTING);
		glLightfv(GL_LIGHT0, GL_AMBIENT, [ 0.2 , 0.2 , 0.2 ] );
		glLightfv(GL_LIGHT0, GL_DIFFUSE, [ 0.9 , 0.9 , 0.9 ] );
		glLightfv(GL_LIGHT0, GL_SPECULAR,[ 0.3 , 0.3 , 0.3 ] );
		glLightfv(GL_LIGHT0, GL_POSITION, [ 0 , 200 , 0 ] );
		glEnable(GL_LIGHT0); 
						 
	def set_fov( self , fov ) :
		self.fov = fov
		self._update_proj()

	def set_near( self , near ) :
		self.near = near
		self._update_proj()

	def set_ratio( self , ratio ) :
		self.ratio = ratio
		self._update_proj()

	def set_screen_size( self , w , h ) :
		self.width  = w 
		self.height = h
		self.set_ratio( float(w)/float(h) )

	def mouse_move( self , df ) :
		self.camera.rot( *map( lambda x : -x*.2 , df ) )

	def key_pressed( self , mv ) :
		self.camera.move( *map( lambda x : x*25 , mv ) )

	def set_flat_drill( self , s ) :
		self.solid.set_flat_drill( s )

	def set_round_drill( self , s ) :
		self.solid.set_round_drill( s )

	def sim_run( self ) :
		self.running = True

	def sim_stop( self ) :
		self.running = False

	def reset( self ) :
		self.parser.set_off( self.solid.newbeg )
		self.parser.reset()
		self.solid.reset()
		self.reset_drill( self.parser.get_drill() )
		return self.parser.get_drill() 

	def load_path( self , filename ) :
		self.parser = Parser( filename , self.solid.newbeg )
		self.reset_drill( self.parser.get_drill() )
		return self.parser.get_drill() 
	
	def reset_drill( self , drill ) :
		if drill[0] == Parser.FLAT :
			self.solid.set_flat_drill( drill[1] )
		elif drill[0] == Parser.ROUND :
			self.solid.set_round_drill( drill[1] )

	def set_precision( self , prec ) :
		self.solid.set_prec( (prec,prec) )

	def set_size( self , beg , end ) :
		self.solid.set_size( beg , end )

