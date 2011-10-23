import sys

import pygtk
pygtk.require('2.0')
import gtk

import operator as op

from OpenGL.GL import *

from glwidget import GLDrawingArea

from scene import Scene
from parser import Parser

ui_file = "c3sim.ui"

meshes = [ 'data/mesh{0}.mesh'.format(i) for i in range(1,7) ]

class App(object):
	"""Application main class"""

	def __init__(self):

		self.move = [0,0,0]

		self.dirskeys = ( ( ['w'] , ['s'] ) , ( ['a'] , ['d'] ) , ( ['e'] , ['q'] ) )

		for d in self.dirskeys :
			for e in d :
				for i in range(len(e)) : e[i] = ( gtk.gdk.unicode_to_keyval(ord(e[i])) , False )

		self.near = 1
		self.far = 1000
		self.fov  = 60

		builder = gtk.Builder()
		builder.add_from_file(ui_file)

		glconfig = self.init_glext()

		self.drawing_area = GLDrawingArea(glconfig)
		self.drawing_area.set_events( gtk.gdk.BUTTON_PRESS_MASK | gtk.gdk.BUTTON_RELEASE_MASK | gtk.gdk.BUTTON3_MOTION_MASK )
		self.drawing_area.set_size_request(640,480)

		builder.get_object("vbox1").pack_start(self.drawing_area)
		self.tb_run = builder.get_object("tb_run")
		self.sp_xb = builder.get_object("sp_xb")
		self.sp_xe = builder.get_object("sp_xe")
		self.sp_yb = builder.get_object("sp_yb")
		self.sp_ye = builder.get_object("sp_ye")
		self.sp_zb = builder.get_object("sp_zb")
		self.sp_ze = builder.get_object("sp_ze")

		self.sp_ms = builder.get_object("sp_miller_size")
		self.cb_m  = builder.get_object("cb_miller")

		win_main = builder.get_object("win_main")

		win_main.set_events( gtk.gdk.KEY_PRESS_MASK | gtk.gdk.KEY_RELEASE_MASK )

		self.nosetdrill = False

		win_main.connect('key-press-event'  , self._on_key_pressed  )
		win_main.connect('key-release-event', self._on_key_released )

		win_main.show_all()

		width = self.drawing_area.allocation.width
		height = self.drawing_area.allocation.height
		ratio = float(width)/float(height)

		self.scene = Scene( self.fov , ratio , self.near , self.far , meshes )
		self.scene.set_speed( builder.get_object("sp_spd").get_value() )

		self.drawing_area.add( self.scene )

		builder.connect_signals(self)

		self.drawing_area.connect('motion_notify_event',self._on_mouse_motion)
		self.drawing_area.connect('button_press_event',self._on_button_pressed)
		self.drawing_area.connect('configure_event',self._on_reshape)
		self.drawing_area.connect_after('expose_event',self._after_draw)

		gtk.timeout_add( 1 , self._refresh )

	def _refresh( self ) :
		self.drawing_area.queue_draw()
		return True

	def _after_draw( self , widget , data=None ) :
		self.update_statusbar()

	def update_statusbar( self ) :
		pass

	def _on_reshape( self , widget , data=None ) :
		width = self.drawing_area.allocation.width
		height = self.drawing_area.allocation.height

		ratio = float(width)/float(height)

		self.scene.set_screen_size( width , height )

	def _on_button_pressed( self , widget , data=None ) :
		if data.button == 3 :
			self.mouse_pos = data.x , data.y
		self.drawing_area.queue_draw()

	def _on_mouse_motion( self , widget , data=None ) :
		diff = map( op.sub , self.mouse_pos , (data.x , data.y) )

		self.scene.mouse_move( diff )

		self.mouse_pos = data.x , data.y
		self.drawing_area.queue_draw()

#        gtk.gdk.Keymap

	def _on_key_pressed( self , widget , data=None ) :
		if not any(self.move) :
			gtk.timeout_add( 20 , self._move_callback )

		for i in range(len(self.dirskeys)) :
			if (data.keyval,False) in self.dirskeys[i][0] :
				self.dirskeys[i][0][ self.dirskeys[i][0].index( (data.keyval,False) ) ] = (data.keyval,True)
				self.move[i]+= 1
			elif (data.keyval,False) in self.dirskeys[i][1] :
				self.dirskeys[i][1][ self.dirskeys[i][1].index( (data.keyval,False) ) ] = (data.keyval,True)
				self.move[i]-= 1

	
	def _on_key_released( self , widget , data=None ) :
		for i in range(len(self.dirskeys)) :
			if (data.keyval,True) in self.dirskeys[i][0] :
				self.dirskeys[i][0][ self.dirskeys[i][0].index( (data.keyval,True) ) ] = (data.keyval,False)
				self.move[i]-= 1
			elif (data.keyval,True) in self.dirskeys[i][1] :
				self.dirskeys[i][1][ self.dirskeys[i][1].index( (data.keyval,True) ) ] = (data.keyval,False)
				self.move[i]+= 1

	def _move_callback( self ) :
		self.scene.key_pressed( self.move )
		self.drawing_area.queue_draw()
		return any(self.move)

	def on_run_pause( self , wdg , data =None ) :
		if self.scene.running :
			self.scene.sim_stop()
		else :
			self.scene.sim_run()

	def on_reset( self , wdg , data=None ) :
		drill = self.scene.reset()
		self.tb_run.set_active(False)
		self.set_drill( drill )

	def on_load( self , wdg , data=None ) :
		self.set_drill( self.scene.load_path( wdg.get_filename() ) )

	def set_drill( self , drill ) :
		self.nosetdrill = True
		print drill
		self.cb_m.set_active( drill[0] )
		self.sp_ms.set_value( drill[1] )
		self.nosetdrill = False

	def on_speed_changed( self , wdg , data=None ) :
		self.scene.set_speed( wdg.get_value() )

	def on_prec_changed( self , wdg , data=None ) :
		self.scene.set_precision( wdg.get_value_as_int() )

	def on_size_changed( self , wdg , data=None ) :
		self.scene.set_size(
				( self.sp_xb.get_value()
				, self.sp_yb.get_value()
				, self.sp_zb.get_value() ) ,
				( self.sp_xe.get_value() 
				, self.sp_ye.get_value() 
				, self.sp_ze.get_value() ) )

	def on_miller_changed( self , wdg , data=None ) :
		if not self.nosetdrill :
			self.scene.reset_drill( (Parser.FLAT if self.cb_m.get_active() == 0 else Parser.ROUND , self.sp_ms.get_value_as_int() ) )

	def init_glext(self):
		display_mode = (
				gtk.gdkgl.MODE_RGB    |
				gtk.gdkgl.MODE_DEPTH  |
				gtk.gdkgl.MODE_STENCIL|
				gtk.gdkgl.MODE_DOUBLE )
		try:
			glconfig = gtk.gdkgl.Config(mode=display_mode)
		except gtk.gdkgl.NoMatches:
			display_mode &= ~gtk.gdkgl.MODE_DOUBLE
			glconfig = gtk.gdkgl.Config(mode=display_mode)

		return glconfig

	def on_win_main_destroy(self,widget,data=None):
		gtk.main_quit()
		 
	def on_but_quit_clicked(self,widget,data=None):
		gtk.main_quit()

if __name__ == '__main__':
	app = App()
	gtk.main()

