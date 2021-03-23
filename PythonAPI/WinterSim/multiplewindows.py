#!/usr/bin/env python

# Copyright (c) 2019 Aptiv
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
An example of client-side bounding boxes with basic car controls.

Controls:

	W			 : throttle
	S			 : brake
	AD			 : steer
	Space		 : hand-brake

	ESC			 : quit
"""

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================

import glob
import os
import sys

try:
	sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
		sys.version_info.major,
		sys.version_info.minor,
		'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
	pass

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import time
start = time.time()
import carla
import weakref
import random
import cv2

try:
	import pygame
	from pygame.locals import K_ESCAPE
	from pygame.locals import K_SPACE
	from pygame.locals import K_a
	from pygame.locals import K_d
	from pygame.locals import K_s
	from pygame.locals import K_w
	from pygame.locals import K_m
except ImportError:
	raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
	import numpy as np
except ImportError:
	raise RuntimeError('cannot import numpy, make sure numpy package is installed')

VIEW_WIDTH = 1920//4
VIEW_HEIGHT = 1080//4
VIEW_FOV = 90


# ==============================================================================
# -- BasicSynchronousClient ----------------------------------------------------
# ==============================================================================


class BasicSynchronousClient(object):
	""" Basic implementation of a synchronous client."""

	def __init__(self):
		self.client = None
		self.world = None
		self.car = None

		# PyGame window camera
		self.camera = None
		self.image = None
		self.capture = True
		self.display = None

		# Front RGB camera
		self.front_rgb_camera_display = None
		self.front_rgb_camera = None
		self.front_rgb_image = None
		self.front_rgb_capture = True

		# Back RGB camera
		self.back_rgb_camera_display = None
		self.back_rgb_camera = None
		self.back_rgb_image = None
		self.back_rgb_capture = True

		# Front Depth camera
		self.front_depth_camera = None
		self.front_depth_display = None
		self.front_depth_image = None
		self.front_depth_capture = True
		self.front_depth = None

		self.counter = 0
		self.pose = []
		self.log = False
		
	def camera_blueprint(self):
		""" Returns camera blueprint."""

		camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
		camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
		camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
		camera_bp.set_attribute('fov', str(VIEW_FOV))
		return camera_bp
		
	def depth_camera_blueprint(self):
		"""Returns camera blueprint. """

		depth_camera_bp = self.world.get_blueprint_library().find('sensor.camera.depth')
		depth_camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
		depth_camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
		depth_camera_bp.set_attribute('fov', str(VIEW_FOV))
		return depth_camera_bp
		

	def set_synchronous_mode(self, synchronous_mode):
		""" synchronous mode. """

		settings = self.world.get_settings()
		settings.synchronous_mode = synchronous_mode
		self.world.apply_settings(settings)

	def setup_car(self):
		"""Spawns actor-vehicle to be controled. """

		car_bp = self.world.get_blueprint_library().filter('model3')[0]
		location = random.choice(self.world.get_map().get_spawn_points())
		self.car = self.world.spawn_actor(car_bp, location)

	def setup_camera(self):
		""" Spawns actor-camera to be used to render view.
		Sets calibration for client-side boxes rendering.
		"""

		camera_transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))
		self.camera = self.world.spawn_actor(self.camera_blueprint(), camera_transform, attach_to=self.car)
		weak_self = weakref.ref(self)
		self.camera.listen(lambda image: weak_self().set_image(weak_self, image))

		calibration = np.identity(3)
		calibration[0, 2] = VIEW_WIDTH / 2.0
		calibration[1, 2] = VIEW_HEIGHT / 2.0
		calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
		self.camera.calibration = calibration

	def setup_front_rgb_camera(self):
		""" Spawns actor-camera to be used to RGB camera view.
		Sets calibration for client-side boxes rendering.
		"""
		camera_transform = carla.Transform(carla.Location(x=2.0, z=1.0), carla.Rotation(pitch=0))
		self.front_rgb_camera = self.world.spawn_actor(self.camera_blueprint(), camera_transform, attach_to=self.car)
		weak_rgb_self = weakref.ref(self)
		self.front_rgb_camera.listen(lambda front_rgb_image: weak_rgb_self().set_front_rgb_image(weak_rgb_self, front_rgb_image))

		calibration = np.identity(3)
		calibration[0, 2] = VIEW_WIDTH / 2.0
		calibration[1, 2] = VIEW_HEIGHT / 2.0
		calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
		self.camera.calibration = calibration


	def setup_back_rgb_camera(self):
		""" Spawns actor-camera to be used to RGB camera view.
		Sets calibration for client-side boxes rendering.
		"""
		camera_transform = carla.Transform(carla.Location(x=-3.5, z=1.5), carla.Rotation(pitch=-10, yaw=180))
		self.back_rgb_camera = self.world.spawn_actor(self.camera_blueprint(), camera_transform, attach_to=self.car)
		weak_back_rgb_self = weakref.ref(self)
		self.back_rgb_camera.listen(lambda back_rgb_image: weak_back_rgb_self().set_back_rgb_image(weak_back_rgb_self, back_rgb_image))

		calibration = np.identity(3)
		calibration[0, 2] = VIEW_WIDTH / 2.0
		calibration[1, 2] = VIEW_HEIGHT / 2.0
		calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
		self.camera.calibration = calibration
		
	def setup_front_depth_camera(self):
		""" Spawns actor-camera to be used to render view.
		Sets calibration for client-side boxes rendering.
		"""

		depth_camera_transform = carla.Transform(carla.Location(x=0, z=2.8), carla.Rotation(pitch=0))
		self.depth_camera = self.world.spawn_actor(self.depth_camera_blueprint(), depth_camera_transform, attach_to=self.car)
		weak_depth_self = weakref.ref(self)
		self.depth_camera.listen(lambda front_depth_image: weak_depth_self().set_front_depth_image(weak_depth_self, front_depth_image))

		calibration = np.identity(3)
		calibration[0, 2] = VIEW_WIDTH / 2.0
		calibration[1, 2] = VIEW_HEIGHT / 2.0
		calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
		self.depth_camera.calibration = calibration

	def control(self, car):
		""" Applies control to main car based on pygame pressed keys.
		Will return True If ESCAPE is hit, otherwise False to end main loop.
		"""

		keys = pygame.key.get_pressed()
		if keys[K_ESCAPE]:
			return True

		control = car.get_control()
		control.throttle = 0
		
		if keys[K_w]:
			control.throttle = 1
			control.reverse = False
		elif keys[K_s]:
			control.throttle = 1
			control.reverse = True
		if keys[K_a]:
			control.steer = max(-1., min(control.steer - 0.05, 0))
		elif keys[K_d]:
			control.steer = min(1., max(control.steer + 0.05, 0))
		else:
			control.steer = 0
		control.hand_brake = keys[K_SPACE]
		if keys[K_m]:
			if self.log:
				self.log = False
				np.savetxt('log/pose.txt',self.pose)
			else:
				self.log = True
			pass

		
		car.apply_control(control)
		return False

	@staticmethod
	def set_image(weak_self, img):
		""" Sets image coming from camera sensor.
		The self.capture flag is a mean of synchronization - once the flag is
		set, next coming image will be stored.
		"""
		self = weak_self()
		if self.capture:
			self.image = img
			self.capture = False

	@staticmethod
	def set_front_rgb_image(weak_self, img):
		self = weak_self()
		if self.front_rgb_capture:
			self.front_rgb_image = img
			self.front_rgb_capture = False

	@staticmethod
	def set_back_rgb_image(weak_self, img):
		self = weak_self()
		if self.back_rgb_capture:
			self.back_rgb_image = img
			self.back_rgb_capture = False

	@staticmethod
	def set_front_depth_image(weak_depth_self, depth_img):
		""" Sets image coming from camera sensor.
		The self.capture flag is a mean of synchronization - once the flag is
		set, next coming image will be stored.
		"""

		self = weak_depth_self()
		if self.front_depth_capture:
			self.front_depth_image = depth_img
			self.front_depth_capture = False

	def render(self, display):
		"""Transforms image from camera sensor and blits it to main pygame display."""

		if self.image is not None:
			array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
			array = np.reshape(array, (self.image.height, self.image.width, 4))
			array = array[:, :, :3]
			array = array[:, :, ::-1]
			surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
			display.blit(surface, (0, 0))

	def front_depth_render(self, front_depth_display):
		if self.front_depth_image is not None:
			i = np.array(self.front_depth_image.raw_data)
			i2 = i.reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))
			i3 = i2[:, :, :3]
			self.front_depth = i3
			cv2.imshow("front_depth_image", self.front_depth)
	
	def front_rgb_camera_render(self, rgb_display):
		if self.front_rgb_image is not None:
			i = np.array(self.front_rgb_image.raw_data)
			i2 = i.reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))
			i3 = i2[:, :, :3]
			cv2.imshow("front RGB camera", i3)

	def back_rgb_camera_render(self, rgb_display):
		if self.back_rgb_image is not None:
			i = np.array(self.back_rgb_image.raw_data)
			i2 = i.reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))
			i3 = i2[:, :, :3]
			cv2.imshow("back RGB camera", i3)

	def log_data(self):
			global start
			freq = 1/(time.time() - start)

	        #sys.stdout.write("\rFrequency:{}Hz		Logging:{}".format(int(freq),self.log))
			#sys.stdout.write("\r{}".format(self.car.get_transform().rotation))

			sys.stdout.flush()
			if self.log:
				name ='log/' + str(self.counter) + '.png'
				self.front_depth_image.save_to_disk(name)
				position = self.car.get_transform()
				pos=None
				pos = (int(self.counter), position.location.x, position.location.y, position.location.z, position.rotation.roll, position.rotation.pitch, position.rotation.yaw)
				self.pose.append(pos)
				self.counter += 1
			start = time.time()
		
	def game_loop(self):
		"""Main program loop."""

		try:
			pygame.init()

			self.client = carla.Client('127.0.0.1', 2000)
			self.client.set_timeout(2.0)
			self.world = self.client.get_world()

			self.setup_car()
			self.setup_camera()

			self.setup_front_rgb_camera()
			self.setup_back_rgb_camera()

			self.setup_front_depth_camera()
			#self.setup_back_depth_camera()

			self.display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)

			self.front_depth_display = cv2.namedWindow('front_depth_image')
			self.front_rgb_camera_display = cv2.namedWindow('front RGB camera')
			self.back_rgb_camera_display = cv2.namedWindow('back RGB camera')

			pygame_clock = pygame.time.Clock()

			self.set_synchronous_mode(True)
			vehicles = self.world.get_actors().filter('vehicle.*')

			while True:
				self.world.tick()
				self.capture = True
				self.front_depth_capture = True
				self.front_rgb_capture = True
				self.back_rgb_capture = True
				pygame_clock.tick_busy_loop(30)
				self.render(self.display)
				pygame.display.flip()
				pygame.event.pump()

				# front sensors
				self.front_depth_render(self.front_depth_display) 			# render fronnt depth camera to separate window
				self.front_rgb_camera_render(self.front_rgb_camera_display)	# render front RGB camera to separate window

				# back sensors
				self.back_rgb_camera_render(self.back_rgb_camera_display)	# Render back RGB camera to separate window

				self.log_data()
				cv2.waitKey(1)
				if self.control(self.car):
					return
				
		#except Exception as e: print(e)
		finally:
			self.set_synchronous_mode(False)
			self.camera.destroy()
			self.front_rgb_camera.destroy()
			self.depth_camera.destroy()
			self.car.destroy()
			pygame.quit()
			cv2.destroyAllWindows()

# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
	"""Initializes the client-side bounding box demo."""
	try:
		client = BasicSynchronousClient()
		client.game_loop()
	finally:
		print('EXIT')


if __name__ == '__main__':
	main()