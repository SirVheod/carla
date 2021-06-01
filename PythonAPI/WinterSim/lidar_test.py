#!/usr/bin/env python
#Wintersim Carla script
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

import carla
import random
import time
start = time.time()
import weakref
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
	from pygame.locals import K_n
	from pygame.locals import K_b
except ImportError:
	raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
	import numpy as np
except ImportError:
	raise RuntimeError('cannot import numpy, make sure numpy package is installed')

VIEW_WIDTH = 1920
VIEW_HEIGHT = 1080
VIEW_FOV = 90

class Client(object):
	"""
	Basic implementation of a synchronous client.
	"""

	def __init__(self):
		self.client = None
		self.world = None
		self.camera = None
		self.camera2 = None
		self.car = None
		self.display = None
		self.display2 = None	
		self.image = None
		self.capture = True
		self.counter = 0
		self.depth = None
		self.pose = []
		self.spectator = None

	def camera_blueprint(self):
		"""
		Returns camera blueprint.
		"""

		camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
		camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
		camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
		camera_bp.set_attribute('fov', str(VIEW_FOV))
		return camera_bp

	def set_synchronous_mode(self, synchronous_mode):
		"""
		Sets synchronous mode.
		"""

		settings = self.world.get_settings()
		settings.synchronous_mode = synchronous_mode
		self.world.apply_settings(settings)

	def setup_car(self):
		"""
		Spawns actor-vehicle to be controled.
		"""

		car_bp = self.world.get_blueprint_library().filter('model3')[0]
		location = random.choice(self.world.get_map().get_spawn_points())
		self.car = self.world.spawn_actor(car_bp, location)

	def setup_camera(self):
		"""
		Spawns actor-camera to be used to render view.
		"""
		self.spectator = self.world.get_spectator()
		self.spectator.set_transform(carla.Transform(carla.Location(x=0, y=0, z=8500),
		carla.Rotation(pitch=-90, yaw=90)))
		#camera_transform = carla.Transform(carla.Location(x=0, y = 0, z=25000), carla.Rotation(pitch=-90))
		self.camera = self.world.spawn_actor(self.camera_blueprint(), self.spectator.get_transform())
		weak_self = weakref.ref(self)
		self.camera.listen(lambda image: weak_self().set_image(weak_self, image))

		calibration = np.identity(3)
		calibration[0, 2] = VIEW_WIDTH / 2.0
		calibration[1, 2] = VIEW_HEIGHT / 2.0
		calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
		self.camera.calibration = calibration

	def setup_camera2(self):
		"""
		Spawns actor-camera to be used to render view.
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

	def control(self, car):
		"""
		Car controls
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
			spectator = self.world.get_spectator()
			spectator.set_transform(carla.Transform(carla.Location(x=0, y=0, z=800),
			carla.Rotation(pitch=-90, yaw=90)))
			transform = spectator.get_transform()
			self.camera.set_transform(carla.Transform(transform.location + carla.Location(z=50),
			carla.Rotation(pitch=-90, yaw=90)))
		if keys[K_n]:			
			self.camera.set_transform(carla.Transform(carla.Location(x=-5.5, z=2.8),
			carla.Rotation(pitch=-15)))
		car.apply_control(control)
		return False

	@staticmethod
	def set_image(weak_self, img):
		"""
		Sets image coming from camera sensor.
		"""

		self = weak_self()
		if self.capture:
			self.image = img
			self.capture = False

	def render(self, display):
		"""
		Render camera to pygame display.
		"""

		if self.image is not None:
			array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
			array = np.reshape(array, (self.image.height, self.image.width, 4))
			array = array[:, :, :3]
			array = array[:, :, ::-1]
			surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
			display.blit(surface, (0, 0))

	def game_loop(self):
		"""
		Main loop.
		"""

		try:
			pygame.init()
			keys = pygame.key.get_pressed()
			if keys[K_ESCAPE]:
				return True
			self.client = carla.Client('127.0.0.1', 2000)
			self.client.set_timeout(2.0)
			self.world = self.client.get_world()
			self.setup_car()
			self.setup_camera2()
			self.display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
			pygame_clock = pygame.time.Clock()
			self.set_synchronous_mode(True)
			vehicles = self.world.get_actors().filter('vehicle.*')

			while True:
				self.world.tick()
				self.capture = True
				pygame_clock.tick_busy_loop(30)
				self.render(self.display)

				cv2.waitKey(1)
				carImg = pygame.image.load('mainroad.png')
				surf = pygame.Surface((250, 500), pygame.SRCALPHA)
				#surf.set_alpha(190)  # alpha value
				surf.blit(pygame.transform.scale(carImg, (250, 500)), (0, 0))

				self.display.blit(surf, (0, 0))
				pygame.display.flip()
				if self.control(self.car):
					return
				
		#except Exception as e: print(e)
		finally:

			self.set_synchronous_mode(False)
			if self.camera is not None:
				self.camera.destroy()
			if self.car is not None:
				self.car.destroy()
			if self.spectator is not None:
				self.spectator.destroy()
			pygame.quit()
			cv2.destroyAllWindows()

def main():
	try:
		client = Client()
		client.game_loop()
	finally:
		print('EXIT')


if __name__ == '__main__':
	main()