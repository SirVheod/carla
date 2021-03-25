#!/usr/bin/env python

import glob
import os
import sys
import re
import threading
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import pygame
import os
import math
import datetime
import carla
import cv2
import random
import weakref

VIEW_WIDTH = 1920//4
VIEW_HEIGHT = 1080//4
VIEW_FOV = 90

class Wintersim_Windows(object):

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

        self.imagecounter = 0
        self.recordImages = False

        # setup
        #self.setup_front_rgb_camera()
        #self.setup_back_rgb_camera()
        #self.setup_front_depth_camera()

    def on_world_tick(self, timestamp):
        print("todo")

    def Render_all_windows():
        print("todo")

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
        self.front_rgb_camera.listen(lambda front_rgb_image: weak_rgb_self(
        ).set_front_rgb_image(weak_rgb_self, front_rgb_image))

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
        self.back_rgb_camera.listen(lambda back_rgb_image: weak_back_rgb_self(
        ).set_back_rgb_image(weak_back_rgb_self, back_rgb_image))

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
        self.depth_camera = self.world.spawn_actor(
            self.depth_camera_blueprint(), depth_camera_transform, attach_to=self.car)
        weak_depth_self = weakref.ref(self)
        self.depth_camera.listen(lambda front_depth_image: weak_depth_self(
        ).set_front_depth_image(weak_depth_self, front_depth_image))

        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        self.depth_camera.calibration = calibration

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

            if self.recordImages:
                self.imagecounter +=1
                file_name = "img" + str(self.imagecounter)
                saveUtil.save_image(file_name, i3)

    def back_rgb_camera_render(self, rgb_display):
        if self.back_rgb_image is not None:
            i = np.array(self.back_rgb_image.raw_data)
            i2 = i.reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))
            i3 = i2[:, :, :3]
            cv2.imshow("back RGB camera", i3)

    def log_data(self):
        global start
        freq = 1/(time.time() - start)

        # sys.stdout.write("\rFrequency:{}Hz		Logging:{}".format(int(freq),self.log))
        # sys.stdout.write("\r{}".format(self.car.get_transform().rotation))

        sys.stdout.flush()
        if self.log:
            name = 'log/' + str(self.counter) + '.png'
            self.front_depth_image.save_to_disk(name)
            position = self.car.get_transform()
            pos = None
            pos = (int(self.counter), position.location.x, position.location.y, position.location.z,
                   position.rotation.roll, position.rotation.pitch, position.rotation.yaw)
            self.pose.append(pos)
            self.counter += 1
        start = time.time()