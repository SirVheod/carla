import cv2
import random
import weakref
import carla
import glob
import os
import sys
import time
from SaveImageUtil import SaveImageUtil as save
from wintersim_image_detection import ImageDetection as detectionAPI

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

VIEW_WIDTH = 720 #1920//4
VIEW_HEIGHT = 480 #1080//4
VIEW_FOV = 90

class MultipleWindows():
    """ Wintersim multiplewindows class."""

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

    def setup_front_rgb_camera(self):
        """ Spawns actor-camera to be used to RGB camera view.
        Sets calibration for client-side boxes rendering. """

        camera_transform = carla.Transform(carla.Location(x=2.0, z=1.0), carla.Rotation(pitch=0))
        self.front_rgb_camera = self.world.spawn_actor(self.camera_blueprint(), camera_transform, attach_to=self.car)
        weak_rgb_self = weakref.ref(self)
        self.front_rgb_camera.listen(lambda front_rgb_image: weak_rgb_self(
        ).set_front_rgb_image(weak_rgb_self, front_rgb_image))
        self.front_rgb_camera_display = cv2.namedWindow('front RGB camera')

    def setup_back_rgb_camera(self):
        """ Spawns actor-camera to be used to RGB camera view.
        Sets calibration for client-side boxes rendering. """

        camera_transform = carla.Transform(carla.Location(x=-3.5, z=1.5), carla.Rotation(pitch=-10, yaw=180))
        self.back_rgb_camera = self.world.spawn_actor(self.camera_blueprint(), camera_transform, attach_to=self.car)
        weak_back_rgb_self = weakref.ref(self)
        self.back_rgb_camera.listen(lambda back_rgb_image: weak_back_rgb_self(
        ).set_back_rgb_image(weak_back_rgb_self, back_rgb_image))
        self.back_rgb_camera_display = cv2.namedWindow('back RGB camera')

    def setup_front_depth_camera(self):
        """ Spawns actor-camera to be used to render view.
        Sets calibration for client-side boxes rendering. """

        depth_camera_transform = carla.Transform(carla.Location(x=0, z=2.8), carla.Rotation(pitch=0))
        self.depth_camera = self.world.spawn_actor(
            self.depth_camera_blueprint(), depth_camera_transform, attach_to=self.car)
        weak_depth_self = weakref.ref(self)
        self.depth_camera.listen(lambda front_depth_image: weak_depth_self(
        ).set_front_depth_image(weak_depth_self, front_depth_image))
        self.front_depth_display = cv2.namedWindow('front_depth_image')

    @staticmethod
    def set_front_rgb_image(weak_self, img):
        """ Sets image coming from camera sensor. """

        self = weak_self()
        self.front_rgb_image = img

    @staticmethod
    def set_back_rgb_image(weak_self, img):
        """ Sets image coming from camera sensor. """

        self = weak_self()
        self.back_rgb_image = img

    @staticmethod
    def set_front_depth_image(weak_depth_self, depth_img):
        """ Sets image coming from camera sensor. """
        
        self = weak_depth_self()
        self.front_depth_image = depth_img

    def render_front_depth(self, front_depth_display):
        if self.front_depth_image is not None:
            i = np.array(self.front_depth_image.raw_data)
            i2 = i.reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))
            i3 = i2[:, :, :3]
            #i4 = detectionAPI.DetectObjects(i3)
            cv2.imshow("front_depth_image", i3)

    def render_front_rgb_camera(self, rgb_display):
        if self.front_rgb_image is not None:
            i = np.array(self.front_rgb_image.raw_data)
            i2 = i.reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))
            #i4 = i2[:, :, :3]
            i4 = detectionAPI.DetectObjects(i2)
            cv2.imshow("front RGB camera", i4)

            if self.recordImages:
                self.counterimages += 1
                file_name = "img" + str(self.counterimages)
                save.save_image(file_name, i3)

    def render_back_rgb_camera(self, rgb_display):
        if self.back_rgb_image is not None:
            i = np.array(self.back_rgb_image.raw_data)
            i2 = i.reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))
            #i3 = i2[:, :, :3]
            i4 = detectionAPI.DetectObjects(i2)
            cv2.imshow("back RGB camera", i4)

    def render_all_windows(self):
        """ Render all separate sensors (cv2 windows)"""
        self.render_front_rgb_camera(self.front_rgb_camera_display)
        self.render_front_depth(self.front_depth_display)
        self.render_back_rgb_camera(self.back_rgb_camera_display)

    def render_views(self):

        imgs = []

        if self.front_rgb_image is not None:
            i = np.array(self.front_rgb_image.raw_data)
            i2 = i.reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))
            #i4 = i2[:, :, :3]
            i4 = detectionAPI.DetectObjects(i2)
            imgs.append(i4)

        if self.front_depth_image is not None:
            i = np.array(self.front_depth_image.raw_data)
            i2 = i.reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))
            i3 = i2[:, :, :3]
            imgs.append(i3)

        if self.back_rgb_image is not None:
            i = np.array(self.back_rgb_image.raw_data)
            i2 = i.reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))
            #i3 = i2[:, :, :3]
            i4 = detectionAPI.DetectObjects(i2)
            imgs.append(i4)

        return imgs

    def destroy(self):
        """ Destroy spawned sensors and close all cv2 windows"""
        self.front_rgb_camera.destroy()
        self.depth_camera.destroy()
        self.back_rgb_camera.destroy()
        cv2.destroyAllWindows()

    def __init__(self, car, camera, world):
        self.camera = camera
        self.world = world
        self.car = car
      
        self.counter = 0
        self.counterimages = 0
        self.recordImages = False
        self.pose = []
        self.log = False

        # Front RGB camera
        self.front_rgb_camera_display = None
        self.front_rgb_camera = None
        self.front_rgb_image = None

        # Back RGB camera
        self.back_rgb_camera_display = None
        self.back_rgb_camera = None
        self.back_rgb_image = None

        # Front Depth camera
        self.front_depth_camera = None
        self.front_depth_display = None
        self.front_depth_image = None

        detectionAPI.Initialize()

        self.setup_front_rgb_camera()
        self.setup_back_rgb_camera()
        self.setup_front_depth_camera()