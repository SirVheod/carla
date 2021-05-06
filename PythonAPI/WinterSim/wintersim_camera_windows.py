from SaveImageUtil import SaveImageUtil as save
import threading
import weakref
import random
import carla
import time
import glob
import sys
import cv2
import os
import collections
import datetime
import logging
import math
import re

try:
    from wintersim_yolo_gpu_detection import ImageDetection as detectionAPI
except ImportError:
    print("couldn't load wintersim_yolo_gpu_detection")
    pass

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

# cv2 window width, height and camera fov
VIEW_WIDTH = 608
VIEW_HEIGHT = 384
VIEW_FOV = 70

def sumMatrix(A, B):
    A = np.array(A)
    B = np.array(B)
    answer = A + B
    return answer.tolist()

class CameraWindows(threading.Thread):
    """ Wintersim Camerawindows class. Rendering CV2 windows happens in different thread than main pygame loop."""

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
        camera_transform = carla.Transform(carla.Location(x=2.0, z=2.0), carla.Rotation(pitch=0))
        self.front_rgb_camera = self.world.spawn_actor(self.camera_blueprint(), camera_transform, attach_to=self.car)
        weak_rgb_self = weakref.ref(self)
        self.front_rgb_camera.listen(lambda front_rgb_image: weak_rgb_self().set_front_rgb_image(weak_rgb_self, front_rgb_image))
        self.front_rgb_camera_display = cv2.namedWindow('front RGB camera')

    def setup_back_rgb_camera(self):
        """ Spawns actor-camera to be used to vehicle back RGB camera view.
        set_front_depth_image method gets callback from camera"""
        camera_transform = carla.Transform(carla.Location(x=-3.5, z=2.0), carla.Rotation(pitch=-10, yaw=180))
        self.back_rgb_camera = self.world.spawn_actor(self.camera_blueprint(), camera_transform, attach_to=self.car)
        weak_back_rgb_self = weakref.ref(self)
        self.back_rgb_camera.listen(lambda back_rgb_image: weak_back_rgb_self(
        ).set_back_rgb_image(weak_back_rgb_self, back_rgb_image))
        self.back_rgb_camera_display = cv2.namedWindow('back RGB camera')

    def setup_front_depth_camera(self):
        """ Spawns actor-camera to be used front dept-camera.
       front_depth_image method gets callback from camera """
        depth_camera_transform = carla.Transform(carla.Location(x=0, z=2.8), carla.Rotation(pitch=0))
        self.depth_camera = self.world.spawn_actor(
            self.depth_camera_blueprint(), depth_camera_transform, attach_to=self.car)
        weak_depth_self = weakref.ref(self)
        self.depth_camera.listen(lambda front_depth_image: weak_depth_self(
        ).set_front_depth_image(weak_depth_self, front_depth_image))
        self.front_depth_display = cv2.namedWindow('front_depth_image')

    @staticmethod
    def set_front_rgb_image(weak_self, img):
        """ Sets image coming from front RGB camera sensor. """
        self = weak_self()
        self.front_rgb_image = img

    @staticmethod
    def set_back_rgb_image(weak_self, img):
        """ Sets image coming from back RGB camera sensor. """
        self = weak_self()
        self.back_rgb_image = img

    @staticmethod
    def set_front_depth_image(weak_depth_self, depth_img):
        """ Sets image coming from depth camera sensor. """
        self = weak_depth_self()
        self.front_depth_image = depth_img

    def render_front_depth(self, front_depth_display):
        """ Render front depth camera"""
        if self.front_depth_image is not None:
            i = np.asarray(self.front_depth_image.raw_data)
            i2 = i.reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))
            i3 = i2[:, :, :3]
            cv2.imshow("front_depth_image", i3)

    def get_latests_results(self):
        return self.results
   
    def render_front_rgb_camera(self, rgb_display):
        """ Render front RGB camera"""
        if self.front_rgb_image is not None:
            #self.render_lane_detection(self.front_rgb_image)
            i = np.asarray(self.front_rgb_image.raw_data)
            i2 = i.reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))
            i3 = i2[:, :, :3]
            if self.detection:
                i4, self.results = detectionAPI.detect_objects(i2, i3, None)
                cv2.imshow("front RGB camera", i4)
            else:
                cv2.imshow("front RGB camera", i3)

            if self.record_images:
                self.imagecounter += 1
                file_name = "img" + str(self.imagecounter)
                save.save_single_image(file_name, i3)
            
            self.front_rgb_image = None

    def render_back_rgb_camera(self, rgb_display):
        """ Render back RGB camera"""
        if self.back_rgb_image is not None:
            i = np.asarray(self.back_rgb_image.raw_data)
            i2 = i.reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))
            i3 = i2[:, :, :3]
            if self.detection:
                i4 = detectionAPI.detect_objects(i2, i3, None)
                cv2.imshow("back RGB camera", i4)
            else:
                cv2.imshow("back RGB camera", i3)

    def render_all_windows(self):
        """ Render all separate sensors (cv2 windows)"""
        self.render_front_rgb_camera(self.front_rgb_camera_display)
        #self.render_back_rgb_camera(self.back_rgb_camera_display)
        #self.render_front_depth(self.front_depth_display)
       
    def destroy(self):
        """Destroy spawned sensors and close all cv2 windows"""
        if self.record_images:
            save.save_images_to_video()
            self.record_images = False
        self.stop()

        if self.front_rgb_camera is not None:
            self.front_rgb_camera.destroy()

        if self.back_rgb_camera is not None:
            self.back_rgb_camera.destroy()

        if self.front_depth_camera is not None:
            self.front_depth_camera.destroy()

        cv2.destroyAllWindows()

    def __init__(self, car, camera, world, record, detection):
        super(CameraWindows, self).__init__()
        self.__flag = threading.Event()             # The flag used to pause the thread
        self.__flag.set()                           # Set to True
        self.__running = threading.Event()          # Used to stop the thread identification
        self.__running.set()                        # Set running to True
        
        self.camera = camera
        self.world = world
        self.car = car
        self.record_images = record
        self.detection = detection
        self.imagecounter = 0
        
        self.front_rgb_camera_display = None
        self.front_rgb_camera = None
        self.front_rgb_image = None

        self.back_rgb_camera_display = None
        self.back_rgb_camera = None
        self.back_rgb_image = None

        self.front_depth_camera = None
        self.front_depth_display = None
        self.front_depth_image = None
        self.results = None
        self.lane_detection_results = None

        # init save image utility
        save.initialize()

        if self.detection:
            detectionAPI.Initialize()

        #self.setup_back_rgb_camera()
        self.setup_front_rgb_camera()
        #self.setup_front_depth_camera()

    def run(self):
        while self.__running.isSet():
            self.__flag.wait()                      # return immediately when it is True, block until the internal flag is True when it is False
            self.render_all_windows()               # render all cv2 windows when flag is True

    def pause(self):
        self.__flag.clear()                         # Set to False to block the thread

    def resume(self):
        self.__flag.set()                           # Set to True, let the thread stop blocking

    def stop(self):
        self.__flag.set()                           # Resume the thread from the suspended state, if it is already suspended
        self.__running.clear()                      # Set to False

    def render_lane_detection(self, image):
        # code adopted from: https://github.com/angelkim88/CARLA-Lane_Detection

        pt1_sum_ri = (0, 0)
        pt2_sum_ri = (0, 0)
        pt1_avg_ri = (0, 0)
        count_posi_num_ri = 0
        pt1_sum_le = (0, 0)
        pt2_sum_le = (0, 0)
        pt1_avg_le = (0, 0)
        count_posi_num_le = 0

        test_im = np.array(image.raw_data)
        test_im = test_im.copy()
        test_im = test_im.reshape((image.height, image.width, 4))
        test_im = test_im[:, :, :3]
        size_im = cv2.resize(test_im, dsize=(640, 480))
        roi = size_im[240:480, 108:532]
        roi_im = cv2.resize(roi, (424, 240))
        Blur_im = cv2.bilateralFilter(roi_im, d=-1, sigmaColor=3, sigmaSpace=3)
        edges = cv2.Canny(Blur_im, 50, 100)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180.0, threshold=25, minLineLength=10, maxLineGap=20)

        N = lines.shape[0]

        for line in range(N):
            x1 = lines[line][0][0]
            y1 = lines[line][0][1]
            x2 = lines[line][0][2]
            y2 = lines[line][0][3]

            if x2 == x1:
                a = 1
            else:
                a = x2 - x1

            b = y2 - y1
            radi = b / a
            theta_atan = math.atan(radi) * 180.0 / math.pi

            _pt1 = (x1 + 108, y1 + 240)
            pt1_ri = _pt1
            pt1_le = _pt1

            _pt2 = (x2 + 108, y2 + 240)
            pt2_le = _pt2
            pt2_ri = _pt2

            if theta_atan > 30.0 and theta_atan < 80.0:
                count_posi_num_ri += 1
                pt1_sum_ri = sumMatrix(pt1_ri, pt1_sum_ri)
                pt2_sum_ri = sumMatrix(pt2_ri, pt2_sum_ri)

            if theta_atan < -30.0 and theta_atan > -80.0:
                count_posi_num_le += 1
                pt1_sum_le = sumMatrix(pt1_le, pt1_sum_le)
                pt2_sum_le = sumMatrix(pt2_le, pt2_sum_le)

        posi_ri_array = np.array(count_posi_num_ri)
        posi_le_array = np.array(count_posi_num_le)
        pt1_avg_ri = pt1_sum_ri // posi_ri_array
        pt2_avg_ri = pt2_sum_ri // posi_ri_array
        pt1_avg_le = pt1_sum_le // posi_le_array
        pt2_avg_le = pt2_sum_le // posi_le_array

        x1_avg_ri, y1_avg_ri = pt1_avg_ri
        x2_avg_ri, y2_avg_ri = pt2_avg_ri
        a_avg_ri = ((y2_avg_ri - y1_avg_ri) / (x2_avg_ri - x1_avg_ri))
        b_avg_ri = (y2_avg_ri - (a_avg_ri * x2_avg_ri))
        pt2_y2_fi_ri = 480

        if a_avg_ri > 0:
            pt2_x2_fi_ri = int((pt2_y2_fi_ri - b_avg_ri) // a_avg_ri)
        else:
            pt2_x2_fi_ri = 0

        pt2_fi_ri = (pt2_x2_fi_ri, pt2_y2_fi_ri)
        x1_avg_le, y1_avg_le = pt1_avg_le
        x2_avg_le, y2_avg_le = pt2_avg_le

        a_avg_le = ((y2_avg_le - y1_avg_le) / (x2_avg_le - x1_avg_le))
        b_avg_le = (y2_avg_le - (a_avg_le * x2_avg_le))

        pt1_y1_fi_le = 480
        if a_avg_le < 0:
            pt1_x1_fi_le = int((pt1_y1_fi_le - b_avg_le) // a_avg_le)
        else:
            pt1_x1_fi_le = 0

        pt1_fi_le = (pt1_x1_fi_le, pt1_y1_fi_le)

        cv2.line(size_im, tuple(pt1_avg_ri), tuple(pt2_fi_ri), (0, 255, 0), 2)  # right lane
        cv2.line(size_im, tuple(pt1_fi_le), tuple(pt2_avg_le), (0, 255, 0), 2)  # left lane
        cv2.line(size_im, (320, 480), (320, 360), (0, 228, 255), 1)             # middle lane
        FCP_img = np.zeros(shape=(480, 640, 3), dtype=np.uint8) + 0
        FCP = np.array([pt2_avg_le, pt1_fi_le, pt2_fi_ri, pt1_avg_ri])
        cv2.fillConvexPoly(FCP_img, FCP, color=(255, 242, 213))
        alpha = 0.9
        size_im = cv2.addWeighted(size_im, alpha, FCP_img, 1 - alpha, 0)

        lane_center_y_ri = 360
        if a_avg_ri > 0:
            lane_center_x_ri = int((lane_center_y_ri - b_avg_ri) // a_avg_ri)
        else:
            lane_center_x_ri = 0

        lane_center_y_le = 360
        if a_avg_le < 0:
            lane_center_x_le = int((lane_center_y_le - b_avg_le) // a_avg_le)
        else:
            lane_center_x_le = 0

        cv2.line(size_im, (lane_center_x_le, lane_center_y_le - 10), (lane_center_x_le, lane_center_y_le + 10), (0, 228, 255), 1)
        cv2.line(size_im, (lane_center_x_ri, lane_center_y_ri - 10), (lane_center_x_ri, lane_center_y_ri + 10), (0, 228, 255), 1)
        lane_center_x = ((lane_center_x_ri - lane_center_x_le) // 2) + lane_center_x_le
        cv2.line(size_im, (lane_center_x, lane_center_y_ri - 10), (lane_center_x, lane_center_y_le + 10), (0, 228, 255), 1)

        text_left = 'Turn Left'
        text_right = 'Turn Right'
        text_center = 'Center'
        text_non = ''
        org = (320, 440)
        font = cv2.FONT_HERSHEY_SIMPLEX
        #print('lane_center_x=', lane_center_x)

        if 0 < lane_center_x <= 315:
            cv2.putText(size_im, text_left, org, font, 0.7, (0, 0, 255), 2)

        elif 315 < lane_center_x < 325:
            cv2.putText(size_im, text_center, org, font, 0.7, (0, 0, 255), 2)

        elif lane_center_x >= 325:
            cv2.putText(size_im, text_right, org, font, 0.7, (0, 0, 255), 2)

        elif lane_center_x == 0:
            cv2.putText(size_im, text_non, org, font, 0.7, (0, 0, 255), 2)

        cv2.imshow('Lane detection', size_im)
        cv2.waitKey(1)