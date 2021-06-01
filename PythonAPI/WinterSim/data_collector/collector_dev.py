import glob
import os
import sys

try:
    sys.path.append(glob.glob('C:/carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import logging
import math
import pygame
import random
import queue
import numpy as np
from data_collector.bounding_box import create_kitti_datapoint
from data_collector.constants import *
import data_collector.image_converter
from data_collector.dataexport import *
from scipy.spatial import distance

""" OUTPUT FOLDER GENERATION """
PHASE = "training"
OUTPUT_FOLDER = os.path.join("_out", PHASE)
folders = ['calib', 'image_2', 'label_2', 'velodyne', 'planes']


def maybe_create_dir(path):
    if not os.path.exists(directory):
        os.makedirs(directory)


for folder in folders:
    directory = os.path.join(OUTPUT_FOLDER, folder)
    maybe_create_dir(directory)

""" DATA SAVE PATHS """
GROUNDPLANE_PATH = os.path.join(OUTPUT_FOLDER, 'planes/{0:06}.txt')
LIDAR_PATH = os.path.join(OUTPUT_FOLDER, 'velodyne/{0:06}.bin')
#C:/Users/rekla/Documents/Complex-YOLOv3/data/KITTI/object/training
#C:/carla/PythonAPI/WinterSim/object_detection/data/KITTI/object/training
x_path = os.path.join('C:/carla/PythonAPI/WinterSim/object_detection/data/KITTI/object/training', 'velodyne/{0:06}.bin')
LABEL_PATH = os.path.join(OUTPUT_FOLDER, 'label_2/{0:06}.txt')
IMAGE_PATH = os.path.join(OUTPUT_FOLDER, 'image_2/{0:06}.png')
CALIBRATION_PATH = os.path.join(OUTPUT_FOLDER, 'calib/{0:06}.txt')


class SynchronyModel(object):
    def __init__(self, world, client):
        self.x = world
        self.world = world.world
        self.client = client
        self.init_setting, self.traffic_manager = self._make_setting()
        self.blueprint_library = self.world.get_blueprint_library()
        self.actor_list = []
        self.frame = None
        self.player = None
        self.captured_frame_no = 0
        self.sensors = []
        self._queues = []
        self.point_cloud = None
        self._span_player()

    def __enter__(self):
        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data

    def __exit__(self, *args, **kwargs):
        # cover the world settings
        self.world.apply_settings(self.init_setting)

    def _make_setting(self):
        traffic_manager = self.client.get_trafficmanager(8000)
        traffic_manager.set_global_distance_to_leading_vehicle(1.0)
        traffic_manager.set_synchronous_mode(True)
        init_setting = self.world.get_settings()
        return init_setting, traffic_manager

    def _span_player(self):
        my_vehicle = self.x.player
        self._span_sensor(my_vehicle)
        self.actor_list.append(my_vehicle)
        self.player = my_vehicle

    def _span_sensor(self, player):
        lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '50')
        lidar_bp.set_attribute('rotation_frequency', '20')
        lidar_bp.set_attribute('upper_fov', '2')
        lidar_bp.set_attribute('lower_fov', '-26.8')
        lidar_bp.set_attribute('points_per_second', '320000')
        lidar_bp.set_attribute('channels', '32')

        transform_sensor = carla.Transform(carla.Location(x=0, y=0, z=CAMERA_HEIGHT_POS))

        my_lidar = self.world.spawn_actor(lidar_bp, transform_sensor, attach_to=player)

        self.actor_list.append(my_lidar)
        self.sensors.append(my_lidar)

        k = np.identity(3)
        k[0, 2] = WINDOW_WIDTH_HALF
        k[1, 2] = WINDOW_HEIGHT_HALF
        f = WINDOW_WIDTH / (2.0 * math.tan(90.0 * math.pi / 360.0))
        k[0, 0] = k[1, 1] = f
        return k

    def _save_training_files(self, point_cloud):
        lidar_fname = x_path.format(self.captured_frame_no)
        save_lidar_data(lidar_fname, point_cloud)

