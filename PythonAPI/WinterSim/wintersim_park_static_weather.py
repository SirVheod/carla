#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Open3D Lidar visuialization example for CARLA"""

import glob
import os
import sys
import argparse
import time
from datetime import datetime
import random
import numpy as np
from matplotlib import cm
import open3d as o3d
import math
import weakref
import threading
import time
from queue import Queue

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
LABEL_COLORS = np.array([
    (255, 255, 255), # None
    (70, 70, 70),    # Building
    (100, 40, 40),   # Fences
    (55, 90, 80),    # Other
    (220, 20, 60),   # Pedestrian
    (153, 153, 153), # Pole
    (157, 234, 50),  # RoadLines
    (128, 64, 128),  # Road
    (244, 35, 232),  # Sidewalk
    (107, 142, 35),  # Vegetation
    (0, 0, 142),     # Vehicle
    (102, 102, 156), # Wall
    (220, 220, 0),   # TrafficSign
    (70, 130, 180),  # Sky
    (81, 0, 81),     # Ground
    (150, 100, 100), # Bridge
    (230, 150, 140), # RailTrack
    (180, 165, 180), # GuardRail
    (250, 170, 30),  # TrafficLight
    (110, 190, 160), # Static
    (170, 120, 50),  # Dynamic
    (45, 60, 150),   # Water
    (145, 170, 100), # Terrain
    (255, 0, 174), #DOTS
]) / 255.0 # normalize each channel [0-1] since is what Open3D uses


def semantic_lidar_callback(point_cloud, point_list):
    """Prepares a point cloud with semantic segmentation
    colors ready to be consumed by Open3D"""
    data = np.frombuffer(point_cloud.raw_data, dtype=np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]))
    # We're negating the y to correclty visualize a world that matches
    # what we see in Unreal since Open3D uses a right-handed coordinate system
    points = np.array([data['x'], -data['y'], data['z']]).T

    # # An example of adding some noise to our data if needed:
    # points += np.random.uniform(-0.05, 0.05, size=points.shape)

    # Colorize the pointcloud based on the CityScapes color palette
    labels = np.array(data['ObjTag'])
    int_color = LABEL_COLORS[labels]

    # # In case you want to make the color intensity depending
    # # of the incident ray angle, you can use:
    # int_color *= np.array(data['CosAngle'])[:, None]

    point_list.points = o3d.utility.Vector3dVector(points)
    point_list.colors = o3d.utility.Vector3dVector(int_color)


def lidar_callback(point_cloud, point_list, main):
    """Prepares a point cloud with intensity
    colors ready to be consumed by Open3D"""
    data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))
    # Isolate the intensity and compute a color for it
    intensity = data[:, -1]
    intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
    int_color = np.c_[
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]

    # Isolate the 3D data
    points = data[:, :-1]

    # We're negating the y to correclty visualize a world that matches
    # what we see in Unreal since Open3D uses a right-handed coordinate system
    points[:, :1] = -points[:, :1]

    # # An example of converting points from sensor to vehicle space if we had
    # # a carla.Transform variable named "tran":
    # points = np.append(points, np.ones((points.shape[0], 1)), axis=1)
    # points = np.dot(tran.get_matrix(), points.T).T
    # points = points[:, :-1]

    point_list.points = o3d.utility.Vector3dVector(points)
    point_list.colors = o3d.utility.Vector3dVector(int_color)
    ##record data
    if main.record is True:
        point_cloud.save_to_disk('pointclouds/robosense_pointclouds/%.6d.ply' % point_cloud.frame)


def generate_lidar_bp(arg, blueprint_library, delta, is_ouster):
    if arg.semantic:
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast_semantic')
    else: 
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    if is_ouster:
        lidar_bp.set_attribute('upper_fov', str(16.6))
        lidar_bp.set_attribute('lower_fov', str(-16.6))
        lidar_bp.set_attribute('channels', str(32.0))
        lidar_bp.set_attribute('range', str(150.0))
        lidar_bp.set_attribute('rotation_frequency', str(1.0 / delta))
        lidar_bp.set_attribute('points_per_second', str(655360.0))
        if arg.semantic is False:
            lidar_bp.set_attribute('atmosphere_attenuation_rate', str(arg.atmosphere_attenuation_rate))
            lidar_bp.set_attribute('noise_stddev', str(arg.noise_stddev))
    else:
        lidar_bp.set_attribute('upper_fov', str(arg.upper_fov))
        lidar_bp.set_attribute('lower_fov', str(arg.lower_fov))
        lidar_bp.set_attribute('horizontal_fov', str(158))
        lidar_bp.set_attribute('channels', str(arg.channels))
        lidar_bp.set_attribute('range', str(arg.range))
        lidar_bp.set_attribute('rotation_frequency', str(1.0 / delta))
        lidar_bp.set_attribute('points_per_second', str(arg.points_per_second))
        if arg.semantic is False:
            lidar_bp.set_attribute('atmosphere_attenuation_rate', str(arg.atmosphere_attenuation_rate))
            lidar_bp.set_attribute('noise_stddev', str(arg.noise_stddev))
    return lidar_bp


def add_open3d_axis(vis):
    """Add a small 3D axis on Open3D Visualizer"""
    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    axis.lines = o3d.utility.Vector2iVector(np.array([
        [0, 1],
        [0, 2],
        [0, 3]]))
    axis.colors = o3d.utility.Vector3dVector(np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    vis.add_geometry(axis)


class RadarSensor(object):
    def __init__(self, blueprint_library, spawn_point, world):
        self.sensor = None
        self.debug = world.debug
        self.velocity_range = 7.5 # m/s
        radar_bp = blueprint_library.find('sensor.other.radar')
        radar_bp.set_attribute('horizontal_fov', str(35))
        radar_bp.set_attribute('vertical_fov', str(20))
        self.sensor = world.spawn_actor(radar_bp, spawn_point)
        # We need a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        # To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
        # points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        # points = np.reshape(points, (len(radar_data), 4))

        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)

            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            norm_velocity = detect.velocity / self.velocity_range # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            self.debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(255, 0, 0))

class main(object):
    def __init__(self, args):
        self.record = False
        self.timestart = 0
        self.f(args)

    def change(self):
        self.record = True

    """Main function of the script"""
    def f(self, arg):
        client = carla.Client(arg.host, arg.port)
        client.set_timeout(5.0)
        world = client.get_world()

        try:
            original_settings = world.get_settings()
            settings = world.get_settings()

            delta = 0.05

            settings.fixed_delta_seconds = delta
            settings.synchronous_mode = True
            world.apply_settings(settings)

            blueprint_library = world.get_blueprint_library()
            spawn_points = world.get_map().get_spawn_points()
            robosense_bp = generate_lidar_bp(arg, blueprint_library, delta, False)
            robosense = world.spawn_actor(robosense_bp, spawn_points[0]) #robo = 0 ouster = 2

            radar = RadarSensor(blueprint_library, spawn_points[1], world)

            point_list = o3d.geometry.PointCloud()
            if arg.semantic:
                robosense.listen(lambda data: semantic_lidar_callback(data, point_list))
            else:
                robosense.listen(lambda data: lidar_callback(data, point_list, self)) 

            ##open3d window for robosense
            vis = o3d.visualization.Visualizer()
            vis.create_window(
                window_name='Robosense Lidar',
                width=960,
                height=540,
                left=480,
                top=270)
            vis.get_render_option().background_color = [0.05, 0.05, 0.05]
            vis.get_render_option().point_size = 2
            vis.get_render_option().show_coordinate_frame = True


            if arg.show_axis:
                add_open3d_axis(vis)

            frame = 0
            dt0 = datetime.now()

            ##Thread for input data
            input_thread = Input(self)
            input_thread.start()
            while True:
                if self.record:
                    ##start 10 second timer
                    if self.timestart is 0:
                        self.timestart = time.time() + 10
                    if time.time() >= self.timestart and self.timestart is not 0:
                        print('PointCloud data saved to pointclouds/robosense_pointclouds/')
                        self.record = False
                        self.timestart = 0
                    
                if frame == 2:
                    vis.add_geometry(point_list)
                vis.update_geometry(point_list)

                vis.poll_events()
                vis.update_renderer()
                # # This can fix Open3D jittering issues:
                time.sleep(0.005)
                world.tick()

                process_time = datetime.now() - dt0
                #sys.stdout.write('\r' + 'FPS: ' + str(1.0 / process_time.total_seconds()))
                #sys.stdout.flush()
                dt0 = datetime.now()
                frame += 1      

        finally:
            world.apply_settings(original_settings)
            robosense.destroy()
            radar.sensor.destroy()
            vis.destroy_window()

class Input(threading.Thread):
    def __init__(self, main):
        super(Input, self).__init__()
        self.daemon = True
        self.input = None
        self.main = main

    def run(self):
        while True:
            ##Get input for recordin
            if self.main.record is not True:
                print('')
                self.input = input('Press enter to save 10 seconds of a PointCloud data')
                print('Started recording...')
                self.main.record = True
            time.sleep(3)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host CARLA Simulator (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port of CARLA Simulator (default: 2000)')
    argparser.add_argument(
        '--show-axis',
        action='store_true',
        default=True,
        help='show the cartesian coordinates axis')
    argparser.add_argument(
        '--semantic',
        action='store_true',
        default=False,
        help='use the semantic lidar instead, which provides ground truth'
        ' information')
    argparser.add_argument(
        '--upper-fov',
        default=15.0,
        type=float,
        help='lidar\'s upper field of view in degrees (default: 15.0)')
    argparser.add_argument(
        '--lower-fov',
        default=-15.0,
        type=float,
        help='lidar\'s lower field of view in degrees (default: -25.0)')
    argparser.add_argument(
        '--channels',
        default=16.0,
        type=float,
        help='lidar\'s channel count (default: 64)')
    argparser.add_argument(
        '--range',
        default=150.0,
        type=float,
        help='lidar\'s maximum range in meters (default: 100.0)')
    argparser.add_argument(
        '--points-per-second',
        default=600000,
        type=int,
        help='lidar\'s points per second (default: 500000)')
    argparser.add_argument(
        '--atmosphere-attenuation-rate',
        default=0.000,
        type=float,
        help='lidar\'s testi1')
    argparser.add_argument(
        '--noise-stddev',
        default=0.02,
        type=float,
        help='lidar\'s testi2')
    args = argparser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        print(' - Exited by user.')
