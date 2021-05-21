#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA WinterSim Autopilot.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down
    CTRL + W     : toggle constant velocity mode at 60 km/h

    L            : toggle next light type
    SHIFT + L    : toggle high beam
    Z/X          : toggle right/left blinker
    I            : toggle interior light

    TAB          : change sensor position
    ` or N       : next sensor
    [1-9]        : change to sensor [1-9]
    G            : toggle radar visualization
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    V            : Select next map layer (Shift+V reverse)
    B            : Load current selected map layer (Shift+B to unload)

    R            : toggle recording images to disk

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    F8           : toggle camera sensors with object detection
    F9           : toggle camera sensors without object detection
    F10          : toggle all sensors with object detection
    H/?          : toggle help
    ESC          : quit;
"""

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

from __future__ import print_function

import glob
import os
import sys
import re
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
from carla import ColorConverter as cc
import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
import wintersim_hud
import wintersim_sensors

from data_collector import collector_dev as collector
from data_collector.bounding_box import create_kitti_datapoint
from data_collector.constants import *
from data_collector import image_converter
from data_collector.dataexport import *

from matplotlib import cm
import open3d as o3d

from wintersim_lidar_object_detection import LidarObjectDetection as LidarObjectDetection
from object_detection import test_both_side_detection_dev as object_detection
from wintersim_camera_windows import CameraWindows
from wintersim_camera_manager import CameraManager
from autopilot import Autopilot

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_F2
    from pygame.locals import K_F4
    from pygame.locals import K_F8
    from pygame.locals import K_F9
    from pygame.locals import K_F10
    from pygame.locals import K_F12
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_b
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_g
    from pygame.locals import K_h
    from pygame.locals import K_i
    from pygame.locals import K_l
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_v
    from pygame.locals import K_w
    from pygame.locals import K_x
    from pygame.locals import K_z
    from pygame.locals import K_o
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name
    

# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================

class World(object):
    def __init__(self, carla_world, hud_wintersim, args):
        self.world = carla_world
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.autopilot = None
        self.wintersim_autopilot = False
        self.original_settings = None
        self.settings = None
        self.data_thread = None
        self.render_lidar_detection = False
        self.dataLidar = None
        self.args = args
        self.multiple_windows_enabled = args.detection
        self.cv2_windows = None
        self.hud_wintersim = hud_wintersim
        self.ud_friction = True
        self.preset = None
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self._weather_presets = []
        self._weather_presets_all = find_weather_presets()
        for preset in self._weather_presets_all:
            if preset[0].temperature <= 0: #get only presets what are for wintersim
                self._weather_presets.append(preset)
        self._weather_index = 0
        self._actor_filter = args.filter
        self._gamma = args.gamma
        self.restart()
        preset = self._weather_presets[0]
        self.world.set_weather(preset[0])
        self.player.gud_frictiong_enabled = False
        self.recording_start = 0
        self.record_data = False
        self.constant_velocity_enabled = False
        self.current_map_layer = 0
        self.world.on_tick(self.hud_wintersim.on_world_tick)
        self.map_layer_names = [
            carla.MapLayer.NONE,
            carla.MapLayer.Buildings,
            carla.MapLayer.Decals,
            carla.MapLayer.Foliage,
            carla.MapLayer.Ground,
            carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles,
            carla.MapLayer.Props,
            carla.MapLayer.StreetLights,
            carla.MapLayer.Walls,
            carla.MapLayer.All
        ]

    def restart(self):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a vehicle according to arg parameter.
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))

        # Get the ego vehicle
        while self.player is None:
            print("Waiting for the ego vehicle...")
            time.sleep(1)
            possible_vehicles = self.world.get_actors().filter('vehicle.*')
            for vehicle in possible_vehicles:
                if vehicle.attributes['role_name'] == "hero":
                    print("Ego vehicle found")
                    self.player = vehicle
                    break
                
        # Set up the sensors.
        self.collision_sensor = wintersim_sensors.CollisionSensor(self.player, self.hud_wintersim)
        self.lane_invasion_sensor = wintersim_sensors.LaneInvasionSensor(self.player, self.hud_wintersim)
        self.gnss_sensor = wintersim_sensors.GnssSensor(self.player)
        self.imu_sensor = wintersim_sensors.IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud_wintersim, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud_wintersim.notification(actor_type)
        self.multiple_window_setup = False
        self.detection = True

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        self.preset = self._weather_presets[self._weather_index]
        self.hud_wintersim.notification('Weather: %s' % self.preset[1])
        self.hud_wintersim.update_sliders(self.preset[0])
        self.player.get_world().set_weather(self.preset[0])

    def next_map_layer(self, reverse=False):
        self.current_map_layer += -1 if reverse else 1
        self.current_map_layer %= len(self.map_layer_names)
        selected = self.map_layer_names[self.current_map_layer]
        self.hud_wintersim.notification('LayerMap selected: %s' % selected)

    def load_map_layer(self, unload=False):
        selected = self.map_layer_names[self.current_map_layer]
        if unload:
            self.hud_wintersim.notification('Unloading map layer: %s' % selected)
            self.world.unload_map_layer(selected)
        else:
            self.hud_wintersim.notification('Loading map layer: %s' % selected)
            self.world.load_map_layer(selected)

    def toggle_radar(self):
        if self.radar_sensor is None:
            self.radar_sensor = wintersim_sensors.RadarSensor(self.player)
            self.autopilot.set_radar(self.radar_sensor)
        elif self.radar_sensor.sensor is not None:
            self.radar_sensor.destroy_radar()
            self.radar_sensor = None
            self.autopilot.set_radar(None)
           
    def tick(self, clock, hud_wintersim):
        self.hud_wintersim.tick(self, clock, hud_wintersim)

    def render_object_detection(self):
        ''' Render camera object detection if enabled, uses another thread'''
        if self.multiple_windows_enabled and self.multiple_window_setup:
            # if multiplewindows enabled and setup done, enable MultipleWindows thread flag
            self.cv2_windows.resume()

        if not self.multiple_window_setup and self.multiple_windows_enabled:
            # setup wintersim_camera_windows.py
            self.cv2_windows = CameraWindows(self.player, self.camera_manager.sensor, self.world, self.args.record, self.detection)
            self.autopilot.set_camera(self.cv2_windows)
            self.multiple_window_setup = True
            self.cv2_windows.start()
            self.cv2_windows.pause()

    def toggle_lidar(self, world, client):
        ''' Toggle lidar render/detection'''
        if world.record_data and not world.render_lidar_detection:                  # Resumes to lidar object detection      
            world.data_thread.make_lidar(world.player, world)                       # If theres no lidar lets make a new one
            world.render_lidar_detection = True
            client.get_world().apply_settings(world.settings)                       # apply custom settings
            world.data_thread.resume()                                              # resume object detection thread
            self.autopilot.set_lidar(True)

        if not world.record_data and world.render_lidar_detection:
            world.render_lidar_detection = False
            client.get_world().apply_settings(world.original_settings)              # set default settings
            world.data_thread.pause()                                               # pause object detection thread
            world.data_thread.destroy_lidar()
            self.autopilot.set_lidar(False)

    def toggle_autonomous_autopilot(self):
            self.wintersim_autopilot = not self.wintersim_autopilot

    def block_camera_object_detection(self):
        if self.multiple_windows_enabled and self.cv2_windows is not None:
            # if multiplewindows enabled, disable MultipleWindows thread flag
            self.cv2_windows.pause()

    def toggle_cv2_windows(self):
        self.multiple_windows_enabled = not self.multiple_windows_enabled
        if self.multiple_windows_enabled == False and self.cv2_windows is not None:
            self.cv2_windows.destroy()
            self.multiple_window_setup = False

    def render(self, display):
        self.camera_manager.render(display)
        self.hud_wintersim.render(display, self.world)

    def render_UI_sliders(self, world, client, hud_wintersim, display, weather):
        if hud_wintersim.is_hud:
            for s in hud_wintersim.sliders:
                if s.hit:
                    s.move()
                    weather.tick(hud_wintersim, world.preset[0])
                    client.get_world().set_weather(weather.weather)
            for s in hud_wintersim.sliders:
                s.draw(display, s)

    def update_friction(self, iciness):
        actors = self.world.get_actors()
        friction = 5
        friction -= iciness / 100 * 4
        for actor in actors:
            if 'vehicle' in actor.type_id:
                vehicle = actor
                front_left_wheel  = carla.WheelPhysicsControl(tire_friction=friction, damping_rate=1.3, max_steer_angle=70.0, radius=20.0)
                front_right_wheel = carla.WheelPhysicsControl(tire_friction=friction, damping_rate=1.3, max_steer_angle=70.0, radius=20.0)
                rear_left_wheel   = carla.WheelPhysicsControl(tire_friction=friction, damping_rate=1.3, max_steer_angle=0.0,  radius=20.0)
                rear_right_wheel  = carla.WheelPhysicsControl(tire_friction=friction, damping_rate=1.3, max_steer_angle=0.0,  radius=20.0)
                wheels = [front_left_wheel, front_right_wheel, rear_left_wheel, rear_right_wheel]
                physics_control = vehicle.get_physics_control()
                physics_control.wheels = wheels
                vehicle.apply_physics_control(physics_control)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        if self.radar_sensor is not None:
            self.toggle_radar()
        sensors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()


# ==============================================================================
# -- Keyboard ---------------------------------------------------------------
# ==============================================================================

class KeyboardControl(object):
    """Class that handles keyboard input."""
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._lights = carla.VehicleLightState.NONE
            #world.player.set_autopilot(self._autopilot_enabled)
            world.player.set_light_state(self._lights)
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud_wintersim.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, client, world, clock, hud_wintersim):
        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.MOUSEBUTTONDOWN: #slider event
                if world.hud_wintersim.is_hud:
                    pos = pygame.mouse.get_pos()
                    for slider in hud_wintersim.sliders:
                        if slider.button_rect.collidepoint(pos): #get slider what mouse is touching
                            slider.hit = True #slider is being moved
            elif event.type == pygame.MOUSEBUTTONUP: #slider event
                if world.hud_wintersim.is_hud:
                    if hud_wintersim.ice_slider.hit: #if road iciness is updated
                        world.update_friction(hud_wintersim.ice_slider.val)
                    for slider in hud_wintersim.sliders:
                        slider.hit = False #slider moving stopped
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_F1:
                    world.hud_wintersim.toggle_info(world)
                elif event.key == K_F2:
                    world.hud_wintersim.map.toggle()
                elif event.key == K_F4:
                    world.toggle_autonomous_autopilot()
                elif event.key == K_F8:
                    world.detection = True
                    world.toggle_cv2_windows()
                elif event.key == K_F9:
                    world.detection = False
                    world.toggle_cv2_windows()
                elif event.key == K_F10:
                    world.detection = True
                    world.toggle_cv2_windows()
                    world.toggle_radar()
                    world.record_data = not world.record_data
                    world.toggle_lidar(world, client)

                elif event.key == K_F12:
                    # toggle server rendering
                    game_world = client.get_world()
                    settings = game_world.get_settings()
                    settings.no_rendering_mode = not settings.no_rendering_mode
                    game_world.apply_settings(settings)

                elif event.key == K_a:
                    world.wintersim_autopilot = not world.wintersim_autopilot
                elif event.key == K_v and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_map_layer(reverse=True)
                elif event.key == K_v:
                    world.next_map_layer()
                elif event.key == K_b and pygame.key.get_mods() & KMOD_SHIFT:
                    world.load_map_layer(unload=True)
                elif event.key == K_b:
                    world.load_map_layer()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud_wintersim.help_text.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_g:
                    world.toggle_radar()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key == K_n:
                    world.camera_manager.next_sensor()
                elif event.key == K_o:
                    world.record_data = not world.record_data
                    world.toggle_lidar(world, client)
                elif event.key == K_w and (pygame.key.get_mods() & KMOD_CTRL):
                    if world.constant_velocity_enabled:
                        world.player.disable_constant_velocity()
                        world.constant_velocity_enabled = False
                        world.hud_wintersim.notification("Disabled Constant Velocity Mode")
                    else:
                        world.player.enable_constant_velocity(carla.Vector3D(17, 0, 0))
                        world.constant_velocity_enabled = True
                        world.hud_wintersim.notification("Enabled Constant Velocity Mode at 60 km/h")
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if (world.recording_enabled):
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud_wintersim.notification("Recorder is OFF")
                    else:
                        client.start_recorder("manual_recording.rec")
                        world.recording_enabled = True
                        world.hud_wintersim.notification("Recorder is ON")
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    # stop recorder
                    client.stop_recorder()
                    world.recording_enabled = False
                    # work around to fix camera at start of replaying
                    current_index = world.camera_manager.index
                    world.destroy_sensors()
                    # disable autopilot
                    self._autopilot_enabled = False
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud_wintersim.notification("Replaying file 'manual_recording.rec'")
                    # replayer
                    client.replay_file("manual_recording.rec", world.recording_start, 0, 0)
                    world.camera_manager.set_sensor(current_index)
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    world.hud_wintersim.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud_wintersim.notification("Recording start time is %d" % (world.recording_start))
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud_wintersim.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p and not pygame.key.get_mods() & KMOD_CTRL:
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud_wintersim.notification(
                            'Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_CTRL:
                        current_lights ^= carla.VehicleLightState.Special1
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_SHIFT:
                        current_lights ^= carla.VehicleLightState.HighBeam
                    elif event.key == K_l:
                        # Use 'L' key to switch between lights:
                        # closed -> position -> low beam -> fog
                        if not self._lights & carla.VehicleLightState.Position:
                            world.hud_wintersim.notification("Position lights")
                            current_lights |= carla.VehicleLightState.Position
                        else:
                            world.hud_wintersim.notification("Low beam lights")
                            current_lights |= carla.VehicleLightState.LowBeam
                        if self._lights & carla.VehicleLightState.LowBeam:
                            world.hud_wintersim.notification("Fog lights")
                            current_lights |= carla.VehicleLightState.Fog
                        if self._lights & carla.VehicleLightState.Fog:
                            world.hud_wintersim.notification("Lights off")
                            current_lights ^= carla.VehicleLightState.Position
                            current_lights ^= carla.VehicleLightState.LowBeam
                            current_lights ^= carla.VehicleLightState.Fog
                    elif event.key == K_i:
                        current_lights ^= carla.VehicleLightState.Interior
                    elif event.key == K_z:
                        current_lights ^= carla.VehicleLightState.LeftBlinker
                    elif event.key == K_x:
                        current_lights ^= carla.VehicleLightState.RightBlinker

        # if not self._autopilot_enabled or not self.wintersim_autopilot:
        #     if isinstance(self._control, carla.VehicleControl):
        #         self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time(), world)
        #         self._control.reverse = self._control.gear < 0
        #         # Set automatic control-related vehicle lights
        #         if self._control.brake:
        #             current_lights |= carla.VehicleLightState.Brake
        #         else: # Remove the Brake flag
        #             current_lights &= ~carla.VehicleLightState.Brake
        #         if self._control.reverse:
        #             current_lights |= carla.VehicleLightState.Reverse
        #         else: # Remove the Reverse flag
        #             current_lights &= ~carla.VehicleLightState.Reverse
        #         if current_lights != self._lights: # Change the light state only if necessary
        #             self._lights = current_lights
        #             world.player.set_light_state(carla.VehicleLightState(self._lights))
        #     world.player.apply_control(self._control)

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================

def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)        

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        display.fill((0,0,0))
        pygame.display.flip()

        hud_wintersim = wintersim_hud.HUD_WINTERSIM(args.width, args.height, display)
        hud_wintersim.make_sliders()
        world = World(client.get_world(), hud_wintersim, args)
        world.preset = world._weather_presets[0]                            # start weather preset
        hud_wintersim.update_sliders(world.preset[0])                       # update sliders to positions according to preset
        controller = KeyboardControl(world, args.autopilot)
        weather = wintersim_hud.Weather(client.get_world().get_weather())   # weather object to update carla weather with sliders
        clock = pygame.time.Clock()

        q = Queue()
        world.data_thread = LidarObjectDetection(q, args=(False))
        world.data_thread.start()
        world.render_lidar_detection = False
        world.dataLidar = None

        world.original_settings = client.get_world().get_settings()
        world.settings = client.get_world().get_settings()
        world.settings.fixed_delta_seconds = 0.05
        world.settings.synchronous_mode = True

        world.autopilot = Autopilot(world, world.data_thread)

        game_world =  client.get_world()
 
        while True:

            # if world.render_lidar_detection:
            #     clock.tick_busy_loop(20)
            # else:
            #     clock.tick_busy_loop(20)

            clock.tick_busy_loop(20)

            world.render_object_detection()

            if world.wintersim_autopilot:
                world.autopilot.tick_autopilot()

            if controller.parse_events(client, world, clock, hud_wintersim):
               return

            world.tick(clock, hud_wintersim)
            world.render(display)
            world.render_UI_sliders(world, client, hud_wintersim, display, weather)
            pygame.display.flip()

            if world.render_lidar_detection:
                game_world.tick()

    finally:
        if world.dataLidar is not None:
            world.dataLidar.destroy()   

        if world.data_thread is not None:    
            world.data_thread.pause()                                       
            #world.data_thread.destroy()

        if world.original_settings is not None:
            client.get_world().apply_settings(world.original_settings)

        if world.cv2_windows is not None:
            world.cv2_windows.destroy()

        if world is not None:
            world.destroy()

        pygame.quit()

# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================

def main():
    argparser = argparse.ArgumentParser(
        description='WinterSim')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='model3',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--detection',
        default=True,
        type=bool,
        help='detection')
    argparser.add_argument(
        '--record',
        default=False,
        type=bool,
        help='record cv2 windows')
    argparser.add_argument(
        '--scenario',
        default=False,
        type=bool,
        help='is scenario')
    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)
    print(__doc__)
    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

if __name__ == '__main__':
    main()