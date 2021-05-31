#!/usr/bin/env python

# Copyright (c) 2021 FrostBit Software Lab

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import print_function
import glob
import os
import sys
import re
import argparse
import math
import weather_hud

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from fmiopendata.wfs import download_stored_query #ilmatieteenlaitos library

try:
    import pygame
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_c
    from pygame.locals import K_m
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

# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================

class World(object):
    def __init__(self, carla_world, hud, args):
        self.world = carla_world
        self.ud_friction = True
        self.hud = hud
        self.preset = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._gamma = args.gamma

    def next_weather(self, world, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        self.preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % self.preset[1])
        self.hud.update_sliders(self.preset[0])
        self.world.set_weather(self.preset[0])

    def muonio_weather(self, world):
        weather = weather_hud.Weather(world.world.get_weather()) #create weather object
        obs = download_stored_query("fmi::observations::weather::multipointcoverage", args=["place=muonio"]) #get weather data from muonio

        obs_daily = download_stored_query("fmi::observations::weather::daily::multipointcoverage", args=["place=muonio"]) #daily
        
        latest_tstep = max(obs.data.keys()) #latest data
        latest_daily = max(obs_daily.data.keys())
        x = str(latest_tstep).split(" ") #split date and time

        date = x[0].split("-")
        year = int(date[0])
        day = int(date[2])
        month = int(date[1]) - 1 #-1 because with this number we get month from array so it has to be 0-11

        clock = x[1].split(":")
        clock.pop(2)
        clock = float(".".join(clock))

        temp = obs.data[latest_tstep]["Muonio kirkonkylä"]["Air temperature"]['value']

        precipitation = obs_daily.data[latest_daily]["Muonio kirkonkylä"]["Precipitation amount"]['value'] #use daily precipitation value for rain amount
        precipitation = 0 if math.isnan(precipitation) or precipitation is -1 else precipitation #this can be nan or -1 so that would give as error later so let make it 0 in this situation
        precipitation = 10 if precipitation > 10 else precipitation #max precipitation value is 10
        precipitation *= 10 #max precipitation is 10mm multiply by it 10 to get in range of 0-100
        
        wind = obs.data[latest_tstep]["Muonio kirkonkylä"]["Wind speed"]['value']
        wind = 0 if math.isnan(wind) else wind
        wind *= 10 #lets make 10m/s max wind value. Multiply wind by 10 to get it into range of 0-100

        cloudiness = obs.data[latest_tstep]["Muonio kirkonkylä"]["Cloud amount"]['value']
        cloudiness *= 12.5 #max value is 8 so we have to multiply it by 12.5 to get it into range of 0-100

        snow = obs.data[latest_tstep]["Muonio kirkonkylä"]["Snow depth"]['value']
        snow = 100 if snow > 100 else snow #lets set max number of snow to 1meter
        snow = 0 if math.isnan(snow) else snow

        weather.muonio_update(self.hud, temp, precipitation, wind, cloudiness, 0, snow, clock, month) #update weather object with our new data

        self.hud.notification('Weather: Muonio Realtime') #this is notification about weather preset update
        self.hud.update_sliders(weather.weather, month=month, clock=clock) #update sliders positions
        self.world.set_weather(weather.weather) #update weather in simulation

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

    def tick(self, clock, hud): #here we update huds data
        self.hud.tick(self, clock, hud)

    def render(self, display): #and here we render the hud
        self.hud.render(display)

# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================

class KeyboardControl(object):
    """Class that handles keyboard input."""
    def parse_events(self, client, world, clock, hud):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.MOUSEBUTTONDOWN: #slider event
                pos = pygame.mouse.get_pos()
                for slider in hud.sliders:
                    if slider.button_rect.collidepoint(pos): #get slider what mouse is touching
                        slider.hit = True #slider is being moved
            elif event.type == pygame.MOUSEBUTTONUP: #slider event
                if hud.ice_slider.hit: #if road iciness slider is moved
                    world.update_friction(hud.ice_slider.val)
                for slider in hud.sliders:
                    slider.hit = False #slider moving stopped
            elif event.type == pygame.KEYUP:
                if event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT: #previous weather preset
                    world.next_weather(world, reverse=True)
                elif event.key == K_c:
                    world.next_weather(world, reverse=False) #next weather preset
                elif event.key == K_m:
                    world.muonio_weather(world) #get muonios latest weather

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

        display = pygame.display.set_mode((args.width, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        display.fill((0,0,0))
        pygame.display.flip()

        hud = weather_hud.INFO_HUD(args.width, args.height, display)    # hud where we show numbers and all that 
        world = World(client.get_world(), hud, args)                    # instantiate our world object
        controller = KeyboardControl()                                  # controller for changing weather presets
        weather = weather_hud.Weather(client.get_world().get_weather()) # weather object to update carla weather with sliders
        hud.update_sliders(weather.weather)                             # update sliders according to preset parameters
        clock = pygame.time.Clock()

        game_world =  client.get_world()

        while True:
            clock.tick_busy_loop(30)
            if controller.parse_events(client, world, clock, hud): 
                return
            world.tick(clock, hud)
            world.render(display)

            for slider in hud.sliders:
                if slider.hit:
                    slider.move()
                    weather.tick(hud, world._weather_presets[0])
                    game_world.set_weather(weather.weather)
                    
            for slider in hud.sliders:
                slider.draw(display, slider)

            pygame.display.flip()

    finally:
        pygame.quit()

# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================

def main():
    argparser = argparse.ArgumentParser(
        description='WinterSim')
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
        '--res',
        metavar='WIDTHxHEIGHT',
        default='500x1000',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]
    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

if __name__ == '__main__':
    main()