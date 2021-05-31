#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#

# Copyright (c) 2021 FrostBit Software Lab

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
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
    F8           : spawn  separate front and back RGB camera windows
    H/?          : toggle help
    ESC          : quit;
"""

import glob
import os
import sys
import re
import time
import numpy as np

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
import wintersim_control
import carla

# ==============================================================================
# -- HUD_WINTERSIM -------------------------------------------------------------
# ==============================================================================

class HUD_WINTERSIM(object):
    def __init__(self, width, height, display):
        self.dim = (width, height)
        self.screen = display
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self.snow_amount_slider = Slider #sliders
        self.ice_slider = Slider
        self.temp_slider = Slider
        self.rain_slider = Slider
        self.fog_slider = Slider
        self.wind_slider = Slider
        self.sliders = []
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help_text = HelpText(pygame.font.Font(mono, 16), width, height, self)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self.is_hud = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()
        self.logo = pygame.image.load('WinterSim_White_Color.png')
        self.logo = pygame.transform.scale(self.logo, (262,61))
        self.logo_rect = self.logo.get_rect()
        self.make_sliders()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def make_sliders(self):
        self.snow_amount_slider = Slider("Snow", 0, 100, 0, 240)
        self.ice_slider = Slider("Road Ice", 0, 100, 0, 370)
        self.temp_slider = Slider("Temp", 0, 40, -40, 500)
        self.rain_slider =Slider("Rain", 0, 100, 0, 630)
        self.fog_slider = Slider("Fog", 0, 100, 0, 760)
        self.wind_slider = Slider("Wind", 0, 100, 0, 890)
        self.sliders = [self.snow_amount_slider, self.ice_slider, self.temp_slider, self.rain_slider, self.fog_slider, self.wind_slider]

    def update_sliders(self, preset):
        '''Initialize sliders'''
        self.snow_amount_slider.val = preset.snow_amount
        self.ice_slider.val = preset.ice_amount
        self.temp_slider.val = preset.temperature
        self.rain_slider.val = preset.precipitation
        self.fog_slider.val = preset.precipitation/2
        self.wind_slider.val = preset.wind_intensity*100.0

    def tick(self, world, clock, hud_wintersim):
        '''tick hud'''
        self._notifications.tick(world, clock)

        if not self.is_hud:
            return
        
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        compass = world.imu_sensor.compass
        heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        heading += 'S' if 90.5 < compass < 269.5 else ''
        heading += 'E' if 0.5 < compass < 179.5 else ''
        heading += 'W' if 180.5 < compass < 359.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'WinterSim Control',
            '',
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Amount of Snow:  {}'.format(int(hud_wintersim.snow_amount_slider.val)),
            'Iciness:  {}.00%'.format(int(hud_wintersim.ice_slider.val)),
            'Temp:  {}°C'.format(int(hud_wintersim.temp_slider.val)),
            'Rain:  {}%'.format(int(hud_wintersim.rain_slider.val)),
            'Fog:  {}%'.format(int(hud_wintersim.fog_slider.val)),
            'Wind Intensity {}%'.format(int(hud_wintersim.wind_slider.val)),
            '',
            'Vehicle: % 20s' % wintersim_control.get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
            'Accelero: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.accelerometer),
            'Gyroscop: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.gyroscope),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0:
                    break
                vehicle_type = wintersim_control.get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))
                
    def toggle_info(self, world):
        if self.is_hud:
            if self.help_text.visible:
                self.help_text.toggle()
            self.is_hud = not self.is_hud
        else:
            self.is_hud = not self.is_hud

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display, world):
        if self.is_hud:
            display_rect = display.get_rect()
            self.logo_rect.topright = tuple(map(lambda i, j: i - j, display_rect.topright, (5,-5))) 
            display.blit(self.logo, self.logo_rect)
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(200)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106            
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help_text.render(display)

# ==============================================================================
# -- SliderObject -------------------------------------------------------------
# ==============================================================================

class Slider():
    def __init__(self, name, val, maxi, mini, pos):
        BLACK = (0, 0, 0)
        GREY = (200, 200, 200)
        ORANGE = (255, 183, 0)
        WHITE = (255, 255, 255)
        self.font = pygame.font.SysFont("ubuntumono", 14)
        self.name = name
        self.val = val      # start value
        self.maxi = maxi    # maximum at slider position right
        self.mini = mini    # minimum at slider position left
        self.xpos = pos     # x-location on screen
        self.ypos = 20
        self.surf = pygame.surface.Surface((100, 50))
        self.hit = False    # the hit attribute indicates slider movement due to mouse interaction

        self.txt_surf = self.font.render(name, 1, BLACK)
        self.txt_rect = self.txt_surf.get_rect(center=(50, 15))

        # Static graphics - slider background #
        pygame.draw.rect(self.surf, WHITE, [10, 10, 80, 10], 3)
        pygame.draw.rect(self.surf, WHITE, [10, 10, 80, 10], 0)
        pygame.draw.rect(self.surf, ORANGE, [10, 35, 80, 1], 0)
        #borders
        line_width = 1
        width = 100
        height = 50
        # top line #first = starting point on width, second = starting point on height, third = width, fourth = height
        pygame.draw.rect(self.surf, WHITE, [0,0,width,line_width])
        # bottom line
        pygame.draw.rect(self.surf, WHITE, [0,height-line_width,width,line_width])
        # left line
        pygame.draw.rect(self.surf, WHITE, [0,0,line_width, height])
        # right line
        pygame.draw.rect(self.surf, WHITE, [width-line_width,0,line_width, height+line_width])

        self.surf.blit(self.txt_surf, self.txt_rect)  # this surface never changes
        self.surf.set_alpha(200)

        # dynamic graphics - button surface #
        self.button_surf = pygame.surface.Surface((20, 40))
        self.button_surf.fill((1, 1, 1))
        self.button_surf.set_colorkey((1, 1, 1))
        pygame.draw.rect(self.button_surf, WHITE, [6,15,6,15], 0)

    def draw(self, screen, slider):
        """ Combination of static and dynamic graphics in a copy ofthe basic slide surface"""
        # static
        surf = self.surf.copy()
        # dynamic
        pos = (10+int((self.val-self.mini)/(self.maxi-self.mini)*80), 33)
        self.button_rect = self.button_surf.get_rect(center=pos)
        surf.blit(self.button_surf, self.button_rect)
        self.button_rect.move_ip(self.xpos, self.ypos)  # move of button box to correct screen position
        # screen
        screen.blit(surf, (self.xpos, self.ypos))

    def move(self):
        """The dynamic part; reacts to movement of the slider button."""
        self.val = (pygame.mouse.get_pos()[0] - self.xpos - 10) / 80 * (self.maxi - self.mini) + self.mini
        if self.val < self.mini:
            self.val = self.mini
        if self.val > self.maxi:
            self.val = self.maxi

# ==============================================================================
# -- WeatherObject -------------------------------------------------------------
# ==============================================================================

class Weather(object):
    def __init__(self, weather):
        self.weather = weather

    def tick(self, hud_wintersim, preset):
        self.weather.cloudiness = hud_wintersim.rain_slider.val
        self.weather.precipitation = hud_wintersim.rain_slider.val
        self.weather.precipitation_deposits = hud_wintersim.rain_slider.val
        self.weather.wind_intensity = hud_wintersim.wind_slider.val /100.0
        self.weather.fog_density = hud_wintersim.fog_slider.val
        self.weather.wetness = preset.wetness
        self.weather.sun_azimuth_angle = preset.sun_azimuth_angle
        self.weather.sun_altitude_angle = preset.sun_altitude_angle
        self.weather.snow_amount = hud_wintersim.snow_amount_slider.val
        self.weather.temperature = hud_wintersim.temp_slider.val
        self.weather.ice_amount = hud_wintersim.ice_slider.val

    # def __str__(self):
    #     return '%s %s' % (self._sun, self._storm)

# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================

class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)

# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================

class HelpText(object):
    """Helper class to handle text output using pygame"""
    def __init__(self, font, width, height, hud):
        lines = __doc__.split('\n')
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self.visible = False
        self.surface.set_alpha(220)

    def toggle(self):
        self.visible = not self.visible

    def render(self, display):
        if self.visible:
            display.blit(self.surface, self.pos)