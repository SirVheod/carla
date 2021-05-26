#!/usr/bin/env python

# Copyright (c) 2021 FrostBit Software Lab

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>

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

import pygame
import math
import carla
# ==============================================================================
# -- INFO_HUD -------------------------------------------------------------
# ==============================================================================


class INFO_HUD(object):
    def __init__(self, width, height, display): #init hud
        self.dim = (width, height)
        self.screen = display
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self.snow_amount_slider = Slider #sliders for updating weather parameters
        self.ice_slider = Slider
        self.temp_slider = Slider
        self.rain_slider = Slider
        self.fog_slider = Slider
        self.wind_slider = Slider
        self.time_slider = Slider
        self.month_slider = Slider
        self.sliders = [] #slider list
        self._font_mono = pygame.font.Font(mono, 18 if os.name == 'nt' else 18)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self._info_text = []

    def make_sliders(self): #make sliders and add them in to list
        self.temp_slider = Slider("Temp", 0, 40, -40, 20)
        self.snow_amount_slider = Slider("Snow", 0, 100, 0, 140)
        self.ice_slider = Slider("Road Ice", 0, 100, 0, 260)
        self.rain_slider =Slider("Rain", 0, 100, 0, 380)
        self.fog_slider = Slider("Fog", 0, 100, 0, 500)
        self.wind_slider = Slider("Wind", 0, 100, 0, 620)
        self.time_slider = Slider("Time", 0, 24, 0, 740)
        self.month_slider = Slider("Month", 0, 11, 0, 860)
        self.sliders = [self.temp_slider, self.snow_amount_slider, self.ice_slider, self.rain_slider, self.fog_slider, self.wind_slider, self.time_slider, self.month_slider]

    def update_sliders(self, preset, month=None, clock=None): #update slider positions if weather is changed without moving sliders
        ##real values
        self.snow_amount_slider.val = preset.snow_amount
        self.ice_slider.val = preset.ice_amount
        self.temp_slider.val = preset.temperature
        self.rain_slider.val = preset.precipitation
        self.fog_slider.val = preset.fog_density
        self.wind_slider.val = preset.wind_intensity*100.0
        if month and clock:
            self.month_slider.val = month
            self.time_slider.val = clock
            self.month_slider.val_draw = month*2
            self.time_slider.val_draw = clock*2
        ##values that are used to draw sliders must be multiplied by 2
        self.snow_amount_slider.val_draw = preset.snow_amount*2
        if preset.ice_amount > 0:
            self.ice_slider.val_draw = preset.ice_amount*2
        self.temp_slider.val_draw = preset.temperature*2
        self.rain_slider.val_draw = preset.precipitation*2
        self.fog_slider.val_draw = preset.fog_density*2
        self.wind_slider.val_draw = preset.wind_intensity*100.0 *2

    def get_month(self, val): #get month name and sun position according to month number
        months = ['January','February','March','April','May','June','July','August','September','October','November','December']
        sun = [[12.5, 1.36, -43.6],[12.5, 9.25, -35.11],[12.5, 20.13, -24.24],[12.5, 31.99, -12.37],[12.5, 41.03, -2.74],[12.5, 45.39, 1.60],[12.5, 43.51, 0.05],[12.5, 35.97, -8.07],[12.5, 24.94, -19.04],[12.5, 13.44, -30.56],[12.5, 3.66, -40.75],[12.5, -0.56, -45.32]]
        return months[val], sun[val]

    def tick(self, world, clock, hud): #update hud text values
        self._notifications.tick(world, clock)
        month, sundata = self.get_month(int(hud.month_slider.val))
        self._info_text = [
            'Weather Control',
            '',
            '',
            'Temp:  {}°C'.format(round(hud.temp_slider.val,1)),
            '',
            'Amount of Snow:  {}cm'.format(round(hud.snow_amount_slider.val)),
            '',
            'Iciness:  {}.00%'.format(int(hud.ice_slider.val)),
            '',
            'Rain:  {}mm'.format(round((hud.rain_slider.val/10), 1)),
            '',
            'Fog:  {}%'.format(int(hud.fog_slider.val)),
            '',
            'Wind Intensity: {}m/s'.format(round((hud.wind_slider.val/10), 1)),
            '',
            'Time: {}.00'.format(int(hud.time_slider.val)),
            '',
            'Month: {}'.format(month),
            '',
            'Press M to get',
            'weather from Muonio']

    def notification(self, text, seconds=2.0): #notification about changing weather preset
        self._notifications.set_text(text, seconds=seconds)

    def render(self, display): #render hud texts into pygame window
        info_surface = pygame.Surface((270, self.dim[1]))
        info_surface.set_alpha(100)
        info_surface.fill((75, 75, 75))
        display.blit(info_surface, (0, 0))
        v_offset = 4           
        for item in self._info_text:
            surface = self._font_mono.render(item, True, (255, 255, 255))
            display.blit(surface, (8, v_offset))
            v_offset += 18
        self._notifications.render(display)


# ==============================================================================
# -- SliderObject -------------------------------------------------------------
# ==============================================================================


class Slider():
    def __init__(self, name, val, maxi, mini, pos):
        BLACK = (0, 0, 0)
        GREY = (200, 200, 200)
        ORANGE = (255, 183, 0)
        WHITE = (255, 255, 255)
        self.font = pygame.font.SysFont("ubuntumono", 16)
        self.name = name
        self.val = val  # slider start value
        self.val_draw = val 
        self.maxi = maxi # maximum at slider position right
        self.mini = mini # minimum at slider position left
        self.xpos = 280  # x-location on screen
        self.ypos = pos  # y-location on screen
        self.surf = pygame.surface.Surface((200, 100))
        self.hit = False  # the hit attribute indicates slider movement due to mouse interaction
        self.txt_surf = self.font.render(name, 1, BLACK)

        if name is "Temp": # temperature slider has different size than other sliders
            pygame.draw.rect(self.surf, ORANGE, [10, 70, 120, 1], 0)
            pygame.draw.rect(self.surf, WHITE, [10, 20, 120, 20], 3)
            pygame.draw.rect(self.surf, WHITE, [10, 20, 120, 20], 0)
            self.txt_rect = self.txt_surf.get_rect(center=(70, 30))
            width = 140
        else:
            pygame.draw.rect(self.surf, ORANGE, [10, 70, 160, 1], 0)
            pygame.draw.rect(self.surf, WHITE, [10, 20, 160, 20], 3)
            pygame.draw.rect(self.surf, WHITE, [10, 20, 160, 20], 0)
            self.txt_rect = self.txt_surf.get_rect(center=(90, 30))
            width = 180

        line_width = 1
        height = 100
        #borders
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
        pos = (10+int((self.val_draw-self.mini)/(self.maxi-self.mini)*80), 66)
        self.button_rect = self.button_surf.get_rect(center=pos)
        surf.blit(self.button_surf, self.button_rect)
        self.button_rect.move_ip(self.xpos, self.ypos)  # move of button box to correct screen position
        # screen
        screen.blit(surf, (self.xpos, self.ypos))

    def move(self):
        """The dynamic part; reacts to movement of the slider button."""
        self.val = (pygame.mouse.get_pos()[0] - self.xpos - 10) / 80 * (self.maxi - self.mini) + self.mini
        self.val_draw = self.val
        #these are the real values of sliders
        if self.val < self.mini:
            self.val = self.mini
        if self.val > 0:
            self.val = self.val /2
        if self.val > self.maxi:
            self.val = self.maxi
        #these are the values used to draw sliders
        if self.val_draw < self.mini:
            self.val_draw = self.mini
        if self.val_draw > self.maxi * 2:
            self.val_draw = self.maxi * 2

 
# ==============================================================================
# -- SunObject -------------------------------------------------------------
# ==============================================================================

class Sun(object):
    def __init__(self, azimuth, altitude):
        self.azimuth = azimuth
        self.altitude = altitude

    def SetSun(self, highest_time, sun_highest, sun_lowest, clock): #overal handler for sun altitude and azimuth
        if clock is highest_time:
            self.altitude = sun_highest
        elif clock < highest_time:
            D = highest_time - (highest_time - clock)
            X= float(D/highest_time)
            Y = math.sin(X*87*math.pi/180)
            A = sun_highest
            B = sun_lowest
            self.altitude = (Y * A) + ((1-Y) * B)
        else:
            D = highest_time - (clock - highest_time)
            X= float(D/highest_time)
            Y = math.sin(X*87*math.pi/180)
            A = sun_highest
            B = sun_lowest
            self.altitude = (Y * A) + ((1-Y) * B)
        self.azimuth = 348.98 + clock * 15
        if self.azimuth > 360: 
            self.azimuth -= 360    
    def __str__(self):
        return 'Sun(alt: %.2f, azm: %.2f)' % (self.altitude, self.azimuth)

# ==============================================================================
# -- WeatherObject -------------------------------------------------------------
# ==============================================================================


class Weather(object):
    def __init__(self, weather):
        self.weather = weather
        self.sun = Sun(weather.sun_azimuth_angle, weather.sun_altitude_angle) #instantiate sun object and pass angles 

    def tick(self, hud, preset): #this is called always when slider is being moved
        preset = preset[0]
        month, sundata = hud.get_month(int(hud.month_slider.val))
        clock = hud.time_slider.val #update sun time variable
        self.sun.SetSun(sundata[0],sundata[1],sundata[2], clock)
        self.weather.cloudiness = hud.rain_slider.val
        self.weather.precipitation = hud.rain_slider.val
        self.weather.precipitation_deposits = hud.rain_slider.val
        self.weather.wind_intensity = hud.wind_slider.val /100.0
        self.weather.fog_density = hud.fog_slider.val
        self.weather.wetness = preset.wetness
        self.weather.sun_azimuth_angle = self.sun.azimuth
        self.weather.sun_altitude_angle = self.sun.altitude
        self.weather.snow_amount = hud.snow_amount_slider.val
        self.weather.temperature = hud.temp_slider.val
        self.weather.ice_amount = hud.ice_slider.val

    def muonio_update(self, hud, temp, precipitation, wind, cloudiness, visibility, snow, clock, m):
        month, sundata = hud.get_month(m)
        self.sun.SetSun(sundata[0],sundata[1],sundata[2], clock)
        self.weather.cloudiness = cloudiness
        self.weather.precipitation = precipitation
        self.weather.precipitation_deposits = precipitation
        self.weather.wind_intensity = wind / 100.0
        self.weather.fog_density = visibility
        self.weather.wetness = 0
        self.weather.sun_azimuth_angle = self.sun.azimuth
        self.weather.sun_altitude_angle = self.sun.altitude
        self.weather.snow_amount = snow
        self.weather.temperature = temp
        self.weather.ice_amount = 0

    def __str__(self):
        return '%s %s' % (self._sun, self._storm)


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
