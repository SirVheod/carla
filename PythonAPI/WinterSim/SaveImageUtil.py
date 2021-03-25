# WinterSim save image utility
import os
import cv2
import sys

class SaveImageUtil():

    path = ""
    
    @staticmethod
    def initialize():
        '''Initilize image utility. Make images folder to current script folder if it doesn't already exits'''
        if not os.path.exists('images'):
            os.makedirs('images')
            
        script_path = os.path.dirname(os.path.realpath(__file__))
        global path
        path = script_path + "/images"
        print(path)

    @staticmethod
    def save_image(filename, img):
        '''Save image to disk.'''
        cv2.imwrite(os.path.join(path , filename + '.png'), img)

    @staticmethod
    def set_custom_path(custom_path):
        '''set custom path.'''
        if os.path.isdir(custom_path):
            global path
            path = custom_path