# test script

from SaveImageUtil import SaveImageUtil as save
import cv2
import os

img = cv2.imread('C:/Carla/carla/PythonAPI/WinterSim/muonio_map.png', 1)

save.initialize()

save.save_image("test", img)