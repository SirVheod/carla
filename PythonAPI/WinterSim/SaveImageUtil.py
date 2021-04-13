# WinterSim save image utility
import glob
import cv2
import sys
import os
import time

class SaveImageUtil():

    path = ""
    
    @staticmethod
    def initialize():
        '''Initilize image utility.'''
        if not os.path.exists('images'):
            os.makedirs('images')
            
        script_path = os.path.dirname(os.path.realpath(__file__))
        global path
        #path = script_path + "/images"
        path = script_path + "\\images/images" + '_' + str(time.time()) + '/'
        os.mkdir(path)

    @staticmethod
    def save_single_image(filename, img):
        '''Save image to disk.'''
        cv2.imwrite(os.path.join(path , filename + '.jpg'), img)

    def save_images_to_video():
        image_folder = path
        video_name = 'video' + str(time.time()) + '.avi'

        images = [img for img in os.listdir(image_folder)
                if img.endswith(".jpg") or img.endswith(".jpeg") or img.endswith("png")]

        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name,cv2.VideoWriter_fourcc(*'DIVX'), 7, (width, height))
    
        # Appending the images to the video one by one
        for image in images: 
            video.write(cv2.imread(os.path.join(image_folder, image))) 
        
        # Deallocating memories taken for window creation
        video.release()
        
        # delete all images from /images folder
        # test = os.listdir(path)
        # for images in test:
        #     if images.endswith(".jpg"):
        #         os.remove(os.path.join(path, images))