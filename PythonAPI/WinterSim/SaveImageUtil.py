# WinterSim save image utility
import glob
import cv2
import sys
import os

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
    def save_single_image(filename, img):
        '''Save image to disk.'''
        cv2.imwrite(os.path.join(path , filename + '.png'), img)

    
    def clear_images():
        test = os.listdir(path)
        for images in test:
            if images.endswith(".jpg"):
                os.remove(os.path.join(path, images))

    def save_images_to_video():
        image_folder = path
        video_name = 'mygeneratedvideo.avi'
        
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