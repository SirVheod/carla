from darkflow.net.build import TFNet
import numpy as np
import time
import cv2

class ImageDetection():

    options = None
    tfnet = None
    color = None

    @staticmethod
    def Initialize():
        '''Initialize ImageDetection class. 
        Call this before calling detect_objects function'''
        global options, tfnet, color

        # load cfg, weight file, set threshold and set GPU memory allocation.
        # gpu value means: allocated memory = available GPU memory * value
        # Example if you have GPU with 8 GB of VRAM and set 0.5 as value, 4GB of VRAM will be allocated.
        options = {"model": "config/yolo.cfg", "load": "config/yolo.weights", "threshold": 0.5, "gpu": 0.2}                       # slowest but best accuracy, change load.py line 121 to 16
        #options = {"model": "config/yolov2.cfg", "load": "config/yolov2.weights", "threshold": 0.5, "gpu": 0.5}                  # slow, change load.py line 121 to 16
        #options = {"model": "config/yolov2-tiny-voc.cfg", "load": "config/yolov2-tiny.weights", "threshold": 0.5, "gpu": 0.5}    # fast, change load.py line 121 to 20

        tfnet = TFNet(options)
        color = (255, 0, 0)

    @staticmethod
    def detect_objects(img, image, callback):
        '''Detect objects from input image.'''

        start_time = time.time()
       
        # Pass image to detector
        result = tfnet.return_predict(image)

        if callback is not None:
            callback(result)

        for index in range(len(result)):
            label = result[index]['label']
            x = result[index]['topleft_x']
            y = result[index]['topleft_y']
            xw = result[index]['bottomright_x']
            yh = result[index]['bottomright_y']
            
            # Draw text and boxes around detected object
            cv2.rectangle(img, (x,y), (xw,yh), color, 2)
            cv2.putText(img, label, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # draw object detection fps to cv2 top left corner
        # fps = int(1.0 / (time.time() - start_time))
        # fps_text = str(fps) + " FPS"
        # cv2.putText(img, fps_text, (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return img, result