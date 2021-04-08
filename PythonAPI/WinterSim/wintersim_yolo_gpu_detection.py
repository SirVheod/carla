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
        global options, tfnet, color

        # load cfg and weight file
        options = {"model": "cfg/yolo.cfg", "load": "yolo.weights", "threshold": 0.5, "gpu": 0.5}                       # slowest but best accuracy, change load.py line 121 to 16
        #options = {"model": "cfg/yolov2.cfg", "load": "yolov2.weights", "threshold": 0.5, "gpu": 0.5}                  # slow, change load.py line 121 to 16
        #options = {"model": "cfg/yolov2-tiny-voc.cfg", "load": "yolov2-tiny.weights", "threshold": 0.5, "gpu": 0.5}    # fast, change load.py line 121 to 20

        tfnet = TFNet(options)
        color = (255, 0, 0)

    @staticmethod
    def DetectObjects(img, image):
        '''Detect objects from image'''

        # Pass image to detector
        result = tfnet.return_predict(image)

        for index in range(len(result)):
            # Parse list of dictionaries
            label = result[index]['label']
            confidence = result[index]['confidence']
            conf = int(confidence * 100)
            fulllabel = str(label + ' ' + str(conf) + '%')
            x = result[index]['topleft_x']
            y = result[index]['topleft_y']
            xw = result[index]['bottomright_x']
            yh = result[index]['bottomright_y']

            # Draw text and boxes around detected object
            cv2.rectangle(img, (x,y), (xw,yh), color, 2)
            cv2.putText(img, fulllabel, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return img