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
        options = {"model": "cfg/yolo.cfg", "load": "yolo.weights", "threshold": 0.5, "gpu": 0.6}                       # slowest but best accuracy, change load.py line 121 to 16
        #options = {"model": "cfg/yolov2.cfg", "load": "yolov2.weights", "threshold": 0.5, "gpu": 0.5}                  # slow, change load.py line 121 to 16
        #options = {"model": "cfg/yolov2-tiny-voc.cfg", "load": "yolov2-tiny.weights", "threshold": 0.5, "gpu": 0.5}    # fast, change load.py line 121 to 20

        tfnet = TFNet(options)
        color = (255, 0, 0)

    @staticmethod
    def detect_objects(img, image):
        '''Detect objects from image'''

        start_time = time.time()
       
        # Pass image to detector
        result = tfnet.return_predict(image)

        for index in range(len(result)):
            # Parse list of dictionaries
            label = result[index]['label']

            x = result[index]['topleft_x']
            y = result[index]['topleft_y']
            xw = result[index]['bottomright_x']
            yh = result[index]['bottomright_y']

            # uncomment to show: label + confidence (%)
            #confidence = result[index]['confidence']
            #conf = int(confidence * 100)
            #fulllabel = str(label + ' ' + str(conf) + '%')

            # Draw text and boxes around detected object
            cv2.rectangle(img, (x,y), (xw,yh), color, 2)
            cv2.putText(img, label, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # draw object detection fps to cv2 top left corner
        fps = int(1.0 / (time.time() - start_time))
        fps_text = str(fps) + " FPS"
        cv2.putText(img, fps_text, (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        #print("Object detection FPS: ", fps)

        return img