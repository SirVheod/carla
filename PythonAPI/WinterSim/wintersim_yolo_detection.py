from matplotlib import pyplot as plt
import numpy as np
import time
import cv2

class ImageDetection():

    classes = None
    net = None
    layer_names = None
    output_layers = None
    COLORS = None

    @staticmethod
    def Initialize():
        global classes, net, layer_names, output_layers, COLORS

        weights = "yolov3_tiny.weights"
        config = "yolov3_tiny.cfg"

        with open("yolov3.txt", 'r') as f:
            classes = [line.strip() for line in f.readlines()]

        net = cv2.dnn.readNet(weights, config)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    @staticmethod
    def DetectObjects(img, image):
        '''Detect objects from image'''

        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392

        blob = cv2.dnn.blobFromImage(image, scale, (320, 320), (0, 0, 0), True, crop=False) # 416
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        for i in indices:
            i = i[0]
            box = boxes[i]
            x = round(box[0])
            y = round(box[1])
            w = box[2]
            h = box[3]
            xw = round(x+w)
            yh = round(y+h)
            label = str(classes[class_ids[i]])
            color = COLORS[class_ids[i]]
            cv2.rectangle(img, (x,y), (xw,yh), color, 2)
            cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return img