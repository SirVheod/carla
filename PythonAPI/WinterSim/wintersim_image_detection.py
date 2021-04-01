import cv2

class ImageDetection():

    dataset = None

    @staticmethod
    def Initialize():
        '''Initilize image detection class. Load .xml dataset.'''
        global dataset
        dataset = cv2.CascadeClassifier('stop_data.xml')

        if dataset is None:
            print("dataset is null!")

    @staticmethod
    def DetectObjects(img):
        '''Detect objects from input RGB image.'''

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        found = dataset.detectMultiScale(img_gray, minSize=(20, 20))

        # Don't do anything if there's no detections
        amount_found = len(found)
        if amount_found != 0:
            
            # Loop through all detections
            for (x, y, width, height) in found:
                
                # Draw a green rectangle around detected object
                cv2.rectangle(img, (x, y), (x + height, y + width), (0, 255, 0), 5)

        return img