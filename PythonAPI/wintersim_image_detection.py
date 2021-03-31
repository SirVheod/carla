from matplotlib import pyplot as plt
import cv2

class ImageDetectionTest():

    dataset = None

    @staticmethod
    def initialize():
        global dataset
        dataset = cv2.CascadeClassifier('stop_data.xml')

        #plt.ioff()  # Use non-interactive mode.
        #plt.plot([0, 1])  # You won't see the figure yet.
        #lt.show()  # Show the figure. Won't return until the figure is closed.
        #print("is open")


    @staticmethod
    def Exit():
        plt.close()

    @staticmethod
    def DetectFromImg(img):
        '''Detect vehicles and pedestrians from image'''

        plt.clf()

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        found = dataset.detectMultiScale(img_gray, minSize =(20, 20))

        # Don't do anything if there's 
        # no sign
        amount_found = len(found)
        
        if amount_found != 0:
            # There may be more than one
            # sign in the image
            for (x, y, width, height) in found:
                
                # We draw a green rectangle around
                # every recognized sign
                cv2.rectangle(img_rgb, (x, y), 
                            (x + height, y + width), 
                            (0, 255, 0), 5)
                
        # Creates the environment of 
        # the picture and shows it
        plt.subplot(1, 1, 1)

        plt.imshow(img_rgb)
        #print("Pausing...")
        #plt.pause(5)
        #plt.show()