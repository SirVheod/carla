import carla
from carla import ColorConverter as cc
import threading
from data_collector import collector_dev as collector
from data_collector.bounding_box import create_kitti_datapoint
from data_collector.constants import *
from data_collector import image_converter
from data_collector.dataexport import *
from matplotlib import cm
import open3d as o3d
from object_detection import test_both_side_detection_dev as object_detection
#import cv2

class LidarObjectDetection(threading.Thread):
    def __init__(self, queue, args=(), kwargs=None):
        threading.Thread.__init__(self, args=(), kwargs=None)
        self.opt, self.model, self.tensor = object_detection.main()
        self.queue = queue
        self.daemon = True
        self.paused = args
        self.state = threading.Condition()
        self.data = None
        self.lidar = None
        self.latest_detections = None

    def run(self):
        self.pause()
        while True:
            with self.state:
                if self.paused:
                    self.state.wait()   # Block execution until notified.
            if self.data is not None:   # if there is data we run this
                data = np.copy(np.frombuffer(self.data.raw_data, dtype=np.dtype('f4')))
                data = np.reshape(data, (int(data.shape[0] / 4), 4))
                points = data[:, :-1]
                self.data = None
                lidar_array = [[point[0], -point[1], point[2], 1.0] for point in points]
                lidar_array = np.array(lidar_array).astype(np.float32).reshape(-1, 4)
                if points.any() and len(points) > 0:
                    object_detection.detect(self.opt, self.model, lidar_array, self.tensor) # do the detection magick

    def pause(self):
        with self.state:
            self.paused = True      # Block self.

    def resume(self):
        with self.state:
            self.paused = False
            self.state.notify()     # Unblock self if waiting.

    def update(self, pointcloud):
        '''updates data to object detection thread'''
        with self.state:
            self.data = pointcloud

    def make_lidar(self, player, world):
        self.lidar = self.spawn_lidar(player, world)
        self.lidar.listen(lambda data: self.update(data))

    def destroy_lidar(self):
        self.lidar.destroy()
        self.lidar = None
        object_detection.destroy_window()

    def spawn_lidar(self, player, world): # this is the lidar we are using for object detection
        blueprint_library = world.world.get_blueprint_library()
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '50')
        lidar_bp.set_attribute('rotation_frequency', '20')
        lidar_bp.set_attribute('upper_fov', '2')
        lidar_bp.set_attribute('lower_fov', '-26.8')
        lidar_bp.set_attribute('points_per_second', '320000')
        lidar_bp.set_attribute('channels', '32')
        transform_sensor = carla.Transform(carla.Location(x=0, y=0, z=2.3))
        my_lidar = world.world.spawn_actor(lidar_bp, transform_sensor, attach_to=player)
        return my_lidar