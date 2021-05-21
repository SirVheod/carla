import carla
import math

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

from object_detection import test_detection_dev as radar_object_detection
from object_detection import test_both_side_detection_dev as lidar_object_detection


"""
Welcome to WinterSim CARLA Autopilot.
Unlike Carla's own autopilot, this autopilot relies mostly on sensor(s) object detection information 
and maneuvers based on that information.

Distances to other actors are read from the simulation.

TODO:
- Better vehicle speed logic (currently vehicle accelerates unless speed is over max_autopilot_speed limit)
- Read Road speed limit?
- Get distances to detected object(s) from camera/lidar/radar?

"""

class Autopilot(object):
    def __init__(self, world, lidar):
        self.world = world
        self._control = carla.VehicleControl()
        self.lidar_detection = lidar

        self.emergency_break = False
        self.vehicle_is_stopped = False
        self.vehicle_is_moving = False
        self.vehicle_speed = 0
        self.vehicle_breaking = False
        self.detected_vehicle = False
        self.vehicle_in_front = False
        self.vehicle_behind = False
        self.radar_data = None
        self.camera_windows = None
        self.frame_counter = 0
        self.front_camera_enabled = False
        self.back_camera_enabled = False
        self.max_autopilot_speed = 40
        self.closest_distance_to_actor = 100000
        self.distances_to_actors = []
        self.other_actors = False

        self.lidar_detected_vehicle_in_front  = False
        self.lidar_detected_vehicle_behind  = False
        self.lidar_detected_frame_counter = 0
        self.lidar_detection_enabled = False

        self.radar_detected_vehicle_in_front  = False
        self.radar_detected_vehicle_behind  = False
        self.radar_detected_frame_counter = 0
        self.radar_detection = None

        self.radar_detected_vehicle = False
        self.camera_detected_vehicle = False

    def set_camera(self, camera_windows):
        ''' Set camera detection'''
        self.camera_windows = camera_windows
        self.front_camera_enabled = True

    def set_radar(self, radar):
        ''' Set radar detection'''
        self.radar_detection = radar

    def set_lidar(self, lidar):
        ''' Set lidar detection'''
        self.lidar_detection_enabled = lidar

    def parse_front_camera(self):
        ''' Parse front camera detection results'''
        if not self.front_camera_enabled:
            return

        if self.camera_windows is not None:
            result = self.camera_windows.get_latests_results()
            if result is not None:
                for index in range(len(result)):
                    label = result[index]['label']
                    # confidence = result[index]['confidence']
                    # conf = int(confidence * 100)
                    if label == "car" or label == "truck":
                        self.vehicle_in_front = True
                        self.camera_detected_vehicle = True

    def parse_lidar_data(self):
        ''' Parse Lidar detection results'''
        if not self.lidar_detection_enabled:
            return

        self.lidar_detected_vehicle_in_front, self.lidar_detected_vehicle_behind = lidar_object_detection.get_latest_results()

        # if self.lidar_detected_vehicle_behind:
        #     print("detected vehicle behind!")

        if self.lidar_detected_vehicle_in_front:
            self.lidar_detected_frame_counter += 1
        else:
            self.lidar_detected_frame_counter = 0

        if self.lidar_detected_frame_counter >= 3:
            # Lidar must have detected vehicle, 3 simulation frames in row or more
            # this ensures if lidar detection has 'hiccups' it will be ignored
            #print("Detected vehicle in front with lidar!")
            self.vehicle_in_front = True

    def parse_radar_data(self):
        ''' Parse Radar detection results'''
        if self.radar_detection is None:
            return

        self.radar_detected_vehicle_in_front = radar_object_detection.get_latest_results()
        radar_object_detection.reset()
        #print(self.radar_detected_vehicle_in_front)

        if self.radar_detected_vehicle_in_front:
            self.radar_detected_frame_counter += 1
        else:
            self.radar_detected_frame_counter = 0

        if self.radar_detected_frame_counter >= 3:
            # Radar must have detected vehicle 3 simulation frames in row
            # this ensures if ridar detection has 'hiccups' it will be ignored
            #print("Detected vehicle in front with radar!")
            #self.radar_detected_vehicle = True
            self.vehicle_in_front = True

    def calculate_distances_to_actors(self, world):
        '''Calculate distance to other actors
        @Todo: this doesn't take into account if vehicle is in front or back!
        '''
        t = world.player.get_transform()
        vehicles = world.world.get_actors().filter('vehicle.*')
        if len(vehicles) > 1:
            self.distances_to_actors.clear()
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0:
                    continue
                self.other_actors = True
                self.distances_to_actors.append(d)
                if d < self.closest_distance_to_actor:
                    self.closest_distance_to_actor = d
        else:
            self.other_actors = False
            

    def calculate_vehicle_speed(self, world):
        v = world.player.get_velocity()
        self.vehicle_speed = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        if self.vehicle_speed is not 0:
            self.vehicle_is_moving = True
            self.vehicle_is_stopped = False
        else:
            self.vehicle_is_moving = True
            self.vehicle_is_stopped = False
            self.emergency_break = False

    def reset_parameters(self):
        ''' Reset parameters from last frame'''
        self.vehicle_in_front = False
        self.closest_distance_to_actor = 10000
        self.radar_detected_vehicle = False
        self.camera_detected_vehicle = False

    def check_emergency_break(self):
        '''Check emergency break'''
        if not self.emergency_break and self.closest_distance_to_actor < 5 and not self.vehicle_breaking and not self.vehicle_is_stopped:
            self.emergency_break = True

        if self.closest_distance_to_actor < 15 and self.vehicle_speed > 40 and not self.vehicle_breaking and not self.vehicle_is_stopped:
            self.emergency_break = True

        if self.emergency_break:
            print("emergency breaking activated!")
            self._control.throttle = 0.0
            self._control.brake = min(self._control.brake + 0.2, 1)
            self.world.player.apply_control(self._control)
            self.reset_parameters()
            return

    def handle_throttle(self):
        if self.emergency_break:
            return

        #print(self.vehicle_in_front)
        if not self.vehicle_in_front:
            if self.vehicle_speed < self.max_autopilot_speed:
                self._control.throttle = min(self._control.throttle + 0.03, 1)
            else:
                self._control.throttle -= 0.04
                if self._control.throttle < 0.0:
                    self._control.throttle = 0
        else:
            if self.other_actors:
                    # if distance to other actor over 25 m, don't break yet
                    if self.closest_distance_to_actor > 25.0:   
                        return 
                    else:
                        self._control.brake = min(self._control.brake + 0.01, 1)
            else:
                # no other actors to be found..
                #print("dist: ", self.closest_distance_to_actor)
                self._control.throttle = 0.0
                self._control.brake = min(self._control.brake + 0.2, 1)
        
    def tick_autopilot(self):
        '''Tick autopilot'''

        self.calculate_distances_to_actors(self.world)
        self.calculate_vehicle_speed(self.world)
        self.parse_front_camera()
        self.parse_lidar_data()
        self.parse_radar_data()

        self.check_emergency_break()
        self.handle_throttle()
    
        self.world.player.apply_control(self._control)
        self.reset_parameters()