from operator import truediv
import carla
import math

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

"""
Welcome to WinterSim CARLA Autopilot.
Unlike Carla's own autopilot, 
this autopilot relies purely on sensor(s) object detection information and maneuvers based on that information.
"""

def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

class Autopilot(object):
    def __init__(self, world):
        self.world = world
        self._control = carla.VehicleControl()
        self.emergency_break = False
        self.vehicle_is_stopped = False
        self.vehicle_is_moving = False
        self.vehicle_speed = 0
        self.vehicle_breaking = False
        self.detected_vehicle = False
        self.vehicle_in_front = False
        self.vehicle_behind = False
        self.detected_pedestrian = False
        self.radar_data = None
        self.cv2_windows = None
        self.frame_counter = 0
        self.last_frame_vehicle_in_front = False
        self.front_camera_enabled = False
        self.back_camera_enabled = False
        self.lidar_detection_enabled = False
        self.radar_detection_enabled = False
        self.radar_enabled = False
        self.max_autopilot_speed = 40
        self.closest_distance_to_actor = 0
        self.other_actors = False

    def set_camera(self, cv2_windows):
        ''' Set camera detection'''
        self.cv2_windows = cv2_windows
        self.front_camera_enabled = True
    
    def parse_front_camera(self):
        ''' Parse front camera detection results'''
        if not self.front_camera_enabled:
            return

        if self.cv2_windows is not None:
            result = self.cv2_windows.get_latests_results()
            if result is not None:
                for index in range(len(result)):
                    label = result[index]['label']
                    # confidence = result[index]['confidence']
                    # conf = int(confidence * 100)
                    if label == "car" or label == "truck":
                        self.vehicle_in_front = True
                        #print("detected vehicle in front!")


    def calculate_distance(self, world):
        '''Calculate distance to other actors'''
        t = world.player.get_transform()
        vehicles = world.world.get_actors().filter('vehicle.*')
        if len(vehicles) > 1:
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0:
                    break
                #vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self.closest_distance_to_actor = d
                self.other_actors = True
                #print('% 4dm %s' % (d, vehicle_type))
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

    def reset_parameters(self):
        ''' Reset parameters from last frame'''
        self.vehicle_in_front = False

    def tick_autopilot(self, clock):
        self.calculate_distance(self.world)
        self.calculate_vehicle_speed(self.world)
        self.parse_front_camera()

        # check worst case first
        if self.emergency_break:
            self._control.throttle = 0.0
            self._control.brake = min(self._control.brake + 0.2, 1)
            self.world.player.apply_control(self._control)
            self.reset_parameters()
            return


        if not self.vehicle_in_front:
            if self.vehicle_speed < self.max_autopilot_speed:
                self._control.throttle = min(self._control.throttle + 0.03, 1)
            else:
                self._control.throttle -= 0.04
                if self._control.throttle < 0.0:
                    self._control.throttle = 0
        else:
            if self.other_actors:
                    if self.closest_distance_to_actor > 25.0:
                        return # if distance to other actor over 25 m, don't break yet
                    else:
                        self._control.brake = min(self._control.brake + 0.01, 1)

            else:
                print("dist: ", self.closest_distance_to_actor)
                self._control.throttle = 0.0
                self._control.brake = min(self._control.brake + 0.2, 1)


        self.world.player.apply_control(self._control)
        self.reset_parameters()