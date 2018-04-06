

from carla.agent.agent import Agent
from carla.client import VehicleControl

class Forward(Agent):
    """
    Sample redefinition of the Agent,
    An agent that goes straight
    """
    def run_step(self, measurements, sensor_data, directions, target):
        control = VehicleControl()
        control.throttle = 0.9

        return control
