from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters import StateWrapper
import math
import numpy as np


class CustomStateSetterExo1(StateSetter):
    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        for car in state_wrapper.cars:
            if car.team_num == 0:
                desired_car_pos = [0, 0, 17]  # x, y, z
                yaw = 0
            elif car.team_num == 1:
                desired_car_pos = [0, 0, 17]  # x, y, z
                yaw = 0

        car.set_pos(*desired_car_pos)
        car.set_rot(yaw=yaw)
        car.boost = 0

        # Now we will spawn the ball in the center of the field, floating in the air.
        state_wrapper.ball.set_pos(x=0, y=0,
                                   z=193)  # height_ball=153.0 mais le rayon de la balle est 93.0u  & hitbox_car = 36 so ball spawn at 189u & #max height 2044
        state_wrapper.ball.set_lin_vel(50, 0, 0)  # la vitesse de la balle dans les trois dimensions de l'espace)
        state_wrapper.ball.set_ang_vel(0, 0,
                                       0)  # la vitesse de rotation de la balle autour de chacun de ses axes (en roulis, tangage et lacet)


class CustomStateSetterExo2(StateSetter):
    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        for car in state_wrapper.cars:
            if car.team_num == 0:
                desired_car_pos = [0, 0, 17]  # x, y, z
                yaw = 0
            elif car.team_num == 1:
                desired_car_pos = [0, 0, 17]  # x, y, z
                yaw = 0

        car.set_pos(*desired_car_pos)
        car.set_rot(yaw=yaw)
        car.boost = 0

        state_wrapper.ball.set_pos(x=0, y=0,
                                   z=800) 
        state_wrapper.ball.set_lin_vel(200, 0, 0) 
        state_wrapper.ball.set_ang_vel(0, 0,
                                       0)  


class CustomStateSetterExo3(StateSetter):
    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        for car in state_wrapper.cars:
            if car.team_num == 0:
                desired_car_pos = [0, 0, 17]  # x, y, z
                yaw = 0
            elif car.team_num == 1:
                desired_car_pos = [0, 0, 17]  # x, y, z
                yaw = 0

        car.set_pos(*desired_car_pos)
        car.set_rot(yaw=yaw)
        car.boost = 0

        state_wrapper.ball.set_pos(x=-400, y=0,
                                   z=800)  
        state_wrapper.ball.set_lin_vel(200, 0, 0) 
        state_wrapper.ball.set_ang_vel(0, 0,
                                       0)


class CustomStateSetterExo4(StateSetter):
    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        for car in state_wrapper.cars:
            if car.team_num == 0:
                desired_car_pos = [0, 0, 17]  # x, y, z
                yaw = 0
            elif car.team_num == 1:
                desired_car_pos = [0, 0, 17]  # x, y, z
                yaw = 0

        car.set_pos(*desired_car_pos)
        car.set_rot(yaw=yaw)
        car.boost = 0

        state_wrapper.ball.set_pos(x=-2000, y=0,
                                   z=1500) 

        state_wrapper.ball.set_lin_vel(1000, 0, 0)
        state_wrapper.ball.set_ang_vel(0, 0, 0)

class CustomStateSetterExo5(StateSetter):
    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        for car in state_wrapper.cars:
            if car.team_num == 0:
                desired_car_pos = [0, 0, 17]  # x, y, z
                sign = np.random.choice([-1, 1])
                yaw = np.random.uniform(sign*math.pi/4,sign*math.pi*3/4)
            elif car.team_num == 1:
                desired_car_pos = [0, 0, 17]  # x, y, z
                sign = np.random.choice([-1, 1])
                yaw = np.random.uniform(sign*math.pi/4,sign*math.pi*3/4)

        car.set_pos(*desired_car_pos)
        car.set_rot(yaw=yaw)
        car.boost = 0

        state_wrapper.ball.set_pos(x=-2000, y=0,
                                   z=1500)

        state_wrapper.ball.set_lin_vel(1000, 0, 0)
        state_wrapper.ball.set_ang_vel(0, 0, 0)


class CustomStateSetterExo6(StateSetter):
    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        for car in state_wrapper.cars:
            if car.team_num == 0:
                desired_car_pos = [0, 0, 17]  # x, y, z
                yaw = 0
            elif car.team_num == 1:
                desired_car_pos = [0, 0, 17]  # x, y, z
                yaw = 0

        car.set_pos(*desired_car_pos)
        car.set_rot(yaw=yaw)
        car.boost = 0

        state_wrapper.ball.set_pos(x=-3200, y=-4000,
                                   z=400)
        state_wrapper.ball.set_lin_vel(2000, 2000, 2000)
        state_wrapper.ball.set_ang_vel(0, 0,
                                       0)
