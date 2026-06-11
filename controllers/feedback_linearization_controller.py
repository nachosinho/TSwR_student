import numpy as np
from models.manipulator_model import ManipulatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.Tp = Tp
        self.model = ManipulatorModel(Tp)

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        #5.1-5.7
        # q_dot = np.array([
        #     [x[2]],
        #     [x[3]]
        # ])
        #
        # v = np.array(q_r_ddot).reshape(2, 1)
        #
        # M = self.model.M(x)
        # C = self.model.C(x)
        #
        # tau = M @ v + C @ q_dot
        #
        # return tau.flatten()
        #----------------------------------
        #5.8
        q = np.array([
            [x[0]],
            [x[1]]
        ])

        q_dot = np.array([
            [x[2]],
            [x[3]]
        ])

        q_r = np.array(q_r).reshape(2, 1)
        q_r_dot = np.array(q_r_dot).reshape(2, 1)
        q_r_ddot = np.array(q_r_ddot).reshape(2, 1)

        Kp = np.array([
            [25.0, 0.0],
            [0.0, 25.0]
        ])

        Kd = np.array([
            [10.0, 0.0],
            [0.0, 10.0]
        ])

        e = q_r - q
        e_dot = q_r_dot - q_dot

        v = q_r_ddot + Kd @ e_dot + Kp @ e

        M = self.model.M(x)
        C = self.model.C(x)

        tau = M @ v + C @ q_dot

        return tau.flatten()
