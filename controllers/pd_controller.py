import numpy as np
from .controller import Controller


class PDDecentralizedController(Controller):
    def __init__(self, kp, kd):
        self.kp = kp
        self.kd = kd

    def calculate_control(self, q, q_dot, q_d, q_d_dot, q_d_ddot):
        q = np.asarray(q, dtype=float).flatten()
        q_dot = np.asarray(q_dot, dtype=float).flatten()
        q_d = np.asarray(q_d, dtype=float).flatten()
        q_d_dot = np.asarray(q_d_dot, dtype=float).flatten()
        q_d_ddot = np.asarray(q_d_ddot, dtype=float).flatten()

        e = q_d - q
        e_dot = q_d_dot - q_dot

        u = q_d_ddot + self.kd * e_dot + self.kp * e

        return u.flatten()
