import numpy as np


class ManipulatorModel:
    def __init__(self, Tp):
        self.Tp = Tp
        self.l1 = 0.5
        self.r1 = 0.01
        self.m1 = 1.
        self.l2 = 0.5
        self.r2 = 0.01
        self.m2 = 1.
        self.I_1 = 1 / 12 * self.m1 * (3 * self.r1 ** 2 + self.l1 ** 2)
        self.I_2 = 1 / 12 * self.m2 * (3 * self.r2 ** 2 + self.l2 ** 2)
        self.m3 = 0.1
        self.r3 = 0.05
        self.I_3 = 2. / 5 * self.m3 * self.r3 ** 2


        #5.1
        # self.m3 = 0.0
        # self.I_3 = 0.0

    def M(self, x):
        q1, q2, q1_dot, q2_dot = x

        d1 = self.l1 / 2
        d2 = self.l2 / 2

        alpha = (self.m1 * d1 ** 2 + self.I_1 + self.m2 * (self.l1 ** 2 + d2 ** 2) + self.I_2 + self.m3 * (self.l1 ** 2 + self.l2 ** 2) + self.I_3)
        beta = self.m2 * self.l1 * d2 + self.m3 * self.l1 * self.l2
        gamma = (self.m2 * d2 ** 2 + self.I_2 + self.m3 * self.l2 ** 2 + self.I_3)
        c2 = np.cos(q2)

        M = np.array([
            [alpha + 2 * beta * c2, gamma + beta * c2],
            [gamma + beta * c2,      gamma]
        ])

        return M

    def C(self, x):
        q1, q2, q1_dot, q2_dot = x

        d2 = self.l2 / 2

        beta = self.m2 * self.l1 * d2 + self.m3 * self.l1 * self.l2
        s2 = np.sin(q2)

        C = np.array([
            [-beta * s2 * q2_dot, -beta * s2 * (q1_dot + q2_dot)],
            [ beta * s2 * q1_dot,  0.0]
        ])

        return C