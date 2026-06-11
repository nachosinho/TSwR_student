import numpy as np

from observers.eso import ESO
from .adrc_joint_controller import ADRCJointController
from .controller import Controller
from models.manipulator_model import ManipulatorModel


class ADRFLController(Controller):
    def __init__(self, Tp, q0, Kp, Kd, p):
        self.Tp = Tp
        self.model = ManipulatorModel(Tp)

        self.Kp = np.array(Kp, dtype=float)
        self.Kd = np.array(Kd, dtype=float)
        self.p = np.array(p, dtype=float)

        q0 = np.asarray(q0, dtype=float).flatten()

        A = np.zeros((6, 6))
        B = np.zeros((6, 2))
        W = np.array([
            [1., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0.]
        ])

        L = np.array([
            [3 * self.p[0], 0.],
            [0., 3 * self.p[1]],
            [3 * self.p[0] ** 2, 0.],
            [0., 3 * self.p[1] ** 2],
            [self.p[0] ** 3, 0.],
            [0., self.p[1] ** 3]
        ])

        z0 = np.array([
            q0[0],
            q0[1],
            q0[2],
            q0[3],
            0.,
            0.
        ])

        self.eso = ESO(A, B, W, L, z0, Tp)
        self.update_params(q0[:2], q0[2:])

    def update_params(self, q, q_dot):
        x = np.array([
            q[0],
            q[1],
            q_dot[0],
            q_dot[1]
        ])

        M = self.model.M(x)
        C = self.model.C(x)

        M_inv = np.linalg.inv(M)

        A = np.zeros((6, 6))

        A[0:2, 2:4] = np.eye(2)
        A[2:4, 2:4] = -M_inv @ C
        A[2:4, 4:6] = np.eye(2)

        B = np.zeros((6, 2))
        B[2:4, :] = M_inv

        self.eso.A = A
        self.eso.B = B

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        x = np.asarray(x, dtype=float).flatten()

        q = x[:2]
        q_dot = x[2:]

        self.update_params(q, q_dot)

        z_hat = self.eso.get_state()

        q_hat = z_hat[0:2]
        q_dot_hat = z_hat[2:4]
        f_hat = z_hat[4:6]

        q_d = np.asarray(q_d, dtype=float).flatten()
        q_d_dot = np.asarray(q_d_dot, dtype=float).flatten()
        q_d_ddot = np.asarray(q_d_ddot, dtype=float).flatten()

        e = q_d - q_hat
        e_dot = q_d_dot - q_dot_hat

        v = q_d_ddot + self.Kd @ e_dot + self.Kp @ e

        M = self.model.M(x)
        C = self.model.C(x)

        u = M @ (v - f_hat) + C @ q_dot_hat
        # u = M @ v + C @ q_dot_hat

        u = np.asarray(u, dtype=float).flatten()
        u = np.clip(u, -50.0, 50.0)

        self.eso.update(q, u)

        return u