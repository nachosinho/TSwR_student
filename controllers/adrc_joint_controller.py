import numpy as np
from observers.eso import ESO
from .controller import Controller
from .pd_controller import PDDecentralizedController

class ADRCJointController(Controller):
    def __init__(self, b, kp, kd, p, q0, Tp):
        self.b = b
        self.kp = kp
        self.kd = kd
        self.Tp = Tp
        self.pd = PDDecentralizedController(kp, kd)

        A = np.array([
            [0., 1., 0.],
            [0., 0., 1.],
            [0., 0., 0.]
        ])

        B = np.array([0., b, 0.])

        W = np.array([[1., 0., 0.]])

        L = np.array([
            3. * p,
            3. * p ** 2,
            p ** 3
        ])

        self.eso = ESO(A, B, W, L, q0, Tp)

    def set_b(self, b):
        self.b = b
        B = np.array([0., b, 0.])
        self.eso.set_B(B)

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        z_hat = self.eso.get_state()

        q = float(x[0])
        q_dot_hat = float(z_hat[1])
        f_hat = float(z_hat[2])

        v = self.pd.calculate_control(
            q=np.array([q]),
            q_dot=np.array([q_dot_hat]),
            q_d=np.array([q_d]),
            q_d_dot=np.array([q_d_dot]),
            q_d_ddot=np.array([q_d_ddot])
        )[0]

        u = (v + f_hat) / self.b
        # 5
        # u = v / self.b

        u = np.clip(u, -50.0, 50.0)

        self.eso.update(q, u)

        return float(np.asarray(u).squeeze())
