import numpy as np
from .adrc_joint_controller import ADRCJointController
from .controller import Controller


class ADRController(Controller):
    def __init__(self, Tp, params, model=None):
        self.joint_controllers = []
        self.model = model
        for param in params:
            self.joint_controllers.append(ADRCJointController(*param, Tp))

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        q_d = np.asarray(q_d).flatten()
        q_d_dot = np.asarray(q_d_dot).flatten()
        q_d_ddot = np.asarray(q_d_ddot).flatten()

        if self.model is not None:
            M = self.model.M(x)
            M_inv = np.linalg.inv(M)

            b1 = M_inv[0, 0]
            b2 = M_inv[1, 1]

            self.joint_controllers[0].set_b(b1)
            self.joint_controllers[1].set_b(b2)

        u = []

        for i, controller in enumerate(self.joint_controllers):
            ui = controller.calculate_control(
                [float(x[i]), float(x[i + 2])],
                float(q_d[i]),
                float(q_d_dot[i]),
                float(q_d_ddot[i])
            )
            u.append(float(np.asarray(ui).squeeze()))

        return np.asarray(u, dtype=float).flatten()
