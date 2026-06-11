import numpy as np
from .controller import Controller
from models.manipulator_model import ManipulatorModel


class MMAController(Controller):
    def __init__(self, Tp):
        self.Tp = Tp
        model_1 = ManipulatorModel(Tp)
        model_1.m3 = 0.1
        model_1.r3 = 0.05
        model_1.I_3 = 2. / 5 * model_1.m3 * model_1.r3 ** 2

        model_2 = ManipulatorModel(Tp)
        model_2.m3 = 0.01
        model_2.r3 = 0.01
        model_2.I_3 = 2. / 5 * model_2.m3 * model_2.r3 ** 2

        model_3 = ManipulatorModel(Tp)
        model_3.m3 = 1.0
        model_3.r3 = 0.3
        model_3.I_3 = 2. / 5 * model_3.m3 * model_3.r3 ** 2

        self.models = [model_1, model_2, model_3]
        self.i = 0
        self.x_models = None
        self.last_u = np.zeros((2, 1))

    def choose_model(self, x):
        x = np.array(x)

        if self.x_models is None:
            self.x_models = [x.copy(), x.copy(), x.copy()]
            self.i = 0
            return

        errors = []

        for x_model in self.x_models:
            error = np.linalg.norm(x - x_model)
            errors.append(error)

        self.i = int(np.argmin(errors))

        for j, model in enumerate(self.models):
            x_model = self.x_models[j]

            q_dot = x_model[2:].reshape(2, 1)

            M = model.M(x_model)
            C = model.C(x_model)

            q_ddot = np.linalg.inv(M) @ (self.last_u - C @ q_dot)

            x_dot = np.array([
                x_model[2],
                x_model[3],
                q_ddot[0, 0],
                q_ddot[1, 0]
            ])
            # przewidywany stan modelu metodą Eulera:
            self.x_models[j] = x_model + self.Tp * x_dot

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)

        q = np.array(x[:2]).reshape(2, 1)
        q_dot = np.array(x[2:]).reshape(2, 1)

        q_r = np.array(q_r).reshape(2, 1)
        q_r_dot = np.array(q_r_dot).reshape(2, 1)
        q_r_ddot = np.array(q_r_ddot).reshape(2, 1)

        Kp = np.diag([25.0, 25.0])
        Kd = np.diag([10.0, 10.0])

        e = q_r - q
        e_dot = q_r_dot - q_dot

        v = q_r_ddot + Kd @ e_dot + Kp @ e

        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)

        u = M @ v + C @ q_dot

        self.last_u = u

        return u.flatten()