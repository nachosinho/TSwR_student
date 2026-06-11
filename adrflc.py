import matplotlib.pyplot as plt
import numpy as np
from controllers.adrc_flc_controller import ADRFLController
from trajectory_generators.constant_torque import ConstantTorque
from trajectory_generators.sinusonidal import Sinusoidal
from trajectory_generators.poly3 import Poly3
from utils.simulation import simulate

Tp = 0.001
end = 5

# traj_gen = ConstantTorque(np.array([0., 1.0])[:, np.newaxis])
traj_gen = Sinusoidal(np.array([0., 1.]), np.array([2., 2.]), np.array([0., 0.]))
# traj_gen = Poly3(np.array([0., 0.]), np.array([pi/4, pi/6]), end)

kp_est_1 = 80.0
kp_est_2 = 100.0

kd_est_1 = 24.0
kd_est_2 = 24.0

p1 = 50.0
p2 = 20.0


q0, qdot0, _ = traj_gen.generate(0.)
q1_0 = np.array([q0[0], qdot0[0]])
q2_0 = np.array([q0[1], qdot0[1]])

Kp = np.diag([kp_est_1, kp_est_2])
Kd = np.diag([kd_est_1, kd_est_2])
p = np.array([p1, p2])

controller = ADRFLController(Tp, np.concatenate([q0, qdot0]), Kp, Kd, p)


Q, Q_d, u, T = simulate("PYBULLET", traj_gen, controller, Tp, end)

eso = np.array(controller.eso.states)

plt.figure("ADRC + FLC - estymacja ESO", figsize=(10, 7))

plt.subplot(221)
plt.plot(T, eso[:, 0], label="estymowane q1")
plt.plot(T, Q[:, 0], "r", label="rzeczywiste q1")
plt.title("ESO: pozycja stawu 1")
plt.xlabel("czas [s]")
plt.ylabel("q1 [rad]")
plt.legend()
plt.grid(True)

plt.subplot(222)
plt.plot(T, eso[:, 2], label="estymowane q1_dot")
plt.plot(T, Q[:, 2], "r", label="rzeczywiste q1_dot")
plt.title("ESO: prędkość stawu 1")
plt.xlabel("czas [s]")
plt.ylabel("q1_dot [rad/s]")
plt.legend()
plt.grid(True)

plt.subplot(223)
plt.plot(T, eso[:, 1], label="estymowane q2")
plt.plot(T, Q[:, 1], "r", label="rzeczywiste q2")
plt.title("ESO: pozycja stawu 2")
plt.xlabel("czas [s]")
plt.ylabel("q2 [rad]")
plt.legend()
plt.grid(True)

plt.subplot(224)
plt.plot(T, eso[:, 3], label="estymowane q2_dot")
plt.plot(T, Q[:, 3], "r", label="rzeczywiste q2_dot")
plt.title("ESO: prędkość stawu 2")
plt.xlabel("czas [s]")
plt.ylabel("q2_dot [rad/s]")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

plt.figure("ADRC + FLC - śledzenie trajektorii i sterowanie", figsize=(10, 7))

plt.subplot(221)
plt.plot(T, Q[:, 0], "r", label="rzeczywiste q1")
plt.plot(T, Q_d[:, 0], "b", label="zadane q1")
plt.title("Śledzenie trajektorii: staw 1")
plt.xlabel("czas [s]")
plt.ylabel("q1 [rad]")
plt.legend()
plt.grid(True)

plt.subplot(222)
plt.plot(T, Q[:, 1], "r", label="rzeczywiste q2")
plt.plot(T, Q_d[:, 1], "b", label="zadane q2")
plt.title("Śledzenie trajektorii: staw 2")
plt.xlabel("czas [s]")
plt.ylabel("q2 [rad]")
plt.legend()
plt.grid(True)

plt.subplot(223)
plt.plot(T, u[:, 0], "r", label="moment u1")
plt.plot(T, u[:, 1], "b", label="moment u2")
plt.title("Sygnały sterujące ADRC + FLC")
plt.xlabel("czas [s]")
plt.ylabel("moment [Nm]")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()