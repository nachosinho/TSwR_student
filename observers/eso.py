from copy import copy
import numpy as np


class ESO:
    def __init__(self, A, B, W, L, state, Tp):
        self.A = np.array(A, dtype=float)
        self.B = np.array(B, dtype=float)
        #C=W
        self.W = np.array(W, dtype=float)
        self.L = np.array(L, dtype=float)
        self.state = np.pad(np.array(state, dtype=float),
                            (0, self.A.shape[0] - len(state)))
        self.Tp = Tp
        self.states = []

    def set_B(self, B):
        self.B = np.array(B, dtype=float)

    def update(self, q, u):
        self.states.append(copy(self.state))

        y = np.atleast_1d(q).astype(float)
        u = np.atleast_1d(u).astype(float)

        error = y - self.W @ self.state

        if self.B.ndim == 1:
            Bu = self.B * u.item()
        else:
            Bu = self.B @ u

        if self.L.ndim == 1:
            correction = self.L * error.item()
        else:
            correction = self.L @ error

        state_dot = self.A @ self.state + Bu + correction

        self.state = self.state + self.Tp * state_dot

        return self.state

    def get_state(self):
        return self.state
