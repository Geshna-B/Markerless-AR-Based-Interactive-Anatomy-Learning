import numpy as np
from scipy.spatial.transform import Rotation


class PoseSmoother:
    def __init__(self, alpha=0.3):
        """
        alpha: how much weight to give the NEW pose each frame.
               0.1 = very smooth but slow to respond
               0.5 = responsive but noisier
               0.3 = good balance (default)
        """
        assert 0 < alpha <= 1, "alpha must be between 0 and 1"
        self.alpha = alpha
        self._state = {}  # stores last known R and t per organ

    def smooth(self, organ, R, t):
        t = t.flatten()

        # First time seeing this organ — no previous state to blend with
        if organ not in self._state:
            self._state[organ] = {"R": R.copy(), "t": t.copy()}
            return R, t.reshape(-1, 1)

        prev_R = self._state[organ]["R"]
        prev_t = self._state[organ]["t"]

        # Smooth translation: simple EMA
        t_smooth = self.alpha * t + (1 - self.alpha) * prev_t

        # Smooth rotation: SLERP in quaternion space (avoids gimbal lock)
        q_prev = Rotation.from_matrix(prev_R).as_quat()
        q_new  = Rotation.from_matrix(R).as_quat()

        # Always interpolate along the shortest arc
        if np.dot(q_prev, q_new) < 0:
            q_new = -q_new

        q_smooth = (1 - self.alpha) * q_prev + self.alpha * q_new
        q_smooth /= np.linalg.norm(q_smooth)
        R_smooth = Rotation.from_quat(q_smooth).as_matrix()

        # Save smoothed state for next frame
        self._state[organ] = {"R": R_smooth, "t": t_smooth}

        return R_smooth, t_smooth.reshape(-1, 1)

    def reset(self, organ=None):
        """Call this when tracking is lost so we don't blend stale poses."""
        if organ:
            self._state.pop(organ, None)
        else:
            self._state.clear()