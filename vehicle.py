"""
A simple car model + path follower.

Just enough vehicle to make the pipeline visible: a kinematic bicycle model
(the standard teaching model) steered by pure pursuit, with a rate-limited
throttle/brake so speed changes look physical. The emergency brake follows the
rules' deceleration target (T15.4.2: average deceleration > 10 m/s^2).

It is NOT a dynamic tyre model -- no slip, no load transfer -- which is why the
skidpad "fast line" is left to the geometric centreline (see raceline.py).
"""

import numpy as np

EBS_DECEL = 11.0       # m/s^2, emergency brake (rules T15.4.2 require > 10)
MAX_ACCEL = 4.0        # m/s^2, normal acceleration
MAX_DECEL = 6.0        # m/s^2, normal braking
MAX_STEER = np.radians(28.0)


class Vehicle:
    def __init__(self, pose, wheelbase=1.55, max_speed=18.0):
        self.x, self.y, self.theta = pose
        self.v = 0.0
        self.steer = 0.0
        self.wheelbase = wheelbase
        self.max_speed = max_speed

    @property
    def pose(self):
        return (self.x, self.y, self.theta)

    @property
    def position(self):
        return np.array([self.x, self.y])

    def is_stopped(self, eps=0.15):
        return self.v < eps

    # ---- control ----------------------------------------------------------
    def _pure_pursuit(self, path_world, lookahead):
        """Steering angle toward a point ~lookahead metres along the path."""
        if path_world is None or len(path_world) < 2:
            return 0.0
        p = self.position
        d = np.linalg.norm(path_world - p, axis=1)
        i0 = int(np.argmin(d))
        target = path_world[-1]
        for i in range(i0, len(path_world)):
            if np.linalg.norm(path_world[i] - p) >= lookahead:
                target = path_world[i]
                break
        to_target = target - p
        alpha = np.arctan2(to_target[1], to_target[0]) - self.theta
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))   # wrap to [-pi, pi]
        ld = max(np.linalg.norm(to_target), 1e-3)
        return np.arctan2(2.0 * self.wheelbase * np.sin(alpha), ld)

    def step(self, dt, path_world, target_speed, brake=False, lookahead=4.0):
        """Advance one tick following `path_world` toward `target_speed`."""
        if brake:
            target_speed = 0.0

        # longitudinal: rate-limited approach to the target speed
        if target_speed >= self.v:
            self.v = min(target_speed, self.v + MAX_ACCEL * dt)
        else:
            decel = EBS_DECEL if brake else MAX_DECEL
            self.v = max(target_speed, self.v - decel * dt)
        self.v = float(np.clip(self.v, 0.0, self.max_speed))

        # lateral: pure pursuit (scale the lookahead a little with speed)
        if not brake and self.v > 1e-3:
            ld = float(np.clip(lookahead + 0.4 * self.v, 2.5, 10.0))
            self.steer = float(np.clip(self._pure_pursuit(path_world, ld),
                                       -MAX_STEER, MAX_STEER))
        # bicycle-model kinematics
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt
        self.theta += self.v / self.wheelbase * np.tan(self.steer) * dt
        self.theta = float(np.arctan2(np.sin(self.theta), np.cos(self.theta)))
