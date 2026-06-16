import numpy as np

EBS_DECEL = 11.0
MAX_ACCEL = 4.0
MAX_DECEL = 6.0
MAX_STEER = np.radians(28.0)

LOOKAHEAD_GAIN = 0.4
LOOKAHEAD_MIN = 2.0
LOOKAHEAD_MAX = 3.5


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

    def _pure_pursuit(self, path_world, lookahead):
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
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
        ld = max(np.linalg.norm(to_target), 1e-3)
        return np.arctan2(2.0 * self.wheelbase * np.sin(alpha), ld)

    def step(self, dt, path_world, target_speed, brake=False, lookahead=4.0):
        if brake:
            target_speed = 0.0

        if target_speed >= self.v:
            self.v = min(target_speed, self.v + MAX_ACCEL * dt)
        else:
            decel = EBS_DECEL if brake else MAX_DECEL
            self.v = max(target_speed, self.v - decel * dt)
        self.v = float(np.clip(self.v, 0.0, self.max_speed))

        if not brake and self.v > 1e-3:
            ld = float(np.clip(lookahead + LOOKAHEAD_GAIN * self.v,
                               LOOKAHEAD_MIN, LOOKAHEAD_MAX))
            self.steer = float(np.clip(self._pure_pursuit(path_world, ld),
                                       -MAX_STEER, MAX_STEER))
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt
        self.theta += self.v / self.wheelbase * np.tan(self.steer) * dt
        self.theta = float(np.arctan2(np.sin(self.theta), np.cos(self.theta)))
