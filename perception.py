import numpy as np


class Detection:
    __slots__ = ("xy", "color")

    def __init__(self, xy, color):
        self.xy = np.asarray(xy, float)
        self.color = color


class Perception:
    def __init__(self, track, max_range=14.0, fov_deg=180.0,
                 noise_std=0.10, detect_prob=0.92, seed=0):
        if max_range <= 0.0:
            raise ValueError("max_range must be positive.")
        if not 0.0 < fov_deg <= 360.0:
            raise ValueError("fov_deg must be in (0, 360].")
        self.left = np.asarray(track.left, float)
        self.right = np.asarray(track.right, float)
        self.max_range = max_range
        self.half_fov = np.radians(fov_deg) / 2.0
        self.noise_std = noise_std
        self.detect_prob = detect_prob
        self.rng = np.random.default_rng(seed)

    def sense(self, pose):
        px, py, theta = pose
        origin = np.array([px, py])
        c, s = np.cos(-theta), np.sin(-theta)
        rot = np.array([[c, -s], [s, c]])

        out = []
        for cones, color in ((self.left, "blue"), (self.right, "yellow")):
            for cone in cones:
                rel = cone - origin
                dist = np.hypot(rel[0], rel[1])
                if dist > self.max_range:
                    continue
                local = rot @ rel
                bearing = np.arctan2(local[1], local[0])
                if abs(bearing) > self.half_fov:
                    continue
                if self.rng.random() > self.detect_prob:
                    continue
                measured = local + self.noise_std * self.rng.standard_normal(2)
                out.append(Detection(measured, color))
        return out

    @staticmethod
    def split(detections):
        left = [d.xy for d in detections if d.color == "blue"]
        right = [d.xy for d in detections if d.color == "yellow"]
        left = np.array(left) if left else np.zeros((0, 2))
        right = np.array(right) if right else np.zeros((0, 2))
        return left, right
