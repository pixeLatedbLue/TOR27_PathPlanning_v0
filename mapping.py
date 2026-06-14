import numpy as np


class _ConeStore:

    def __init__(self, assoc_radius):
        self.assoc_radius = assoc_radius
        self.pts = []
        self.hits = []

    def add(self, p):
        if self.pts:
            arr = np.asarray(self.pts)
            d = np.linalg.norm(arr - p, axis=1)
            j = int(np.argmin(d))
            if d[j] <= self.assoc_radius:
                n = self.hits[j]
                self.pts[j] = (arr[j] * n + p) / (n + 1)
                self.hits[j] = n + 1
                return False
        self.pts.append(np.asarray(p, float))
        self.hits.append(1)
        return True

    def array(self):
        return np.asarray(self.pts) if self.pts else np.zeros((0, 2))


class ConeMap:
    def __init__(self, start_pose, assoc_radius=1.5):
        self.start = np.array(start_pose[:2], float)
        self.left = _ConeStore(assoc_radius)
        self.right = _ConeStore(assoc_radius)
        self._away = False
        self._closed = False
        self.traveled = 0.0
        self._last_pos = np.array(start_pose[:2], float)

    def update(self, detections, pose):
        px, py, theta = pose
        origin = np.array([px, py])
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s], [s, c]])

        new_count = 0
        for d in detections:
            world = rot @ d.xy + origin
            store = self.left if d.color == "blue" else self.right
            new_count += int(store.add(world))

        self.traveled += float(np.linalg.norm(origin - self._last_pos))
        self._last_pos = origin
        dist_to_start = float(np.linalg.norm(origin - self.start))
        if dist_to_start > 8.0:
            self._away = True
        if self._away and dist_to_start < 5.0 and self.traveled > 20.0:
            self._closed = True
        return new_count

    def left_cones(self):
        return self.left.array()

    def right_cones(self):
        return self.right.array()

    def loop_closed(self):
        return self._closed

    def cone_count(self):
        return len(self.left.pts) + len(self.right.pts)
