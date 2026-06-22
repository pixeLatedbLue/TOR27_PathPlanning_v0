import numpy as np

from centerline import global_centerline, local_centerline
from raceline import minimum_curvature_raceline, find_apexes
from geometry import (as_points, confidence_from_metrics, curvature_along,
                      path_metrics, resample_loop, speed_profile_from_curvature)


class Planner:
    def __init__(self, car_width=1.5, safety=0.2, n_points=240,
                 max_speed=18.0, min_speed=3.0, mu=1.2):
        if car_width <= 0.0:
            raise ValueError("car_width must be positive.")
        if safety < 0.0:
            raise ValueError("safety must be non-negative.")
        if n_points < 16:
            raise ValueError("n_points must be at least 16 for a stable closed loop.")
        if min_speed <= 0.0 or max_speed <= min_speed:
            raise ValueError("Require 0 < min_speed < max_speed.")
        if mu <= 0.0:
            raise ValueError("mu must be positive.")
        self.car_width = car_width
        self.safety = safety
        self.n_points = n_points
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.mu = mu
        self.mode = "exploration"
        self.result = None
        self._previous_local_path = None

    def explore(self, left_visible, right_visible, look_ahead=15.0, smoothing=0.0,
                pose_delta=None):
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1).")
        path, local_info = local_centerline(
            left_visible, right_visible, look_ahead=look_ahead, return_info=True)
        path = self._smooth_local_path(path, smoothing=smoothing, pose_delta=pose_delta)
        metrics = path_metrics(path, closed=False)
        visible_pairs = min(len(left_visible), len(right_visible))
        confidence = confidence_from_metrics(metrics)
        if visible_pairs < 3:
            confidence *= 0.45
        elif visible_pairs < 5:
            confidence *= 0.75
        if local_info["fallback"]:
            confidence *= 0.5
        return {
            "mode": "exploration",
            "local_path": path,
            "confidence": float(confidence),
            "diagnostics": {
                **metrics,
                **local_info,
                "visible_left": int(len(left_visible)),
                "visible_right": int(len(right_visible)),
                "visible_pairs": int(visible_pairs),
            },
        }

    def _smooth_local_path(self, path, smoothing, pose_delta):
        if smoothing <= 0.0 or pose_delta is None:
            self._previous_local_path = path
            return path
        if self._previous_local_path is None or len(path) < 2:
            self._previous_local_path = path
            return path
        previous = self._transform_previous_local_path(self._previous_local_path, pose_delta)
        previous = self._resample_open(previous, len(path))
        if len(previous) != len(path):
            self._previous_local_path = path
            return path
        blended = (1.0 - smoothing) * path + smoothing * previous
        blended[0] = path[0]
        self._previous_local_path = blended
        return blended

    @staticmethod
    def _transform_previous_local_path(points, pose_delta):
        dx, dy, dtheta = pose_delta
        c, s = np.cos(-dtheta), np.sin(-dtheta)
        rot = np.array([[c, -s], [s, c]])
        return (rot @ (np.asarray(points, float) - np.array([dx, dy])).T).T

    @staticmethod
    def _resample_open(points, n_points):
        pts = np.asarray(points, float)
        if len(pts) < 2 or n_points < 2:
            return pts
        step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        distance = np.concatenate([[0.0], np.cumsum(step)])
        if distance[-1] < 1e-9:
            return pts
        wanted = np.linspace(0.0, distance[-1], n_points)
        x = np.interp(wanted, distance, pts[:, 0])
        y = np.interp(wanted, distance, pts[:, 1])
        return np.column_stack([x, y])

    def finish_mapping(self, left_cones, right_cones, start_pose=None):
        loops = global_centerline(left_cones, right_cones,
                                  start_pose=start_pose, n_points=self.n_points)
        main_loop = loops[0]
        raceline, shift, race_info = minimum_curvature_raceline(
            main_loop, left_cones, right_cones,
            car_width=self.car_width, safety=self.safety, return_info=True)
        curvature = curvature_along(raceline, closed=True)
        speed_profile = speed_profile_from_curvature(
            raceline, mu=self.mu, max_speed=self.max_speed, min_speed=self.min_speed)
        metrics = path_metrics(raceline, closed=True)
        confidence = confidence_from_metrics(metrics, min_width=race_info["min_half_width"])
        if not race_info["optimizer_success"]:
            confidence *= 0.65
        self.result = {
            "mode": "racing",
            "centerline": main_loop,
            "raceline": raceline,
            "apex_indices": find_apexes(raceline),
            "curvature": curvature,
            "speed_profile": speed_profile,
            "confidence": float(confidence),
            "diagnostics": {
                **metrics,
                "loops": int(len(loops)),
                "min_half_width": race_info["min_half_width"],
                "mean_half_width": race_info["mean_half_width"],
                "optimizer_success": race_info["optimizer_success"],
                "optimizer_message": race_info["optimizer_message"],
                "optimizer_iterations": race_info["optimizer_iterations"],
            },
            "all_loops": loops,
        }
        self.mode = "racing"
        return self.result

    def race_from_centerline(self, centerline, left_cones, right_cones):
        main_loop = resample_loop(as_points(centerline, "centerline", min_points=3),
                                  self.n_points)
        raceline, shift, race_info = minimum_curvature_raceline(
            main_loop, left_cones, right_cones,
            car_width=self.car_width, safety=self.safety, return_info=True)
        curvature = curvature_along(raceline, closed=True)
        speed_profile = speed_profile_from_curvature(
            raceline, mu=self.mu, max_speed=self.max_speed, min_speed=self.min_speed)
        metrics = path_metrics(raceline, closed=True)
        confidence = confidence_from_metrics(metrics, min_width=race_info["min_half_width"])
        self.result = {
            "mode": "racing",
            "centerline": main_loop,
            "raceline": raceline,
            "apex_indices": find_apexes(raceline),
            "curvature": curvature,
            "speed_profile": speed_profile,
            "confidence": float(confidence),
            "diagnostics": {
                **metrics,
                "loops": 1,
                "min_half_width": race_info["min_half_width"],
                "mean_half_width": race_info["mean_half_width"],
                "optimizer_success": race_info["optimizer_success"],
                "optimizer_message": race_info["optimizer_message"],
                "optimizer_iterations": race_info["optimizer_iterations"],
            },
            "all_loops": [main_loop],
        }
        self.mode = "racing"
        return self.result

    def race(self):
        if self.result is None:
            raise RuntimeError("finish_mapping() has not been called yet.")
        return self.result

    def reset_exploration(self):
        self._previous_local_path = None
        if self.result is None:
            self.mode = "exploration"
