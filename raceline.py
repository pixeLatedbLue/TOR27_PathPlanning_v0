import numpy as np
from scipy.optimize import minimize
from scipy.signal import find_peaks

from geometry import as_points, boundary_halfwidths, normals_of_loop, curvature_along


def minimum_curvature_raceline(centerline, left_cones, right_cones,
                               car_width=1.5, safety=0.2, return_info=False):
    if car_width <= 0.0:
        raise ValueError("car_width must be positive.")
    if safety < 0.0:
        raise ValueError("safety must be non-negative.")
    centerline = as_points(centerline, "centerline", min_points=3)
    left_cones = as_points(left_cones, "left_cones", min_points=3)
    right_cones = as_points(right_cones, "right_cones", min_points=3)
    normals = normals_of_loop(centerline)
    half_width, left_clearance, right_clearance = boundary_halfwidths(
        centerline, left_cones, right_cones)
    allowed = np.clip(half_width - (car_width / 2 + safety), 0.0, None)

    def total_bending(shift):
        line = centerline + shift[:, None] * normals
        ahead = np.roll(line, -1, axis=0)
        behind = np.roll(line, 1, axis=0)
        bend = ahead - 2.0 * line + behind
        return np.sum(bend[:, 0] ** 2 + bend[:, 1] ** 2)

    bounds = [(-a, a) for a in allowed]
    result = minimize(total_bending, np.zeros(len(centerline)),
                      method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 500, "ftol": 1e-9})
    shift = result.x
    raceline = centerline + shift[:, None] * normals
    if not return_info:
        return raceline, shift
    info = {
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_iterations": int(result.nit),
        "min_half_width": float(np.min(half_width)) if len(half_width) else 0.0,
        "mean_half_width": float(np.mean(half_width)) if len(half_width) else 0.0,
        "left_clearance": left_clearance,
        "right_clearance": right_clearance,
        "allowed_shift": allowed,
    }
    return raceline, shift, info


def find_apexes(raceline, min_gap=12, height_fraction=0.4):
    if len(raceline) < 3:
        return np.array([], dtype=int)
    curv = curvature_along(raceline, closed=True)
    if curv.max() <= 1e-12:
        return np.array([], dtype=int)
    wrapped = np.concatenate([curv, curv[:min_gap]])
    peaks, _ = find_peaks(wrapped, distance=min_gap,
                          height=height_fraction * curv.max())
    return np.unique(peaks % len(curv))
