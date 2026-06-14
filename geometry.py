import numpy as np


def as_points(points, name="points", min_points=0):
    arr = np.asarray(points, float)
    if arr.size == 0:
        arr = np.zeros((0, 2), dtype=float)
    if arr.ndim == 1 and arr.size % 2 != 0:
        raise ValueError(f"{name} must be an Nx2 coordinate array.")
    try:
        arr = arr.reshape(-1, 2)
    except ValueError as exc:
        raise ValueError(f"{name} must be an Nx2 coordinate array.") from exc
    if len(arr) < min_points:
        raise ValueError(f"{name} must contain at least {min_points} points.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite coordinates.")
    return arr


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def remove_duplicates(points, tolerance=0.01):
    kept = []
    for p in points:
        if all(np.linalg.norm(p - q) > tolerance for q in kept):
            kept.append(p)
    return np.array(kept) if kept else np.zeros((0, 2))


def normals_of_loop(points):
    ahead = np.roll(points, -1, axis=0)
    behind = np.roll(points, 1, axis=0)
    travel = ahead - behind
    travel = travel / (np.linalg.norm(travel, axis=1, keepdims=True) + 1e-12)
    return np.column_stack([-travel[:, 1], travel[:, 0]])


def smooth_loop(points, window):
    if window < 2 or len(points) < window:
        return points
    n = len(points)
    half = window // 2
    out = np.zeros_like(points)
    for i in range(n):
        idx = [(i + j) % n for j in range(-half, half + 1)]
        out[i] = points[idx].mean(axis=0)
    return out


def resample_loop(points, n_points):
    loop = np.vstack([points, points[0]])
    step = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    distance = np.concatenate([[0.0], np.cumsum(step)])
    if distance[-1] < 1e-9:
        base = points[0] if len(points) else np.zeros(2)
        return np.repeat(np.asarray(base, float)[None, :], n_points, axis=0)
    wanted = np.linspace(0.0, distance[-1], n_points, endpoint=False)
    x = np.interp(wanted, distance, loop[:, 0])
    y = np.interp(wanted, distance, loop[:, 1])
    return np.column_stack([x, y])


def curvature_along(points, closed=False):
    pts = np.asarray(points, float)
    if closed and len(pts) >= 3:
        ahead = np.roll(pts, -1, axis=0)
        behind = np.roll(pts, 1, axis=0)
        d1 = (ahead - behind) / 2.0
        d2 = ahead - 2.0 * pts + behind
        dx, dy = d1[:, 0], d1[:, 1]
        ddx, ddy = d2[:, 0], d2[:, 1]
    else:
        dx, dy = np.gradient(pts[:, 0]), np.gradient(pts[:, 1])
        ddx, ddy = np.gradient(dx), np.gradient(dy)
    return np.abs(dx * ddy - dy * ddx) / (dx * dx + dy * dy + 1e-9) ** 1.5


def _side_distance(point, normal, tangent, cones, fallback, tangent_window):
    if len(cones) == 0:
        return fallback
    rel = cones - point
    signed = rel @ normal
    along = np.abs(rel @ tangent)
    local = signed[(signed > 0.0) & (along <= tangent_window)]
    if len(local):
        return float(np.min(local))
    ahead = signed[signed > 0.0]
    if len(ahead):
        return float(np.min(ahead))
    return fallback


def boundary_halfwidths(centerline, left_cones, right_cones, fallback_width=3.0):
    line = as_points(centerline, "centerline", min_points=3)
    left = as_points(left_cones, "left_cones")
    right = as_points(right_cones, "right_cones")
    normals = normals_of_loop(line)
    tangents = np.column_stack([normals[:, 1], -normals[:, 0]])
    nearest = nearest_cone_halfwidth(line, left, right)

    votes = []
    if len(left) and len(right):
        for i, point in enumerate(line):
            normal = normals[i]
            ldist = np.linalg.norm(left - point, axis=1)
            rdist = np.linalg.norm(right - point, axis=1)
            lnear = left[np.argsort(ldist)[:min(6, len(left))]]
            rnear = right[np.argsort(rdist)[:min(6, len(right))]]
            lproj = (lnear - point) @ normal
            rproj = (rnear - point) @ normal
            votes.append(np.median(lproj) - np.median(rproj))
        if votes and np.median(votes) < 0.0:
            normals = -normals

    widths = []
    left_clearance = []
    right_clearance = []
    for i, point in enumerate(line):
        normal = normals[i]
        fallback = max(float(nearest[i]), fallback_width)
        tangent_window = max(2.0 * fallback, 4.0)
        ldist = _side_distance(point, normal, tangents[i], left, fallback, tangent_window)
        rdist = _side_distance(point, -normal, tangents[i], right, fallback, tangent_window)
        left_clearance.append(ldist)
        right_clearance.append(rdist)
        widths.append(min(ldist, rdist))
    return np.array(widths), np.array(left_clearance), np.array(right_clearance)


def nearest_cone_halfwidth(centerline, left_cones, right_cones):
    centerline = as_points(centerline, "centerline", min_points=1)
    left = as_points(left_cones, "left_cones")
    right = as_points(right_cones, "right_cones")
    all_cones = np.vstack([left, right])
    if len(all_cones) == 0:
        raise ValueError("At least one cone is required to estimate track width.")
    return np.array([np.min(np.linalg.norm(all_cones - p, axis=1))
                     for p in centerline])


def speed_profile_from_curvature(points, mu=1.2, g=9.81, max_speed=18.0,
                                 min_speed=3.0, max_accel=4.0, max_decel=6.0):
    pts = as_points(points, "points", min_points=2)
    curv = curvature_along(pts, closed=True)
    speed = np.sqrt(mu * g / np.maximum(curv, 1e-4))
    speed = np.clip(speed, min_speed, max_speed)

    closed = np.vstack([pts, pts[0]])
    ds = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    n = len(speed)
    for _ in range(3):
        for step_idx in range(n):
            i = (step_idx + 1) % n
            prev = step_idx % n
            speed[i] = min(speed[i], np.sqrt(speed[prev] ** 2 + 2.0 * max_accel * ds[prev]))
        for step_idx in range(n - 1, -1, -1):
            i = step_idx
            nxt = (i + 1) % n
            speed[i] = min(speed[i], np.sqrt(speed[nxt] ** 2 + 2.0 * max_decel * ds[i]))
    return np.clip(speed, min_speed, max_speed)


def path_metrics(points, closed=True):
    pts = np.asarray(points, float)
    if len(pts) < 2:
        return {
            "length": 0.0,
            "max_curvature": 0.0,
            "mean_curvature": 0.0,
            "closure_error": 0.0,
        }
    loop = np.vstack([pts, pts[0]]) if closed else pts
    step = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    curv = curvature_along(pts, closed=closed) if len(pts) >= 3 else np.zeros(len(pts))
    return {
        "length": float(np.sum(step)),
        "max_curvature": float(np.max(curv)) if len(curv) else 0.0,
        "mean_curvature": float(np.mean(curv)) if len(curv) else 0.0,
        "closure_error": float(np.linalg.norm(pts[-1] - pts[0])) if closed else 0.0,
    }


def confidence_from_metrics(metrics, min_width=None, curvature_limit=0.65):
    score = 1.0
    if metrics.get("length", 0.0) <= 1e-6:
        score *= 0.0
    if metrics.get("max_curvature", 0.0) > curvature_limit:
        score *= 0.65
    if min_width is not None:
        if min_width < 1.0:
            score *= 0.35
        elif min_width < 1.8:
            score *= 0.7
    return float(np.clip(score, 0.0, 1.0))
