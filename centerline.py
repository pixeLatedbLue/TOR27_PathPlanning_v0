import numpy as np
from scipy.spatial import Delaunay, QhullError

from geometry import as_points, remove_duplicates, smooth_loop, resample_loop, unit


def _find_gates(all_cones, side, max_edge_length):
    try:
        tri = Delaunay(all_cones)
    except QhullError as exc:
        raise ValueError("Cone map is degenerate; Delaunay triangulation failed.") from exc
    gate_mid = {}
    triangle_gates = []
    for triangle in tri.simplices:
        here = []
        for k in range(3):
            a, b = int(triangle[k]), int(triangle[(k + 1) % 3])
            if side[a] != side[b]:
                if np.linalg.norm(all_cones[a] - all_cones[b]) <= max_edge_length:
                    key = (a, b) if a < b else (b, a)
                    gate_mid.setdefault(key, (all_cones[a] + all_cones[b]) / 2)
                    here.append(key)
        if len(here) >= 2:
            triangle_gates.append(here)
    return gate_mid, triangle_gates


def global_centerline(left_cones, right_cones, start_pose=None,
                      max_edge_length=7.0, smooth_window=5, n_points=300,
                      min_loop_gates=8):
    if max_edge_length <= 0.0:
        raise ValueError("max_edge_length must be positive.")
    if n_points < 3:
        raise ValueError("n_points must be at least 3.")
    left = remove_duplicates(as_points(left_cones, "left_cones", min_points=3))
    right = remove_duplicates(as_points(right_cones, "right_cones", min_points=3))
    if len(left) < 3 or len(right) < 3:
        raise ValueError("Each cone boundary must contain at least 3 unique cones.")
    all_cones = np.vstack([left, right])
    side = np.array([0] * len(left) + [1] * len(right))

    gate_mid, triangle_gates = _find_gates(all_cones, side, max_edge_length)
    if len(gate_mid) < 3:
        raise ValueError("Not enough gates - check cones or max_edge_length.")

    adjacency = {k: set() for k in gate_mid}
    for here in triangle_gates:
        for i in range(len(here)):
            for j in range(i + 1, len(here)):
                adjacency[here[i]].add(here[j])
                adjacency[here[j]].add(here[i])

    orders = _extract_loops(gate_mid, adjacency, start_pose, min_loop_gates)
    if not orders:
        raise ValueError("No valid centerline loop found; check cone density and max_edge_length.")

    loops = []
    for order in orders:
        line = np.array([gate_mid[k] for k in order])
        line = smooth_loop(line, smooth_window)
        loops.append(resample_loop(line, n_points))
    loops.sort(key=_loop_length, reverse=True)
    return loops


def _loop_length(loop):
    closed = np.vstack([loop, loop[0]])
    return float(np.sum(np.linalg.norm(np.diff(closed, axis=0), axis=1)))


def _choose_next(pos, current, candidates, heading=None, preferred_axis=None):
    best = None
    best_score = -np.inf
    for k in candidates:
        step = pos[k] - pos[current]
        dist = np.linalg.norm(step)
        if dist < 1e-9:
            continue
        direction = step / dist
        heading_score = 0.0 if heading is None else float(np.dot(direction, heading))
        progress_score = 0.0
        if preferred_axis is not None:
            progress_score = float(np.dot(direction, preferred_axis))
        score = 2.5 * heading_score + 1.0 * progress_score - 0.08 * dist
        if score > best_score:
            best_score = score
            best = k
    if best is None:
        return min(candidates, key=lambda k: np.linalg.norm(pos[k] - pos[current]))
    return best


def _extract_loops(gate_mid, adjacency, start_pose, min_loop_gates):
    keys = list(gate_mid)
    pos = gate_mid
    visited = set()
    loops = []
    first = True

    while len(visited) < len(keys):
        unvisited = [k for k in keys if k not in visited]
        if first and start_pose is not None:
            car = np.array(start_pose[:2])
            start = min(unvisited, key=lambda k: np.linalg.norm(pos[k] - car))
            heading = np.array([np.cos(start_pose[2]), np.sin(start_pose[2])])
        else:
            start = min(unvisited, key=lambda k: (pos[k][0], pos[k][1]))
            heading = None
        first = False

        order = [start]
        visited.add(start)
        current = start
        while True:
            candidates = [k for k in adjacency[current] if k not in visited]
            if not candidates:
                break
            nxt = _choose_next(pos, current, candidates, heading=heading)
            heading = unit(pos[nxt] - pos[current])
            order.append(nxt)
            visited.add(nxt)
            current = nxt

        if len(order) >= min_loop_gates:
            loops.append(order)
    return loops


def local_centerline(left_visible, right_visible,
                     max_edge_length=7.0, look_ahead=15.0, return_info=False):
    if max_edge_length <= 0.0:
        raise ValueError("max_edge_length must be positive.")
    if look_ahead <= 0.0:
        raise ValueError("look_ahead must be positive.")
    left = remove_duplicates(as_points(left_visible, "left_visible"))
    right = remove_duplicates(as_points(right_visible, "right_visible"))
    info = {
        "fallback": False,
        "reason": "",
        "gate_count": 0,
    }

    if len(left) == 0 or len(right) == 0:
        info.update({"fallback": True, "reason": "missing one cone boundary"})
        path = np.array([[0.0, 0.0], [look_ahead, 0.0]])
        return (path, info) if return_info else path

    all_cones = np.vstack([left, right])
    side = np.array([0] * len(left) + [1] * len(right))

    midpoints = [np.array([0.0, 0.0])]
    if len(all_cones) >= 3:
        try:
            tri = Delaunay(all_cones)
            for triangle in tri.simplices:
                for k in range(3):
                    a, b = int(triangle[k]), int(triangle[(k + 1) % 3])
                    if side[a] != side[b] and \
                       np.linalg.norm(all_cones[a] - all_cones[b]) <= max_edge_length:
                        midpoints.append((all_cones[a] + all_cones[b]) / 2)
        except QhullError:
            info.update({"fallback": True, "reason": "degenerate visible cone geometry"})
    midpoints = remove_duplicates(np.array(midpoints))
    info["gate_count"] = int(max(0, len(midpoints) - 1))
    if len(midpoints) < 2:
        info.update({"fallback": True, "reason": info["reason"] or "no valid gates"})
        path = np.array([[0.0, 0.0], [look_ahead, 0.0]])
        return (path, info) if return_info else path

    order = [0]
    seen = {0}
    heading = np.array([1.0, 0.0])
    preferred_axis = np.array([1.0, 0.0])
    while len(seen) < len(midpoints):
        remaining = [i for i in range(len(midpoints)) if i not in seen]
        forward = [i for i in remaining if midpoints[i, 0] > midpoints[order[-1], 0] - 0.5]
        candidates = forward if forward else remaining
        pos = {i: midpoints[i] for i in range(len(midpoints))}
        nxt = _choose_next(pos, order[-1], candidates,
                           heading=heading, preferred_axis=preferred_axis)
        step = midpoints[nxt] - midpoints[order[-1]]
        if np.linalg.norm(step) > max_edge_length * 1.5 and len(order) > 1:
            break
        heading = unit(step)
        order.append(nxt)
        seen.add(nxt)

    path = midpoints[order]
    path = path[path[:, 0] > -0.5]
    path = path[path[:, 0] <= look_ahead + 2.0]
    if len(path) < 2:
        info.update({"fallback": True, "reason": "no forward gates"})
        path = np.array([[0.0, 0.0], [look_ahead, 0.0]])
        return (path, info) if return_info else path
    return (path, info) if return_info else path
