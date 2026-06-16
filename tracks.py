import numpy as np

ACCELERATION = "acceleration"
SKIDPAD = "skidpad"
AUTOCROSS = "autocross"
TRACKDRIVE = "trackdrive"
INSPECTION = "inspection"
EBS_TEST = "ebs_test"
MANUAL = "manual"

DRIVING_MISSIONS = (ACCELERATION, SKIDPAD, AUTOCROSS, TRACKDRIVE, EBS_TEST)
ALL_MISSIONS = (ACCELERATION, SKIDPAD, AUTOCROSS, TRACKDRIVE,
                INSPECTION, EBS_TEST, MANUAL)


class Track:

    def __init__(self, mission, left, right, start_pose, centerline,
                 closed, laps_required, finish_x=None, length=0.0):
        self.mission = mission
        self.left = np.asarray(left, float)
        self.right = np.asarray(right, float)
        self.start_pose = start_pose
        self.centerline = np.asarray(centerline, float)
        self.closed = closed
        self.laps_required = laps_required
        self.finish_x = finish_x
        self.length = length


def _offset_boundaries(center, width):
    d = np.gradient(center, axis=0)
    tang = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    normal = np.column_stack([-tang[:, 1], tang[:, 0]])
    left = center + (width / 2.0) * normal
    right = center - (width / 2.0) * normal
    return left, right


def _ring(cx, cy, r, k, phase=0.0):
    a = np.linspace(0.0, 2 * np.pi, k, endpoint=False) + phase
    return np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])


def _acceleration(width=3.0, length=75.0, spacing=5.0):
    xs = np.arange(0.0, length + 1e-6, spacing)
    center = np.column_stack([xs, np.zeros_like(xs)])
    left = np.column_stack([xs, np.full_like(xs, +width / 2.0)])
    right = np.column_stack([xs, np.full_like(xs, -width / 2.0)])
    brake_xs = np.arange(length + spacing, length + 30.0, spacing)
    left = np.vstack([left, np.column_stack([brake_xs, np.full_like(brake_xs, +width / 2)])])
    right = np.vstack([right, np.column_stack([brake_xs, np.full_like(brake_xs, -width / 2)])])
    return Track(ACCELERATION, left, right, start_pose=(0.0, 0.0, 0.0),
                 centerline=center, closed=False, laps_required=1,
                 finish_x=length, length=length)


def _skidpad(lane=3.0):
    r_drive = 9.125
    r_in = r_drive - lane / 2.0
    r_out = r_drive + lane / 2.0
    cx = r_drive

    blue = np.vstack([_ring(+cx, 0, r_in, 16), _ring(-cx, 0, r_in, 16)])
    yellow = np.vstack([_ring(+cx, 0, r_out, 20), _ring(-cx, 0, r_out, 20)])

    def arc(cxc, start, turns, k):
        a = np.linspace(start, start + turns * 2 * np.pi, k)
        return np.column_stack([cxc + r_drive * np.cos(a), r_drive * np.sin(a)])

    right_loops = arc(+cx, np.pi, -2.0, 240)
    left_loops = arc(-cx, 0.0, 2.0, 240)
    center = np.vstack([right_loops, left_loops])
    return Track(SKIDPAD, blue, yellow, start_pose=(0.0, 0.0, np.pi / 2),
                 centerline=center, closed=False, laps_required=4,
                 length=2 * np.pi * r_drive)


def _loop_centerline(n=220, seed=7):
    t = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    rx = 32.0 + 7.0 * np.sin(2 * t) + 3.0 * np.cos(3 * t)
    ry = 22.0 + 5.0 * np.cos(2 * t) + 2.5 * np.sin(3 * t)
    return np.column_stack([rx * np.cos(t), ry * np.sin(t)])


def _closed_loop(mission, laps_required, width=3.5, cone_spacing=4.5):
    center = _loop_centerline()
    left_dense, right_dense = _offset_boundaries(center, width)
    left = _resample_closed(left_dense, cone_spacing)
    right = _resample_closed(right_dense, cone_spacing)
    p0 = center[0]
    heading = np.arctan2(center[1, 1] - center[-1, 1], center[1, 0] - center[-1, 0])
    length = _closed_length(center)
    return Track(mission, left, right, start_pose=(p0[0], p0[1], heading),
                 centerline=center, closed=True, laps_required=laps_required,
                 length=length)


def _resample_closed(points, spacing):
    loop = np.vstack([points, points[0]])
    seg = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    total = dist[-1]
    n = max(8, int(round(total / spacing)))
    want = np.linspace(0.0, total, n, endpoint=False)
    x = np.interp(want, dist, loop[:, 0])
    y = np.interp(want, dist, loop[:, 1])
    return np.column_stack([x, y])


def _closed_length(points):
    loop = np.vstack([points, points[0]])
    return float(np.sum(np.linalg.norm(np.diff(loop, axis=0), axis=1)))


def build_track(mission):
    if mission == ACCELERATION:
        return _acceleration()
    if mission == SKIDPAD:
        return _skidpad()
    if mission == AUTOCROSS:
        return _closed_loop(AUTOCROSS, laps_required=2)
    if mission in (TRACKDRIVE, MANUAL):
        return _closed_loop(mission if mission == TRACKDRIVE else AUTOCROSS,
                            laps_required=10)
    if mission == EBS_TEST:
        t = _acceleration(width=3.0, length=60.0)
        t.mission = EBS_TEST
        return t
    if mission == INSPECTION:
        return Track(INSPECTION, np.zeros((0, 2)), np.zeros((0, 2)),
                     start_pose=(0.0, 0.0, 0.0), centerline=np.zeros((0, 2)),
                     closed=False, laps_required=0, length=0.0)
    raise ValueError(f"unknown mission: {mission}")


def make_loop_track(width=4.0, n=46, seed=1):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    R = 25 + 6 * np.sin(3 * t) + 3 * np.cos(2 * t)
    center = np.column_stack([R * np.cos(t), R * np.sin(t)])
    left, right = _offset_boundaries(center, width)
    return center, left, right


def make_skidpad(ri=6.0, ro=9.0, c=13.0):
    left = np.vstack([_ring(c, 0, ri, 18), _ring(-c, 0, ro, 24)])
    right = np.vstack([_ring(c, 0, ro, 24), _ring(-c, 0, ri, 18)])
    return left, right
