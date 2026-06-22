import numpy as np

from autonomous_system import (AutonomousSystem, AS_OFF, AS_READY, AS_DRIVING,
                               AS_EMERGENCY, AS_FINISHED,
                               READY_HOLD_S, DRIVING_HOLD_S)
from mapping import ConeMap
from perception import Perception
from planner import Planner
from tracks import build_track, ACCELERATION, SKIDPAD, AUTOCROSS, TRACKDRIVE
from vehicle import Vehicle, MAX_DECEL

EXPLORE_SPEED = 5.0
SKIDPAD_SPEED = 7.0
SKIDPAD_FAST_SPEED = 8.5
STRAIGHT_SPEED = 16.0

END_AWAY_M = 8.0
END_NEAR_M = 5.0
OBSTACLE_DETECT_DIST = 14.0


def local_to_world(points_local, pose):
    if points_local is None or len(points_local) == 0:
        return None
    px, py, theta = pose
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    return (rot @ np.asarray(points_local, float).T).T + np.array([px, py])


class Simulation:
    def __init__(self, dt=0.04):
        self.dt = dt
        self.AS = AutonomousSystem()
        self.select_mission(AUTOCROSS)

    def select_mission(self, mission):
        self.mission = mission
        self.track = build_track(mission)
        self.perception = Perception(self.track)
        self.cmap = ConeMap(self.track.start_pose)
        self.vehicle = Vehicle(self.track.start_pose, max_speed=STRAIGHT_SPEED)
        self.planner = Planner(car_width=1.5, safety=0.4, max_speed=14.0,
                               min_speed=3.0)

        self.AS.reset()
        self.AS.select_mission(mission)

        self.phase = "armed"
        self.sim_time = 0.0
        self.lap = 0
        self.mission_finished = False
        self.detections = []
        self.local_path_world = None
        self.race = None
        self.skidpad_loops = None
        self.message = "Select ASMS ON, then press GO"
        self._race_angle = 0.0
        self._prev_angle = None
        self._fixed_idx = 0
        self._stop_timer = 0.0
        self.ebs_decel = 0.0
        self._stop_requested = False
        self._end_requested = False
        self._end_far = False
        self._end_stopping = False
        self._race_requested = False
        self._explore_away = False
        self._explore_lap = 1
        self._skid_eight = 1
        self.obstacle = None
        self._obstacle_blocked = False
        self.f1_name = None

    def select_f1(self, key, extent=650.0, width=6.0):
        from f1_tracks import load_f1
        self.select_mission(TRACKDRIVE)
        track = load_f1(key, extent=extent, width=width)
        self.track = track
        self.f1_name = getattr(track, "f1_name", key)
        self.perception = Perception(track)
        self.cmap = ConeMap(track.start_pose)
        self.vehicle = Vehicle(track.start_pose, max_speed=18.0)
        self.planner = Planner(car_width=1.5, safety=0.5, max_speed=16.0,
                               min_speed=3.0, mu=1.0)
        self.race = self.planner.race_from_centerline(
            track.centerline, track.left, track.right)
        rl = self.race["raceline"]
        start = np.array(track.start_pose[:2])
        self._race_start_idx = int(np.argmin(np.linalg.norm(rl - start, axis=1)))
        self._race_seg = np.linalg.norm(np.diff(np.vstack([rl, rl[0]]), axis=0), axis=1)
        self._prev_angle = None
        self._race_angle = 0.0
        self.message = f"{self.f1_name} -- ASMS ON, then GO"

    def set_asms(self, on):
        self.AS.set_asms(on)

    def press_go(self):
        self.AS.press_go()

    def emergency(self):
        self.AS.trigger_emergency()
        self.message = "EMERGENCY -- EBS safe stop"

    def request_stop(self):
        self._stop_requested = True

    def request_end(self):
        self._end_requested = True
        self._end_far = False
        self._end_stopping = False

    def request_race(self):
        self._race_requested = True
        self._explore_away = False

    def set_obstacle(self, world_pos):
        self.obstacle = np.asarray(world_pos, float)

    def clear_obstacle(self):
        self.obstacle = None

    def continue_drive(self):
        if self._obstacle_blocked and not self._obstacle_ahead():
            self._obstacle_blocked = False

    def reset(self):
        self.select_mission(self.mission)

    def tick(self):
        dt = self.dt
        self.sim_time += dt

        self.AS.update(dt, vehicle_stopped=self.vehicle.is_stopped(),
                       mission_finished=self.mission_finished)
        state = self.AS.state

        if state == AS_DRIVING:
            self._drive(dt)
        elif state == AS_EMERGENCY:
            self._emergency_stop(dt)
        elif state == AS_FINISHED:
            self.vehicle.v = 0.0
            self.message = "AS FINISHED -- mission complete, SDC open"
        else:
            self.vehicle.v = 0.0
            if state == AS_READY:
                wait = max(0.0, READY_HOLD_S - self.AS.time_in_ready)
                self.message = f"AS READY -- GO enabled in {wait:0.1f}s" if wait > 0 \
                    else "AS READY -- press GO"

    def _drive(self, dt):
        self._perceive_and_map()

        if not self.AS.motion_allowed:
            self.vehicle.v = 0.0
            hold = max(0.0, DRIVING_HOLD_S - self.AS.time_in_driving)
            self.message = f"AS DRIVING -- launching in {hold:0.1f}s"
            return

        if self._stop_requested:
            self._coast_to_stop(dt)
            return

        if self._obstacle_blocked or self._obstacle_ahead():
            self._obstacle_blocked = True
            self._obstacle_brake(dt)
            return

        end_self_handled = self.mission in (AUTOCROSS, TRACKDRIVE, SKIDPAD)
        if self._end_requested and not end_self_handled:
            self._coast_to_stop(dt)
            return

        if self.mission == ACCELERATION:
            self._drive_straight(dt)
        elif self.mission == SKIDPAD:
            self._drive_skidpad(dt)
        elif self.mission in (AUTOCROSS, TRACKDRIVE):
            self._drive_loop(dt)

    def _coast_to_stop(self, dt):
        self.vehicle.step(dt, self.local_path_world, 0.0)
        self.message = f"STOPPING -- braking ({self.vehicle.v:0.1f} m/s)"
        if self.vehicle.is_stopped():
            self.mission_finished = True

    def _obstacle_ahead(self):
        if self.obstacle is None:
            return False
        rel = self.obstacle - self.vehicle.position
        dist = float(np.linalg.norm(rel))
        if dist > OBSTACLE_DETECT_DIST:
            return False
        if dist < 1e-6:
            return True
        heading = np.array([np.cos(self.vehicle.theta), np.sin(self.vehicle.theta)])
        return float(np.dot(rel, heading) / dist) > 0.3

    def _obstacle_brake(self, dt):
        self.vehicle.step(dt, self.local_path_world, 0.0, brake=True)
        self.message = "OBSTACLE on track -- stopped (clear it, then press CONTINUE)"

    def _perceive_and_map(self):
        self.detections = self.perception.sense(self.vehicle.pose)
        self.cmap.update(self.detections, self.vehicle.pose)

    def _explore_path(self):
        left, right = Perception.split(self.detections)
        plan = self.planner.explore(left, right)
        self.local_path_world = local_to_world(plan["local_path"], self.vehicle.pose)
        return self.local_path_world, plan

    def _straight_path(self):
        left = self.cmap.left_cones()
        right = self.cmap.right_cones()
        x, y, theta = self.vehicle.pose
        fits = []
        for side in (left, right):
            if len(side) >= 2:
                fits.append(np.polyfit(side[:, 0], side[:, 1], 1))
        if fits:
            m = float(np.mean([f[0] for f in fits]))
            c = float(np.mean([f[1] for f in fits]))
            xs = np.linspace(x - 2.0, x + 60.0, 40)
            path = np.column_stack([xs, m * xs + c])
        else:
            path = np.array([[x, y],
                             [x + 60.0 * np.cos(theta), y + 60.0 * np.sin(theta)]])
        self.local_path_world = path
        return path

    def _drive_straight(self, dt):
        path = self._straight_path()
        self.phase = "straight"
        self.lap = self.track.laps_required
        finish_x = self.track.finish_x or 75.0

        if self.vehicle.x >= finish_x:
            self.vehicle.step(dt, path, 0.0, brake=False)
            self.message = f"Finish line passed -- braking ({self.vehicle.v:0.1f} m/s)"
            if self.vehicle.is_stopped():
                self.mission_finished = True
        else:
            self.vehicle.step(dt, path, STRAIGHT_SPEED)
            self.message = f"Acceleration run -- {self.vehicle.x:0.0f}/{finish_x:0.0f} m"

    def _skidpad_window(self, cl):
        n = len(cl)
        lo = self._fixed_idx
        seg = cl[(np.arange(lo, lo + 40) % n)]
        nxt = (lo + int(np.argmin(
            np.linalg.norm(seg - self.vehicle.position, axis=1)))) % n
        wrapped = nxt < lo
        self._fixed_idx = nxt
        return cl[(np.arange(nxt, nxt + 30) % n)], wrapped

    def _drive_skidpad(self, dt):
        self.phase = "skidpad"
        cl = self.track.centerline
        half = len(cl) // 2
        if self.skidpad_loops is None and half >= 1:
            self.skidpad_loops = [cl[:half], cl[half:]]

        if self._end_requested:
            self._skidpad_end(dt, cl)
            return

        window, wrapped = self._skidpad_window(cl)
        if wrapped:
            self._skid_eight += 1
        warmups = self.track.laps_required
        fast = self._skid_eight > warmups
        speed = SKIDPAD_FAST_SPEED if fast else SKIDPAD_SPEED
        self.vehicle.step(dt, window, speed)
        self.lap = self._skid_eight
        if fast:
            self.message = (f"Skidpad -- FAST LAP {self._skid_eight}  "
                            f"({self.vehicle.v:0.1f} m/s)  [STOP / END]")
        else:
            self.message = (f"Skidpad -- warm-up figure-8 "
                            f"{self._skid_eight}/{warmups}  [STOP / END]")

    def _skidpad_end(self, dt, cl):
        window, _ = self._skidpad_window(cl)
        seg = np.linalg.norm(np.diff(np.vstack([cl, cl[0]]), axis=0), axis=1)
        rem = self._arc_forward(seg, self._fixed_idx, 0)
        bd = self._brake_distance()
        if rem > bd + 5.0:
            self._end_far = True
        if self._end_far and rem <= bd:
            self._end_stopping = True
        if self._end_stopping:
            self.vehicle.step(dt, window, 0.0)
            self.message = f"FINAL LAP -- braking to start ({self.vehicle.v:0.1f} m/s)"
            if self.vehicle.is_stopped():
                self.mission_finished = True
        else:
            fast = self._skid_eight > self.track.laps_required
            self.vehicle.step(dt, window, SKIDPAD_FAST_SPEED if fast else SKIDPAD_SPEED)
            self.message = "FINAL LAP -- driving to start"

    def _drive_loop(self, dt):
        if self.race is None:
            self.phase = "exploration"
            path, plan = self._explore_path()
            start = np.array(self.track.start_pose[:2])
            dist = float(np.linalg.norm(self.vehicle.position - start))
            if dist > END_AWAY_M:
                self._explore_away = True

            if self._end_requested:
                if self._explore_away and dist <= self._brake_distance() + 1.0:
                    self._end_stopping = True
                if self._end_stopping:
                    self.vehicle.step(dt, path, 0.0)
                    self.message = f"ENDING -- stopping at start ({self.vehicle.v:0.1f} m/s)"
                    if self.vehicle.is_stopped():
                        self.mission_finished = True
                else:
                    self.vehicle.step(dt, path, EXPLORE_SPEED)
                    self.message = "FINAL LAP -- returning to start (perception)"
                return

            self.vehicle.step(dt, path, EXPLORE_SPEED)
            lap_done = self._explore_away and dist < END_NEAR_M
            if lap_done:
                self._explore_away = False
                self._explore_lap += 1
            if self._race_requested and lap_done:
                self._finish_mapping()
            if self.race is None:
                self.lap = self._explore_lap
                hint = "[RACE armed]" if self._race_requested else "[press RACE]"
                self.message = (f"PERCEPTION -- lap {self.lap}, "
                                f"{self.cmap.cone_count()} cones  {hint}")
        else:
            self.phase = "racing"
            self._race_lap(dt)

    def _finish_mapping(self):
        try:
            self.race = self.planner.finish_mapping(
                self.cmap.left_cones(), self.cmap.right_cones(),
                start_pose=self.track.start_pose)
        except ValueError as exc:
            self.message = f"Mapping failed: {exc}"
            return
        rl = self.race["raceline"]
        start = np.array(self.track.start_pose[:2])
        self._race_start_idx = int(np.argmin(np.linalg.norm(rl - start, axis=1)))
        self._race_seg = np.linalg.norm(np.diff(np.vstack([rl, rl[0]]), axis=0), axis=1)
        self._prev_angle = None
        self._race_angle = 0.0
        self.message = "MAP COMPLETE -- racing line ready"

    @staticmethod
    def _arc_forward(seg, i, j):
        n = len(seg)
        i %= n
        j %= n
        if j == i:
            return 0.0
        if j > i:
            return float(seg[i:j].sum())
        return float(seg[i:].sum() + seg[:j].sum())

    def _brake_distance(self):
        return self.vehicle.v ** 2 / (2.0 * MAX_DECEL) + 1.5

    def _race_lookahead(self, i0):
        curv = self.race["curvature"]
        n = len(curv)
        kappa = float(np.max(curv[np.arange(i0, i0 + 16) % n]))
        return max(2.0, 0.7 / max(kappa, 1e-3))

    def _race_lap(self, dt):
        raceline = self.race["raceline"]
        speeds = self.race["speed_profile"]
        p = self.vehicle.position
        i0 = int(np.argmin(np.linalg.norm(raceline - p, axis=1)))
        target_speed = float(speeds[i0])
        n = len(raceline)
        window = np.vstack([raceline, raceline])[i0:i0 + max(20, n // 6)]
        self.local_path_world = window
        la = self._race_lookahead(i0)

        if self._end_requested:
            rem = self._arc_forward(self._race_seg, i0, self._race_start_idx)
            bd = self._brake_distance()
            if rem > bd + 5.0:
                self._end_far = True
            if self._end_far and rem <= bd:
                self._end_stopping = True
            if self._end_stopping:
                self.vehicle.step(dt, window, 0.0, max_lookahead=la)
                self._count_laps(raceline)
                self.message = f"FINAL LAP -- braking to start ({self.vehicle.v:0.1f} m/s)"
                if self.vehicle.is_stopped():
                    self.mission_finished = True
                return
            self.vehicle.step(dt, window, target_speed, max_lookahead=la)
            self._count_laps(raceline)
            self.message = (f"FINAL LAP {self.lap} -- racing to start "
                            f"({target_speed:0.1f} m/s)")
            return

        self.vehicle.step(dt, window, target_speed, max_lookahead=la)
        self._count_laps(raceline)
        self.message = (f"RACING -- lap {self.lap}  ({target_speed:0.1f} m/s)"
                        f"  [STOP / END]")

    def _count_laps(self, raceline):
        centroid = raceline.mean(axis=0)
        rel = self.vehicle.position - centroid
        ang = np.arctan2(rel[1], rel[0])
        if self._prev_angle is not None:
            d = np.arctan2(np.sin(ang - self._prev_angle),
                           np.cos(ang - self._prev_angle))
            self._race_angle += d
        self._prev_angle = ang
        self.lap = 1 + int(abs(self._race_angle) // (2 * np.pi))

    def _emergency_stop(self, dt):
        if not self.AS.ebs_reacted:
            self.vehicle.step(dt, self.local_path_world, self.vehicle.v)
            self.message = "EBS triggered -- reacting (<200 ms)"
            return
        v_before = self.vehicle.v
        self.vehicle.step(dt, self.local_path_world, 0.0, brake=True)
        if v_before > 0.3:
            self.ebs_decel = max(self.ebs_decel, (v_before - self.vehicle.v) / dt)
        if self.vehicle.is_stopped():
            self.message = "AS EMERGENCY -- safe state (standstill, SDC open)"

    def snapshot(self):
        left, right = Perception.split(self.detections)
        return {
            "mission": self.mission,
            "phase": self.phase,
            "state": self.AS.state,
            "assi": self.AS.assi(),
            "asms": self.AS.asms_on,
            "pose": self.vehicle.pose,
            "speed": self.vehicle.v,
            "steer": self.vehicle.steer,
            "lap": self.lap,
            "laps_required": self.track.laps_required,
            "sim_time": self.sim_time,
            "message": self.message,
            "track": self.track,
            "det_left_local": left,
            "det_right_local": right,
            "map_left": self.cmap.left_cones(),
            "map_right": self.cmap.right_cones(),
            "local_path": self.local_path_world,
            "race": self.race,
            "skidpad_loops": self.skidpad_loops,
            "ebs_decel": self.ebs_decel,
            "ready_t": self.AS.time_in_ready,
            "driving_t": self.AS.time_in_driving,
            "obstacle": self.obstacle,
            "obstacle_blocked": self._obstacle_blocked,
        }
