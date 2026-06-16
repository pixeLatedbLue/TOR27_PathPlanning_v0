import numpy as np

from autonomous_system import (AutonomousSystem, AS_OFF, AS_READY, AS_DRIVING,
                               AS_EMERGENCY, AS_FINISHED,
                               READY_HOLD_S, DRIVING_HOLD_S)
from mapping import ConeMap
from perception import Perception
from planner import Planner
from tracks import (build_track, ACCELERATION, SKIDPAD, AUTOCROSS, TRACKDRIVE,
                    INSPECTION, EBS_TEST, MANUAL)
from vehicle import Vehicle, EBS_DECEL

EXPLORE_SPEED = 5.0
SKIDPAD_SPEED = 7.0
STRAIGHT_SPEED = 16.0


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
        self.planner = Planner(car_width=1.5, safety=0.2, max_speed=14.0,
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
        self._ebs_triggered = False
        self._ebs_v0 = 0.0
        self.ebs_decel = 0.0

    def set_asms(self, on):
        self.AS.set_asms(on)

    def press_go(self):
        self.AS.press_go()

    def emergency(self):
        self.AS.trigger_emergency()
        self.message = "EMERGENCY -- EBS safe stop"

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

        if self.mission in (ACCELERATION, EBS_TEST):
            self._drive_straight(dt)
        elif self.mission == SKIDPAD:
            self._drive_skidpad(dt)
        elif self.mission in (AUTOCROSS, TRACKDRIVE, MANUAL):
            self._drive_loop(dt)
        elif self.mission == INSPECTION:
            self._drive_inspection(dt)

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

        if self.mission == EBS_TEST and self.vehicle.x >= 25.0 and not self._ebs_triggered:
            self._ebs_triggered = True
            self._ebs_v0 = self.vehicle.v
            self.AS.trigger_emergency()
            return

        if self.vehicle.x >= finish_x:
            self.vehicle.step(dt, path, 0.0, brake=False)
            self.message = f"Finish line passed -- braking ({self.vehicle.v:0.1f} m/s)"
            if self.vehicle.is_stopped():
                self.mission_finished = True
        else:
            self.vehicle.step(dt, path, STRAIGHT_SPEED)
            self.message = f"Acceleration run -- {self.vehicle.x:0.0f}/{finish_x:0.0f} m"

    def _drive_skidpad(self, dt):
        self.phase = "skidpad"
        cl = self.track.centerline
        if self.skidpad_loops is None and len(cl) >= 360:
            self.skidpad_loops = [cl[:120], cl[240:360]]

        path, done = self._follow_fixed(cl, dt, SKIDPAD_SPEED)
        req = self.track.laps_required
        self.lap = min(req, 1 + int(self._fixed_idx * req / len(cl)))
        self.message = f"Skidpad -- loop {self.lap}/{req}"
        if done:
            self.vehicle.step(dt, path, 0.0)
            if self.vehicle.is_stopped():
                self.mission_finished = True

    def _drive_loop(self, dt):
        if self.race is None:
            self.phase = "exploration"
            path, plan = self._explore_path()
            self.vehicle.step(dt, path, EXPLORE_SPEED)
            self.lap = 1
            self.message = (f"EXPLORING -- {self.cmap.cone_count()} cones mapped"
                            f"{'  [fallback]' if plan['diagnostics']['fallback'] else ''}")
            if self.cmap.loop_closed():
                self._finish_mapping()
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
        self._prev_angle = None
        self._race_angle = 0.0
        self.message = "MAP COMPLETE -- racing line ready"

    def _race_lap(self, dt):
        raceline = self.race["raceline"]
        speeds = self.race["speed_profile"]
        p = self.vehicle.position
        i0 = int(np.argmin(np.linalg.norm(raceline - p, axis=1)))
        target_speed = float(speeds[i0])
        n = len(raceline)
        window = np.vstack([raceline, raceline])[i0:i0 + max(20, n // 6)]
        self.vehicle.step(dt, window, target_speed)

        self._count_laps(raceline)
        total = self.track.laps_required
        self.message = f"RACING -- lap {self.lap}/{total}  ({target_speed:0.1f} m/s)"
        if self.lap >= total:
            self.vehicle.step(dt, window, 0.0)
            if self.vehicle.is_stopped():
                self.mission_finished = True

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

    def _drive_inspection(self, dt):
        self.phase = "inspection"
        t = self.AS.time_in_driving
        self.vehicle.steer = np.radians(20.0) * np.sin(2.0 * np.pi * 0.3 * t)
        self.vehicle.v = 0.0
        self.message = f"INSPECTION -- spin + sine steer ({t:0.0f}/27s)"
        if t >= 27.0:
            self.mission_finished = True

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
            if self.mission == EBS_TEST and self._ebs_v0 > 0.1:
                self.mission_finished = True
                self.message = (f"EBS TEST PASSED -- stopped from {self._ebs_v0:0.1f} "
                                f"m/s at ~{EBS_DECEL:0.0f} m/s^2 (rule >10)")
            else:
                self.message = "AS EMERGENCY -- safe state (standstill, SDC open)"

    def _follow_fixed(self, line, dt, speed):
        p = self.vehicle.position
        lo = self._fixed_idx
        hi = min(len(line), lo + 40)
        seg = line[lo:hi]
        rel = np.linalg.norm(seg - p, axis=1)
        self._fixed_idx = lo + int(np.argmin(rel))
        window = line[self._fixed_idx:self._fixed_idx + 30]
        done = self._fixed_idx >= len(line) - 3
        if not done:
            self.vehicle.step(dt, window, speed)
        return window, done

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
        }
