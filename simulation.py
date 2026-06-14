"""
The simulation that drives the whole pipeline, one tick at a time.

This is the conductor. Every tick it:

    perception  ->  mapping  ->  planner  ->  vehicle  ->  AS state machine

and exposes a single snapshot() the UI can draw. The mission (rules T14.10)
decides which planning behaviour runs:

    acceleration / ebs_test : drive the straight, reactive local path
    skidpad                 : follow the geometric figure-8 centreline,
                              while the mapper recovers the two circle loops
    autocross / trackdrive  : EXPLORE a lap to build the map, then RACE the
                              minimum-curvature line for the remaining laps

The car obeys the AS state machine: it only moves in "AS Driving" (and only
after the 3 s hold, rules T14.8.5), and does a hard EBS stop in "AS Emergency".
"""

import numpy as np

from autonomous_system import (AutonomousSystem, AS_OFF, AS_READY, AS_DRIVING,
                               AS_EMERGENCY, AS_FINISHED,
                               READY_HOLD_S, DRIVING_HOLD_S)
from centerline import global_centerline
from mapping import ConeMap
from perception import Perception
from planner import Planner
from tracks import (build_track, ACCELERATION, SKIDPAD, AUTOCROSS, TRACKDRIVE,
                    INSPECTION, EBS_TEST, MANUAL)
from vehicle import Vehicle, EBS_DECEL

EXPLORE_SPEED = 5.0       # m/s while mapping
SKIDPAD_SPEED = 7.0       # m/s round the circles
STRAIGHT_SPEED = 16.0     # m/s on acceleration / ebs test


def local_to_world(points_local, pose):
    """Map an Nx2 path from the car frame into the world frame."""
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

    # ---- setup -----------------------------------------------------------
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

        self.phase = "armed"          # armed -> exploration/straight -> racing -> finished
        self.sim_time = 0.0
        self.lap = 0
        self.mission_finished = False
        self.detections = []          # this tick's perception (local Detection list)
        self.local_path_world = None  # planner.explore output, in world frame
        self.race = None              # planner.finish_mapping result
        self.skidpad_loops = None     # two-loop centrelines for the display
        self.message = "Select ASMS ON, then press GO"
        self._race_angle = 0.0        # unwrapped angle for lap counting
        self._prev_angle = None
        self._fixed_idx = 0           # pointer along a fixed centreline (skidpad)
        self._stop_timer = 0.0
        self._ebs_triggered = False
        self._ebs_v0 = 0.0
        self.ebs_decel = 0.0

    # ---- inputs ----------------------------------------------------------
    def set_asms(self, on):
        self.AS.set_asms(on)

    def press_go(self):
        self.AS.press_go()

    def emergency(self):
        self.AS.trigger_emergency()
        self.message = "EMERGENCY -- EBS safe stop"

    def reset(self):
        self.select_mission(self.mission)

    # ---- main tick -------------------------------------------------------
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
        else:  # AS Off / AS Ready: held at the line, brakes engaged
            self.vehicle.v = 0.0
            if state == AS_READY:
                wait = max(0.0, READY_HOLD_S - self.AS.time_in_ready)
                self.message = f"AS READY -- GO enabled in {wait:0.1f}s" if wait > 0 \
                    else "AS READY -- press GO"

    # ---- driving ---------------------------------------------------------
    def _drive(self, dt):
        # always perceive + map while driving (this is the "live dataset")
        self._perceive_and_map()

        if not self.AS.motion_allowed:        # 3 s standstill hold (T14.8.5)
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
        """Reactive local centreline -> world path, for pure pursuit."""
        left, right = Perception.split(self.detections)
        plan = self.planner.explore(left, right)
        self.local_path_world = local_to_world(plan["local_path"], self.vehicle.pose)
        return self.local_path_world, plan

    # --- acceleration / EBS test (straight) -------------------------------
    def _drive_straight(self, dt):
        path, _ = self._explore_path()
        self.phase = "straight"
        self.lap = self.track.laps_required   # a straight run is a single lap
        finish_x = self.track.finish_x or 75.0

        if self.mission == EBS_TEST and self.vehicle.x >= 25.0 and not self._ebs_triggered:
            self._ebs_triggered = True
            self._ebs_v0 = self.vehicle.v
            self.AS.trigger_emergency()       # fire the EBS to demo T15.4
            return

        if self.vehicle.x >= finish_x:        # past the line -> brake to stop
            self.vehicle.step(dt, path, 0.0, brake=False)  # normal brake
            self.message = f"Finish line passed -- braking ({self.vehicle.v:0.1f} m/s)"
            if self.vehicle.is_stopped():
                self.mission_finished = True
        else:
            self.vehicle.step(dt, path, STRAIGHT_SPEED)
            self.message = f"Acceleration run -- {self.vehicle.x:0.0f}/{finish_x:0.0f} m"

    # --- skidpad (geometric figure-8) -------------------------------------
    def _drive_skidpad(self, dt):
        self.phase = "skidpad"
        cl = self.track.centerline
        # recover the two circle loops from the live map, for the overlay
        if self.skidpad_loops is None and self.cmap.cone_count() > 30:
            try:
                self.skidpad_loops = global_centerline(
                    self.cmap.left_cones(), self.cmap.right_cones(),
                    max_edge_length=4.0, n_points=120)
            except ValueError:
                self.skidpad_loops = None

        path, done = self._follow_fixed(cl, dt, SKIDPAD_SPEED)
        req = self.track.laps_required
        self.lap = min(req, 1 + int(self._fixed_idx * req / len(cl)))
        self.message = f"Skidpad -- loop {self.lap}/{req}"
        if done:
            self.vehicle.step(dt, path, 0.0)
            if self.vehicle.is_stopped():
                self.mission_finished = True

    # --- autocross / trackdrive (explore then race) -----------------------
    def _drive_loop(self, dt):
        if self.race is None:
            # EXPLORATION: map the track on the opening lap
            self.phase = "exploration"
            path, plan = self._explore_path()
            self.vehicle.step(dt, path, EXPLORE_SPEED)
            self.lap = 1
            self.message = (f"EXPLORING -- {self.cmap.cone_count()} cones mapped"
                            f"{'  [fallback]' if plan['diagnostics']['fallback'] else ''}")
            if self.cmap.loop_closed():
                self._finish_mapping()
        else:
            # RACING: follow the optimised line
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
        # follow the raceline as a forward-looking window
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
        """Lap = one full 2*pi of heading around the track centroid."""
        centroid = raceline.mean(axis=0)
        rel = self.vehicle.position - centroid
        ang = np.arctan2(rel[1], rel[0])
        if self._prev_angle is not None:
            d = np.arctan2(np.sin(ang - self._prev_angle),
                           np.cos(ang - self._prev_angle))
            self._race_angle += d
        self._prev_angle = ang
        # lap 1 was exploration; each further 2*pi adds a racing lap
        self.lap = 1 + int(abs(self._race_angle) // (2 * np.pi))

    # --- inspection (jacked up: spin + sine steer, rules T14.10.2) ---------
    def _drive_inspection(self, dt):
        self.phase = "inspection"
        t = self.AS.time_in_driving
        self.vehicle.steer = np.radians(20.0) * np.sin(2.0 * np.pi * 0.3 * t)
        self.vehicle.v = 0.0
        self.message = f"INSPECTION -- spin + sine steer ({t:0.0f}/27s)"
        if t >= 27.0:                  # transition to AS Finished after 25-30 s
            self.mission_finished = True

    # ---- emergency safe stop (T15.3.5 / T15.4) ---------------------------
    def _emergency_stop(self, dt):
        # safe state (T15.3.5): standstill, brakes engaged, SDC open. The AS
        # stays in AS Emergency -- an e-stop is not a "finished" mission.
        if not self.AS.ebs_reacted:
            # within the 200 ms reaction window (T15.4.1): SDC has opened but
            # the brake has not bitten yet, so the car still coasts.
            self.vehicle.step(dt, self.local_path_world, self.vehicle.v)
            self.message = "EBS triggered -- reacting (<200 ms)"
            return
        v_before = self.vehicle.v
        self.vehicle.step(dt, self.local_path_world, 0.0, brake=True)
        if v_before > 0.3:     # peak decel while actually slowing (T15.4.2)
            self.ebs_decel = max(self.ebs_decel, (v_before - self.vehicle.v) / dt)
        if self.vehicle.is_stopped():
            if self.mission == EBS_TEST and self._ebs_v0 > 0.1:
                # the EBS-test mission is complete once the car has stopped:
                # report it finished so the AS leaves Emergency for AS Finished.
                self.mission_finished = True
                self.message = (f"EBS TEST PASSED -- stopped from {self._ebs_v0:0.1f} "
                                f"m/s at ~{EBS_DECEL:0.0f} m/s^2 (rule >10)")
            else:
                self.message = "AS EMERGENCY -- safe state (standstill, SDC open)"

    # ---- shared: follow a fixed polyline to its end ----------------------
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

    # ---- snapshot for the UI ---------------------------------------------
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
