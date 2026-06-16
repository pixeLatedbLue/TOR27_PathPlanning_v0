import unittest

import numpy as np

from autonomous_system import (AutonomousSystem, AS_OFF, AS_READY, AS_DRIVING,
                               AS_EMERGENCY, AS_FINISHED)
from centerline import global_centerline
from geometry import as_points, speed_profile_from_curvature
from mapping import ConeMap
from perception import Perception, Detection
from planner import Planner
from simulation import Simulation
from tracks import (build_track, make_loop_track, make_skidpad,
                    ACCELERATION, AUTOCROSS, SKIDPAD, TRACKDRIVE)
from vehicle import EBS_DECEL


class PlannerTests(unittest.TestCase):
    def test_finish_mapping_returns_racing_outputs(self):
        center, left, right = make_loop_track()
        planner = Planner(n_points=120)

        result = planner.finish_mapping(
            left, right, start_pose=(center[0, 0], center[0, 1], 0.0))

        self.assertEqual(result["mode"], "racing")
        self.assertEqual(result["raceline"].shape, (120, 2))
        self.assertEqual(result["speed_profile"].shape, (120,))
        self.assertGreater(result["confidence"], 0.0)
        self.assertTrue(np.all(np.isfinite(result["raceline"])))
        self.assertTrue(np.all(result["speed_profile"] >= planner.min_speed))
        self.assertTrue(np.all(result["speed_profile"] <= planner.max_speed))

    def test_finish_mapping_on_narrow_track_does_not_crash(self):
        t = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
        center = np.column_stack([10.0 * np.cos(t), 8.0 * np.sin(t)])
        planner = Planner(n_points=60)
        result = planner.finish_mapping(
            center * 1.15, center * 0.85,
            start_pose=(center[0, 0], center[0, 1], 0.0))
        self.assertEqual(result["raceline"].shape, (60, 2))
        self.assertTrue(np.all(np.isfinite(result["raceline"])))

    def test_exploration_fallback_reports_low_confidence(self):
        planner = Planner()
        result = planner.explore(np.zeros((0, 2)), np.array([[3.0, -2.0]]))

        self.assertEqual(result["local_path"].shape, (2, 2))
        self.assertTrue(result["diagnostics"]["fallback"])
        self.assertLess(result["confidence"], 0.5)

    def test_skidpad_is_split_into_two_loops(self):
        left, right = make_skidpad()
        loops = global_centerline(left, right, max_edge_length=4.0, n_points=120)

        self.assertEqual(len(loops), 2)
        self.assertTrue(all(loop.shape == (120, 2) for loop in loops))

    def test_invalid_global_map_fails_clearly(self):
        planner = Planner()
        with self.assertRaisesRegex(ValueError, "at least 3"):
            planner.finish_mapping(np.zeros((2, 2)), np.zeros((2, 2)))

    def test_constructor_rejects_unsafe_dimensions(self):
        with self.assertRaisesRegex(ValueError, "car_width"):
            Planner(car_width=0.0)
        with self.assertRaisesRegex(ValueError, "safety"):
            Planner(safety=-0.1)

    def test_invalid_coordinates_fail_validation(self):
        with self.assertRaisesRegex(ValueError, "NaN|infinite"):
            as_points([[0.0, 0.0], [np.nan, 1.0]], "bad_points")
        with self.assertRaisesRegex(ValueError, "Nx2"):
            as_points([1.0, 2.0, 3.0], "bad_points")

    def test_speed_profile_respects_loop_boundary_acceleration(self):
        center, _, _ = make_loop_track(n=80)
        speed = speed_profile_from_curvature(
            center, max_speed=15.0, min_speed=3.0, max_accel=4.0, max_decel=6.0)
        closed = np.vstack([center, center[0]])
        ds = np.linalg.norm(np.diff(closed, axis=0), axis=1)

        for i in range(len(speed)):
            nxt = (i + 1) % len(speed)
            self.assertLessEqual(speed[nxt] ** 2 - speed[i] ** 2,
                                 2.0 * 4.0 * ds[i] + 1e-6)
            self.assertLessEqual(speed[i] ** 2 - speed[nxt] ** 2,
                                 2.0 * 6.0 * ds[i] + 1e-6)


class PerceptionMappingTests(unittest.TestCase):
    def test_perception_respects_range_and_fov(self):
        track = build_track(AUTOCROSS)
        per = Perception(track, max_range=14.0, fov_deg=180.0,
                         noise_std=0.0, detect_prob=1.0)
        dets = per.sense(track.start_pose)
        self.assertTrue(len(dets) > 0)
        for d in dets:
            self.assertLessEqual(np.hypot(*d.xy), 14.0 + 1e-6)
            self.assertGreaterEqual(d.xy[0], -1e-6)

    def test_conemap_dedupes_repeated_detections(self):
        cmap = ConeMap(start_pose=(0.0, 0.0, 0.0), assoc_radius=1.0)
        pose = (0.0, 0.0, 0.0)
        for _ in range(5):
            cmap.update([Detection([5.0, 2.0], "blue"),
                         Detection([5.0, -2.0], "yellow")], pose)
        self.assertEqual(len(cmap.left_cones()), 1)
        self.assertEqual(len(cmap.right_cones()), 1)


class AutonomousSystemTests(unittest.TestCase):
    def test_ready_then_drive_respects_five_second_hold(self):
        a = AutonomousSystem()
        a.select_mission("autocross")
        a.set_asms(True)
        a.update(0.1)
        self.assertEqual(a.state, AS_READY)
        a.press_go()
        a.update(0.1)
        self.assertEqual(a.state, AS_READY)
        for _ in range(60):
            a.update(0.1)
        self.assertEqual(a.state, AS_DRIVING)
        self.assertFalse(a.motion_allowed)
        for _ in range(40):
            a.update(0.1)
        self.assertTrue(a.motion_allowed)

    def test_emergency_enters_as_emergency_not_finished(self):
        a = AutonomousSystem()
        a.select_mission("autocross")
        a.set_asms(True)
        a.trigger_emergency()
        a.update(0.1, vehicle_stopped=False, mission_finished=False)
        self.assertEqual(a.state, AS_EMERGENCY)

    def test_off_when_asms_off(self):
        a = AutonomousSystem()
        a.select_mission("autocross")
        a.update(0.1)
        self.assertEqual(a.state, AS_OFF)


class MissionTests(unittest.TestCase):
    def _arm_and_launch(self, sim):
        sim.set_asms(True)
        for _ in range(200):
            sim.tick()
            if sim.AS.state == AS_READY and sim.AS.time_in_ready >= 5.0:
                break
        sim.press_go()
        sim.tick()
        self.assertEqual(sim.AS.state, AS_DRIVING)

    def _run(self, mission, max_ticks):
        sim = Simulation(dt=0.04)
        sim.select_mission(mission)
        self._arm_and_launch(sim)
        for _ in range(max_ticks):
            sim.tick()
            if sim.AS.state == AS_FINISHED:
                return sim
        return sim

    def test_acceleration_finishes(self):
        sim = self._run(ACCELERATION, 2000)
        self.assertEqual(sim.AS.state, AS_FINISHED)
        self.assertGreaterEqual(sim.vehicle.x, sim.track.finish_x)

    def test_acceleration_drives_straight(self):
        sim = Simulation(dt=0.04)
        sim.select_mission(ACCELERATION)
        self._arm_and_launch(sim)
        max_dev = 0.0
        for _ in range(2000):
            sim.tick()
            max_dev = max(max_dev, abs(sim.vehicle.y))
            if sim.AS.state == AS_FINISHED:
                break
        self.assertEqual(sim.AS.state, AS_FINISHED)
        self.assertLess(max_dev, 0.25)

    def test_autocross_stays_in_perception_until_race_requested(self):
        sim = Simulation(dt=0.04)
        sim.select_mission(AUTOCROSS)
        self._arm_and_launch(sim)
        for _ in range(4000):
            sim.tick()
        self.assertIsNone(sim.race)
        self.assertEqual(sim.phase, "exploration")

    def test_autocross_races_after_race_requested_then_stops(self):
        sim = Simulation(dt=0.04)
        sim.select_mission(AUTOCROSS)
        self._arm_and_launch(sim)
        sim.request_race()
        for _ in range(8000):
            sim.tick()
            if sim.race is not None and sim.lap >= 2:
                break
        self.assertIsNotNone(sim.race)
        self.assertGreaterEqual(sim.lap, 2)
        self.assertNotEqual(sim.AS.state, AS_FINISHED)
        sim.request_stop()
        for _ in range(800):
            sim.tick()
            if sim.AS.state == AS_FINISHED:
                break
        self.assertEqual(sim.AS.state, AS_FINISHED)
        self.assertTrue(sim.vehicle.is_stopped())

    def test_end_finishes_lap_and_stops_near_start(self):
        sim = Simulation(dt=0.04)
        sim.select_mission(AUTOCROSS)
        self._arm_and_launch(sim)
        sim.request_race()
        for _ in range(8000):
            sim.tick()
            if sim.race is not None and sim.lap >= 2:
                break
        self.assertIsNotNone(sim.race)
        sim.request_end()
        for _ in range(8000):
            sim.tick()
            if sim.AS.state == AS_FINISHED:
                break
        self.assertEqual(sim.AS.state, AS_FINISHED)
        self.assertTrue(sim.vehicle.is_stopped())
        start = np.array(sim.track.start_pose[:2])
        self.assertLess(np.linalg.norm(sim.vehicle.position - start), 4.0)

    def test_skidpad_end_finishes_lap_and_stops_near_start(self):
        sim = Simulation(dt=0.04)
        sim.select_mission(SKIDPAD)
        self._arm_and_launch(sim)
        start = np.array(sim.track.start_pose[:2])
        for _ in range(2000):
            sim.tick()
            if sim.sim_time > 8 and np.linalg.norm(sim.vehicle.position - start) > 10:
                break
        sim.request_end()
        for _ in range(8000):
            sim.tick()
            if sim.AS.state == AS_FINISHED:
                break
        self.assertEqual(sim.AS.state, AS_FINISHED)
        self.assertTrue(sim.vehicle.is_stopped())
        self.assertLess(np.linalg.norm(sim.vehicle.position - start), 4.0)

    def test_skidpad_runs_three_warmups_then_fast_lap(self):
        sim = Simulation(dt=0.04)
        sim.select_mission(SKIDPAD)
        self._arm_and_launch(sim)
        saw_fast = False
        max_speed = 0.0
        for _ in range(8000):
            sim.tick()
            if "FAST LAP" in sim.message:
                saw_fast = True
            max_speed = max(max_speed, sim.vehicle.v)
            if sim.AS.state == AS_FINISHED:
                break
        self.assertEqual(sim.AS.state, AS_FINISHED)
        self.assertEqual(sim.lap, 4)
        self.assertEqual(len(sim.skidpad_loops), 2)
        self.assertTrue(all(np.all(np.isfinite(loop)) for loop in sim.skidpad_loops))
        self.assertTrue(saw_fast)
        self.assertGreater(max_speed, 7.5)

    def test_skidpad_stays_within_the_lane(self):
        sim = Simulation(dt=0.04)
        sim.select_mission(SKIDPAD)
        self._arm_and_launch(sim)
        R = 9.125
        cR, cL = np.array([R, 0.0]), np.array([-R, 0.0])
        worst = 0.0
        for _ in range(6000):
            sim.tick()
            p = sim.vehicle.position
            off = min(abs(np.linalg.norm(p - cR) - R),
                      abs(np.linalg.norm(p - cL) - R))
            worst = max(worst, off)
            if sim.AS.state == AS_FINISHED:
                break
        self.assertEqual(sim.AS.state, AS_FINISHED)
        self.assertLess(worst, 1.5)

    def test_emergency_mid_drive_stops_the_car_hard(self):
        sim = Simulation(dt=0.04)
        sim.select_mission(AUTOCROSS)
        self._arm_and_launch(sim)
        for _ in range(400):
            sim.tick()
            if sim.vehicle.v > 3.0:
                break
        self.assertGreater(sim.vehicle.v, 3.0)
        sim.emergency()
        for _ in range(400):
            sim.tick()
            if sim.vehicle.is_stopped():
                break
        self.assertEqual(sim.AS.state, AS_EMERGENCY)
        self.assertTrue(sim.vehicle.is_stopped())
        self.assertGreater(sim.ebs_decel, 10.0)
        self.assertLessEqual(sim.ebs_decel, EBS_DECEL + 1e-6)


if __name__ == "__main__":
    unittest.main()
