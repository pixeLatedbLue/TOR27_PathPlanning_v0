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
                    ACCELERATION, AUTOCROSS, SKIDPAD, EBS_TEST, INSPECTION,
                    TRACKDRIVE)
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

    def test_autocross_explores_then_races_and_finishes(self):
        sim = self._run(AUTOCROSS, 6000)
        self.assertEqual(sim.AS.state, AS_FINISHED)
        self.assertIsNotNone(sim.race)
        self.assertGreaterEqual(sim.lap, 2)

    def test_skidpad_finishes_four_loops(self):
        sim = self._run(SKIDPAD, 6000)
        self.assertEqual(sim.AS.state, AS_FINISHED)
        self.assertEqual(sim.lap, 4)

    def test_inspection_finishes(self):
        sim = self._run(INSPECTION, 1200)
        self.assertEqual(sim.AS.state, AS_FINISHED)

    def test_ebs_test_finishes_and_meets_decel_target(self):
        sim = self._run(EBS_TEST, 3000)
        self.assertEqual(sim.AS.state, AS_FINISHED)
        self.assertTrue(sim.vehicle.is_stopped())
        self.assertGreater(sim.ebs_decel, 10.0)

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
