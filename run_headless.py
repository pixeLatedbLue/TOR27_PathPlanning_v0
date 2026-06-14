"""
Headless runner for the path-planning pipeline -- no UI, no pygame, no display.

This is what you run on a headless machine (CI, a server, or a Jetson in a
Docker container): it arms a mission, presses GO, drives the whole pipeline to
the finish and prints a short telemetry summary. It imports only the simulation
(numpy + scipy), so it needs no graphics stack at all.

    python run_headless.py                # default: autocross
    python run_headless.py skidpad        # one named mission
    python run_headless.py all            # every mission, one after another

Exit code is 0 only if every mission asked for reached "AS Finished" in time,
so it works as a smoke test ("does the whole thing still run end to end?").
"""

import sys

from simulation import Simulation
from autonomous_system import AS_READY, AS_FINISHED
from tracks import ALL_MISSIONS

# how long (in simulated seconds) we let each mission run before giving up.
# trackdrive is ten laps, so it needs the most.
TIME_BUDGET_S = {
    "acceleration": 60.0,
    "skidpad": 90.0,
    "autocross": 120.0,
    "trackdrive": 400.0,
    "ebs_test": 60.0,
    "inspection": 60.0,
    "manual": 400.0,
}
DEFAULT_BUDGET_S = 200.0


def _arm_and_launch(sim):
    """ASMS on, wait out the 5 s AS-Ready hold, then press GO (rules T14.8.4)."""
    sim.set_asms(True)
    for _ in range(2000):
        sim.tick()
        if sim.AS.state == AS_READY and sim.AS.time_in_ready >= 5.0:
            break
    sim.press_go()
    sim.tick()


def run(mission, dt=0.04):
    """Drive one mission to the finish. Returns (finished, snapshot)."""
    sim = Simulation(dt=dt)
    sim.select_mission(mission)
    _arm_and_launch(sim)

    max_ticks = int(TIME_BUDGET_S.get(mission, DEFAULT_BUDGET_S) / dt)
    finished = False
    for _ in range(max_ticks):
        sim.tick()
        if sim.AS.state == AS_FINISHED:
            finished = True
            break

    snap = sim.snapshot()
    cones = snap["map_left"].shape[0] + snap["map_right"].shape[0]
    print(f"mission     : {mission}")
    print(f"finished    : {'yes' if finished else 'NO (ran out of time)'}")
    print(f"final state : {snap['state']}")
    print(f"laps        : {snap['lap']} / {snap['laps_required']}")
    print(f"sim time    : {snap['sim_time']:.1f} s")
    print(f"cones mapped: {cones}")
    if snap["ebs_decel"] > 0.1:
        print(f"EBS decel   : {snap['ebs_decel']:.1f} m/s^2 (rule T15.4.2: > 10)")
    return finished, snap


def main(argv):
    choice = argv[1].lower() if len(argv) > 1 else "autocross"
    if choice == "all":
        missions = list(ALL_MISSIONS)
    elif choice in ALL_MISSIONS:
        missions = [choice]
    else:
        print(f"unknown mission '{choice}'. choose one of: "
              f"{', '.join(ALL_MISSIONS)}, or 'all'.")
        return 2

    all_ok = True
    for i, mission in enumerate(missions):
        if i:
            print("-" * 48)
        ok, _ = run(mission)
        all_ok = all_ok and ok
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
