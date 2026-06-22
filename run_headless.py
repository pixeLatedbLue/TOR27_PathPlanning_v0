import sys

from simulation import Simulation
from autonomous_system import AS_READY, AS_FINISHED
from tracks import ALL_MISSIONS, AUTOCROSS, TRACKDRIVE, SKIDPAD

CONTINUOUS = (AUTOCROSS, TRACKDRIVE)

TIME_BUDGET_S = {
    "acceleration": 60.0,
    "skidpad": 90.0,
    "autocross": 120.0,
    "trackdrive": 400.0,
}
DEFAULT_BUDGET_S = 200.0


def _arm_and_launch(sim):
    sim.set_asms(True)
    for _ in range(2000):
        sim.tick()
        if sim.AS.state == AS_READY and sim.AS.time_in_ready >= 5.0:
            break
    sim.press_go()
    sim.tick()


def run(mission, dt=0.04):
    sim = Simulation(dt=dt)
    sim.select_mission(mission)
    _arm_and_launch(sim)

    max_ticks = int(TIME_BUDGET_S.get(mission, DEFAULT_BUDGET_S) / dt)
    target_laps = sim.track.laps_required
    race_req = stop_req = end_req = False
    finished = False
    for _ in range(max_ticks):
        sim.tick()
        if mission in CONTINUOUS:
            if not race_req:
                sim.request_race()
                race_req = True
            if not stop_req and sim.race is not None and sim.lap >= target_laps:
                sim.request_stop()
                stop_req = True
        elif mission == SKIDPAD and not end_req and sim.lap >= target_laps + 1:
            sim.request_end()
            end_req = True
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
