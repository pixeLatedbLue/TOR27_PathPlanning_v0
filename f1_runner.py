import sys

import numpy as np

from simulation import Simulation
from autonomous_system import AS_READY, AS_FINISHED
from f1_tracks import f1_names


def _arm(sim):
    sim.set_asms(True)
    for _ in range(2000):
        sim.tick()
        if sim.AS.state == AS_READY and sim.AS.time_in_ready >= 5.0:
            break
    sim.press_go()


def run(name, dt=0.04):
    sim = Simulation(dt=dt)
    sim.select_f1(name)
    _arm(sim)
    sim.request_end()
    cones = np.vstack([sim.track.left, sim.track.right])
    start = np.array(sim.track.start_pose[:2])
    minc = np.inf
    vmax = 0.0
    finished = False
    for _ in range(int(500.0 / dt)):
        sim.tick()
        p = sim.vehicle.position
        minc = min(minc, float(np.min(np.linalg.norm(cones - p, axis=1))))
        vmax = max(vmax, sim.vehicle.v)
        if sim.AS.state == AS_FINISHED:
            finished = True
            break
    d = float(np.linalg.norm(sim.vehicle.position - start))
    clip = "  CLIP" if minc < 0.75 else ""
    print(f"{sim.f1_name:24} len={sim.track.length:5.0f} m  min cone clear={minc:.2f} m"
          f"  vmax={vmax:4.1f}  finished={'yes' if finished else 'NO'}  stop {d:.1f} m{clip}")
    return finished, minc


def main(argv):
    arg = argv[1].lower() if len(argv) > 1 else "monza"
    if arg == "list":
        for n in f1_names():
            print(" ", n)
        return 0
    if arg == "all":
        names = f1_names()
    else:
        names = [arg]
    ok = True
    for n in names:
        try:
            fin, _ = run(n)
            ok = ok and fin
        except Exception as exc:
            print(f"{n:24} *** {type(exc).__name__}: {exc}")
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
