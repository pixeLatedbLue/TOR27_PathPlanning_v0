# Path Planning for a Formula Student Driverless Car

This is a path planning pipeline for a Formula Student Driverless (FSD) car,
with a small live viewer so you can watch it work. The car gets cones from a
(fake) sensor, builds a map of the track, plans a path through it, and drives
that path while following the FSD rules. It is built for the four Formula
Student driving missions: **acceleration, skidpad, autocross and trackdrive**.

## Status: not tested yet

Please read this first. This project has **not been properly tested yet and is
still to be tested**. There are a few unit tests in the repo, but the whole
thing has not been checked on the real Jetson board or in an actual run, so
treat everything here as work in progress. Numbers, behaviour and timings may
still change once it runs on hardware.

## What it does

The idea is simple. The track is marked by cones (blue on the left, yellow on
the right). The car cannot see the whole track at once, it only sees the cones
near it. So it drives slowly first in perception mode to map the track. Once you
switch it to race mode, it plans a faster line and races on it.

It handles all the FSD missions:

| Mission | What happens |
|---------|--------------|
| **Acceleration** | Drive straight down a 75 m lane, then brake to a stop. |
| **Skidpad** | Drive the figure of eight (right circle, then left): three warm-up laps, then fast laps that keep going until you end it. |
| **Autocross** | Map the track in perception mode, then (on the RACE button) race the planned line. |
| **Trackdrive** | Map in perception mode, then (on the RACE button) race lap after lap until you end it. |

It also models the safety side of FSD, not just the driving:

- **ASMS** is the master switch that arms the autonomous system.
- **RES** is the remote the marshal holds. It has the **Go** button (to start)
  and the emergency stop button.
- **R2D** (ready to drive) is only allowed 5 seconds after the car is Ready, and
  only when Go is pressed. This is a rule.
- **EBS** is the emergency brake. When the emergency stop is hit, the car opens
  the safety circuit and brakes hard (the rule says it has to slow down faster
  than 10 m/s squared).
- **AS states** are the states the car moves through: Off, Ready, Driving,
  Emergency and Finished. There is also the **ASSI** light (the status lamp on
  the car) that changes colour with the state.

## The basic algorithm

Here is the whole thing in plain steps. One pass of this runs every tick (every
small time step).

1. **See.** The sensor looks ahead and returns the cones the car can currently
   see, each with a colour and a position.
2. **Map.** Each cone is placed onto a growing map of the track. If a cone is
   close to one already on the map, they are averaged together so noise cancels
   out and the same cone is not added twice.
3. **Explore.** While the map is not finished, the planner looks at the cones
   right in front, finds the middle line between the left and right cones, and
   the car follows that middle line slowly. If it can only see one side, it
   follows that boundary's curve; if it sees no cones at all, it keeps going
   straight rather than stopping.
4. **Switch to racing.** The car keeps lapping in perception mode, mapping as it
   goes, until you press RACE. It then finishes the current lap and switches to
   race mode, now with a full map to plan from.
5. **Plan the racing line.** Now that all the cones are known, the planner joins
   the left and right cones into pairs (using a triangulation), takes the
   midpoints to get the full centre line, then slides each point sideways to
   make the line bend as little as possible while staying inside the cones. That
   smoother line is the racing line.
6. **Plan the speed.** For each point on the racing line it works out a safe
   speed, slow in tight corners and fast on straights, and smooths it so the car
   does not brake or accelerate harder than it can.
7. **Race.** The car follows the racing line using a simple steering method
   (pure pursuit, it aims at a point a little way ahead) and tries to hit the
   planned speed at each point.
8. **Stay safe.** The whole time, the AS state machine checks the rules. If the
   emergency stop is pressed, it jumps to the Emergency state and the EBS brings
   the car to a hard stop.

## Folder structure

```
path_planning/
├── app.py                  the live viewer (pygame window)
├── run_headless.py         runs a mission with no window, prints a summary
├── simulation.py           ties everything together, runs one tick at a time
│
├── perception.py           the fake cone sensor (what the car can see)
├── mapping.py              builds the cone map and detects a finished lap
├── centerline.py           finds the middle line between the cones
├── raceline.py             turns the centre line into a faster racing line
├── planner.py              the brain, switches between explore mode and race mode
├── geometry.py             small maths helpers (curvature, smoothing, speed, etc.)
├── vehicle.py              the car model (movement, steering, braking)
├── autonomous_system.py    the FSD rules and states (Off, Ready, Driving, ...)
├── tracks.py               the cone layouts for each mission
│
├── test_planner.py         unit tests (run headless, no window needed)
├── __init__.py             marks the folder as a package
│
├── requirements.txt        core dependencies (numpy, scipy)
├── requirements-ui.txt     extra dependency for the viewer (pygame)
├── pyproject.toml          project / packaging info
├── Dockerfile              to run it headless in a container (e.g. on a Jetson)
├── .dockerignore
├── .gitignore
│
├── assets/
│   └── TOR.png             the top down car picture used in the viewer
└── .github/
    └── workflows/
        └── tests.yml       runs the tests automatically on push
```

### What each file is for

- **app.py** is the window you watch. It draws the cones, the map, the planned
  path, the car and a side panel with the state and telemetry. You control it
  with buttons or keys.
- **run_headless.py** does the same driving but with no window. It is what you
  run on a server or on the Jetson. It prints how each mission went.
- **simulation.py** is the conductor. Every tick it runs the see, map, plan,
  drive, check loop described above and gives one snapshot of everything for the
  viewer to draw.
- **perception.py** pretends to be the camera and lidar. Given the car position
  it returns the cones in range and in front of the car, with a bit of noise.
- **mapping.py** keeps the growing map of cones, averaging out the sensor noise
  as the same cones are seen again.
- **centerline.py** takes the left and right cones and works out the line down
  the middle of the track, both for the small bit in front (explore) and for the
  whole loop (race).
- **raceline.py** takes that centre line and bends it into a faster racing line
  that still stays inside the cones, and finds the apex points.
- **planner.py** is the part the rest of the car talks to. It runs explore mode
  first, then builds the racing line and switches to race mode when asked.
- **geometry.py** holds the shared maths used everywhere: curvature, smoothing,
  resampling, the speed profile and some checks.
- **vehicle.py** is the simple car: how it moves, how it steers towards the path,
  and how it brakes (including the EBS).
- **autonomous_system.py** is the rules brain. It tracks the state, the timers
  (the 5 second and 3 second holds), the ASSI light and the safe stop.
- **tracks.py** holds the cone layouts for each mission. These are the ground
  truth tracks, the car never sees them directly, it only sees them through the
  sensor.
- **test_planner.py** has the tests that check the planner and the missions still
  work.

## Dependencies

- **Python 3.10 or newer** (3.12 is what we use).
- **numpy** and **scipy** for the maths. These are all you need to run it
  headless or run the tests.
- **pygame** only if you want the live viewer window.

The core dependencies are in `requirements.txt`, and the viewer one is in
`requirements-ui.txt`.

## How to run it

First get the dependencies:

```bash
pip install -r requirements.txt
```

**Run it headless** (no window, good for a server or the Jetson):

```bash
python run_headless.py              # autocross by default
python run_headless.py skidpad      # pick one mission
python run_headless.py all          # run every mission one by one
```

**Run the live viewer** (needs a screen):

```bash
pip install -r requirements-ui.txt
python app.py
```

In the viewer: pick a mission at the top (keys 1 to 4), turn **ASMS ON** (key A),
press **GO** (key G, it unlocks 5 seconds after Ready), and watch it drive. On
the loop missions (autocross, trackdrive) the car starts in **perception mode**,
lapping while it maps the track, with a lap counter shown on the map.
Press **RACE** (key M) to switch it to race mode; the switch happens once it next
completes a lap. **END** (key F) treats the current lap as the last one and
brings the car to a stop back at the start line. **STOP** (key S) stops the car
wherever it is. **EMERGENCY** (key E) fires the EBS, **RESET** (key R) starts
over, and **PAUSE** (key P) freezes the view.

You can also drop an **obstacle** by clicking on the track ahead of the car
(click it again to clear it). When the car sees an obstacle in its path it
brakes and stops; **CONTINUE** (key C) re-checks the track and starts the car
again only if the obstacle has been cleared.

Controls at a glance:

| Key | Button | What it does |
|-----|--------|--------------|
| 1 to 4 | mission tabs | pick the mission |
| A | ASMS | arm the autonomous system |
| G | GO | start the run (unlocks 5 s after Ready) |
| M | RACE | switch perception mode to race mode on the next completed lap |
| F | END | make this the final lap and stop at the start line |
| S | STOP | stop the car where it is |
| C | CONTINUE | re-check the track and resume after an obstacle is cleared |
| E | EMERGENCY | fire the EBS hard stop |
| R | RESET | restart the current mission |
| P | PAUSE | freeze the view |
| click | (on the map) | drop or clear an obstacle on the track |

**Run the tests:**

```bash
python -m unittest -v
```

**Run it in Docker** (this is how it is meant to go on the Jetson, headless):

```bash
docker build -t path-planning .
docker run --rm path-planning                 # runs every mission
docker run --rm path-planning python -m unittest -v
```

## Real F1 circuits (stress testing)

For a harder test than the built-in tracks, the program can run on the real
layouts of 40 Formula 1 circuits. Their centre lines come from the open
[bacinger/f1-circuits](https://github.com/bacinger/f1-circuits) dataset (MIT
licensed) bundled in `assets/f1_circuits.geojson`; `f1_tracks.py` projects them
to metres, scales them up, and builds the cone boundaries.

```bash
python f1_runner.py list           # list the circuits
python f1_runner.py monza          # race one circuit, print clearance
python f1_runner.py all            # run every circuit
python app.py f1 monaco            # drive a circuit in the viewer
```

Two things this surfaced, worth knowing:

- The cone-map recovery (`global_centerline`) was built for one simple loop and
  cannot trace a long, complex circuit, so on F1 tracks the program races
  directly on the known centre line instead of recovering it from the cones.
- Once racing, the planner and controller handle the real layouts: all 40 run
  end to end, and about 32 stay clear of the cones. The rest clip on their
  tightest hairpins and chicanes (Monaco, Baku, Suzuka, Silverstone, ...), where
  pure pursuit still cuts the corner.

## Scope (what this is and is not)

This is a student level project focused on the path planning part. The sensor,
the map and the car model are kept simple on purpose so the planning is the main
thing. It is not a full self driving stack: there is no real SLAM, no real
perception, no proper tyre model and no real controls. Again, it has not been
tested on hardware yet, that part still needs to be done.
