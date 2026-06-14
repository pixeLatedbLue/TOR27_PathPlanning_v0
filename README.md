# Path Planning for a Formula Student Driverless Car

This is a path planning pipeline for a Formula Student Driverless (FSD) car,
with a small live viewer so you can watch it work. The car gets cones from a
(fake) sensor, builds a map of the track, plans a path through it, and drives
that path while following the FSD rules. It is built for the four driving
missions: **acceleration, skidpad, autocross and trackdrive**, plus the **EBS
test**, **inspection** and a **manual** mode.

## Status: not tested yet

Please read this first. This project has **not been properly tested yet and is
still to be tested**. There are a few unit tests in the repo, but the whole
thing has not been checked on the real Jetson board or in an actual run, so
treat everything here as work in progress. Numbers, behaviour and timings may
still change once it runs on hardware.

## What it does

The idea is simple. The track is marked by cones (blue on the left, yellow on
the right). The car cannot see the whole track at once, it only sees the cones
near it. So it drives slowly first to map the track, and once it knows the full
loop it plans a faster line and races on it.

It handles all the FSD missions:

| Mission | What happens |
|---------|--------------|
| **Acceleration** | Drive straight down a 75 m lane, then brake to a stop. |
| **Skidpad** | Drive the figure of eight (two circles), right side twice then left side twice. |
| **Autocross** | Map one lap, then race one fast lap on the planned line. |
| **Trackdrive** | Map one lap, then race the rest (10 laps in total). |
| **EBS test** | Speed up, fire the emergency brake, and show the deceleration. |
| **Inspection** | Car is jacked up, it just spins the wheels and steers side to side for about 27 seconds. |
| **Manual** | Human style baseline, the car drives the loop without the racing line. |

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
   the car follows that middle line slowly.
4. **Close the loop.** When the car comes back near where it started after
   driving far enough, the map is treated as complete.
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
- **mapping.py** keeps the growing map of cones and decides when a full lap has
  been mapped.
- **centerline.py** takes the left and right cones and works out the line down
  the middle of the track, both for the small bit in front (explore) and for the
  whole loop (race).
- **raceline.py** takes that centre line and bends it into a faster racing line
  that still stays inside the cones, and finds the apex points.
- **planner.py** is the part the rest of the car talks to. It runs explore mode
  first, then switches to race mode once the map is done.
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

In the viewer: pick a mission at the top (keys 1 to 7), turn **ASMS ON** (key A),
press **GO** (key G, it unlocks 5 seconds after Ready), and watch it drive.
**EMERGENCY** (key E) fires the EBS, **RESET** (key R) starts over, and **PAUSE**
(key P) and the speed button let you slow it down or speed it up.

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

## Scope (what this is and is not)

This is a student level project focused on the path planning part. The sensor,
the map and the car model are kept simple on purpose so the planning is the main
thing. It is not a full self driving stack: there is no real SLAM, no real
perception, no proper tyre model and no real controls. Again, it has not been
tested on hardware yet, that part still needs to be done.
