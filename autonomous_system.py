"""
Autonomous System status + safety logic (rules T14.8 / T14.9 / T15).

This is the small state machine every FS driverless car must run. It does not
move the car -- it decides what *state* the car is allowed to be in, which the
simulation then obeys. It is a direct reading of Figure 15 (AS Status Flowchart)
plus the timing and indicator rules around it:

    AS Off        nothing armed
    AS Ready      mission picked, ASMS on, brakes held -- waiting for "Go"
    AS Driving    running the mission
    AS Emergency  EBS fired, doing the safe-stop manoeuvre
    AS Finished   mission done and stopped, SDC open

Timers from the rules:
    T14.8.4  may only enter R2D (ready-to-drive) >= 5 s after AS Ready,
             and only on the RES "Go" signal.
    T14.8.5  must not start moving until >= 3 s in AS Driving.
    T15.4.1  EBS reaction time (open SDC -> deceleration) <= 200 ms.
"""

# --- states (T14.8) -------------------------------------------------------
AS_OFF = "AS Off"
AS_READY = "AS Ready"
AS_DRIVING = "AS Driving"
AS_EMERGENCY = "AS Emergency"
AS_FINISHED = "AS Finished"

# --- ASSI indicator look-up (T14.9.1) -------------------------------------
# (colour, blinking?) -- the UI turns this into a lamp.
ASSI = {
    AS_OFF:       ("off",    False),
    AS_READY:     ("yellow", False),   # yellow continuous
    AS_DRIVING:   ("yellow", True),    # yellow flashing
    AS_EMERGENCY: ("blue",   True),    # blue flashing
    AS_FINISHED:  ("blue",   False),   # blue continuous
}

READY_HOLD_S = 5.0     # T14.8.4
DRIVING_HOLD_S = 3.0   # T14.8.5
EBS_REACTION_S = 0.2   # T15.4.1 (200 ms)


class AutonomousSystem:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = AS_OFF
        self.mission = None        # selected mission (the AMI shows this)
        self.asms_on = False       # Autonomous System Master Switch (T14.5)
        self.go = False            # RES "Go" pressed
        self.estop = False         # RES emergency stop pressed
        self.asb_checks_ok = True  # initial ASB self-check (T15.3.1)
        self.ts_active = True      # tractive system live
        self.brakes_engaged = True # holding at standstill
        self.mission_finished = False
        self.vehicle_stopped = True
        self._t_ready = 0.0        # time spent in AS Ready
        self._t_driving = 0.0      # time spent in AS Driving
        self._t_estop = None       # time since estop fired (EBS reaction)
        self._r2d = False          # ready-to-drive, latched once granted

    # ---- driver / marshal inputs -----------------------------------------
    def select_mission(self, mission):
        self.mission = mission

    def set_asms(self, on):
        self.asms_on = bool(on)

    def press_go(self):
        self.go = True

    def trigger_emergency(self):
        """RES emergency stop / EBS -- opens the SDC (rules T14.3.3)."""
        self.estop = True
        if self._t_estop is None:
            self._t_estop = 0.0

    # ---- derived conditions ----------------------------------------------
    @property
    def ebs_activated(self):
        return self.estop

    @property
    def ready_satisfied(self):
        # left branch of Fig 15: mission + ASMS + ASB check + TS
        return (self.mission is not None and self.asms_on
                and self.asb_checks_ok and self.ts_active)

    @property
    def r2d(self):
        """Ready-to-drive: Go pressed after >= 5 s in AS Ready (T14.8.4).

        Latched -- once granted it stays granted, otherwise the AS would drop
        straight back to AS Ready when the ready timer resets.
        """
        return self._r2d

    @property
    def motion_allowed(self):
        """Vehicle may move only after >= 3 s in AS Driving (T14.8.5)."""
        return self.state == AS_DRIVING and self._t_driving >= DRIVING_HOLD_S

    @property
    def time_in_ready(self):
        """Seconds spent in AS Ready (public view of the dwell timer)."""
        return self._t_ready

    @property
    def time_in_driving(self):
        """Seconds spent in AS Driving (public view of the dwell timer)."""
        return self._t_driving

    @property
    def ebs_reacted(self):
        """True once the EBS reaction time has elapsed (T15.4.1)."""
        return self._t_estop is not None and self._t_estop >= EBS_REACTION_S

    # ---- the flowchart (Figure 15), evaluated every tick -----------------
    def update(self, dt, vehicle_stopped=True, mission_finished=False):
        self.vehicle_stopped = vehicle_stopped
        self.mission_finished = mission_finished
        if self._t_estop is not None:
            self._t_estop += dt

        # grant (and latch) ready-to-drive: "Go" after >= 5 s in AS Ready
        if (not self.ebs_activated and self.ready_satisfied and self.go
                and self._t_ready >= READY_HOLD_S):
            self._r2d = True

        if self.ebs_activated:
            # right branch
            if self.mission_finished and self.vehicle_stopped:
                self.state = AS_FINISHED
            else:
                self.state = AS_EMERGENCY     # SDC open at RES -> emergency
        else:
            if not self.ready_satisfied:
                self.state = AS_OFF
                self._r2d = False             # lost arming -> must re-arm
            elif self.r2d:
                self.state = AS_DRIVING
            elif self.brakes_engaged:
                self.state = AS_READY
            else:
                self.state = AS_OFF

        # accumulate the dwell timers
        self._t_ready = self._t_ready + dt if self.state == AS_READY else 0.0
        if self.state == AS_DRIVING:
            self._t_driving += dt
        else:
            self._t_driving = 0.0

        # natural finish (no estop): mission done and stopped, open the SDC
        if (not self.ebs_activated and self.mission_finished
                and self.vehicle_stopped and self.state == AS_DRIVING):
            self.state = AS_FINISHED
        return self.state

    # ---- indicator (T14.9) ------------------------------------------------
    def assi(self):
        return ASSI[self.state]
