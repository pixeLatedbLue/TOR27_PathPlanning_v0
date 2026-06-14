AS_OFF = "AS Off"
AS_READY = "AS Ready"
AS_DRIVING = "AS Driving"
AS_EMERGENCY = "AS Emergency"
AS_FINISHED = "AS Finished"

ASSI = {
    AS_OFF:       ("off",    False),
    AS_READY:     ("yellow", False),
    AS_DRIVING:   ("yellow", True),
    AS_EMERGENCY: ("blue",   True),
    AS_FINISHED:  ("blue",   False),
}

READY_HOLD_S = 5.0
DRIVING_HOLD_S = 3.0
EBS_REACTION_S = 0.2


class AutonomousSystem:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = AS_OFF
        self.mission = None
        self.asms_on = False
        self.go = False
        self.estop = False
        self.asb_checks_ok = True
        self.ts_active = True
        self.brakes_engaged = True
        self.mission_finished = False
        self.vehicle_stopped = True
        self._t_ready = 0.0
        self._t_driving = 0.0
        self._t_estop = None
        self._r2d = False

    def select_mission(self, mission):
        self.mission = mission

    def set_asms(self, on):
        self.asms_on = bool(on)

    def press_go(self):
        self.go = True

    def trigger_emergency(self):
        self.estop = True
        if self._t_estop is None:
            self._t_estop = 0.0

    @property
    def ebs_activated(self):
        return self.estop

    @property
    def ready_satisfied(self):
        return (self.mission is not None and self.asms_on
                and self.asb_checks_ok and self.ts_active)

    @property
    def r2d(self):
        return self._r2d

    @property
    def motion_allowed(self):
        return self.state == AS_DRIVING and self._t_driving >= DRIVING_HOLD_S

    @property
    def time_in_ready(self):
        return self._t_ready

    @property
    def time_in_driving(self):
        return self._t_driving

    @property
    def ebs_reacted(self):
        return self._t_estop is not None and self._t_estop >= EBS_REACTION_S

    def update(self, dt, vehicle_stopped=True, mission_finished=False):
        self.vehicle_stopped = vehicle_stopped
        self.mission_finished = mission_finished
        if self._t_estop is not None:
            self._t_estop += dt

        if (not self.ebs_activated and self.ready_satisfied and self.go
                and self._t_ready >= READY_HOLD_S):
            self._r2d = True

        if self.ebs_activated:
            if self.mission_finished and self.vehicle_stopped:
                self.state = AS_FINISHED
            else:
                self.state = AS_EMERGENCY
        else:
            if not self.ready_satisfied:
                self.state = AS_OFF
                self._r2d = False
            elif self.r2d:
                self.state = AS_DRIVING
            elif self.brakes_engaged:
                self.state = AS_READY
            else:
                self.state = AS_OFF

        self._t_ready = self._t_ready + dt if self.state == AS_READY else 0.0
        if self.state == AS_DRIVING:
            self._t_driving += dt
        else:
            self._t_driving = 0.0

        if (not self.ebs_activated and self.mission_finished
                and self.vehicle_stopped and self.state == AS_DRIVING):
            self.state = AS_FINISHED
        return self.state

    def assi(self):
        return ASSI[self.state]
