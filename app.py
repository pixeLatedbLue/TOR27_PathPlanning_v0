"""
Real-time visualiser for the driverless path-planning pipeline.

Run it:

    python app.py

What you see, live:
    * cones streaming in from "perception" (sensor wedge + rays to each cone),
    * those detections being folded into the growing global map,
    * the planner's output drawn on the map -- the reactive local path while
      EXPLORING, then the minimum-curvature raceline while RACING,
    * the car (sprite "TOR" if you drop one in, else a drawn car) following it,
    * the Autonomous System state, ASSI indicator and telemetry (rules T14/T15).

Controls (buttons on the right, or keys):
    pick a mission (top bar) -> ASMS ON -> GO        keys: 1-7, A, G
    EMERGENCY fires the EBS safe-stop                key: E
    RESET re-arms the current mission                key: R
    pause / fast-forward                             keys: P, +/-
"""

import os
import numpy as np
import pygame

from simulation import Simulation, local_to_world
from autonomous_system import (AS_OFF, AS_READY, AS_DRIVING, AS_EMERGENCY,
                               AS_FINISHED)
from tracks import (ACCELERATION, SKIDPAD, AUTOCROSS, TRACKDRIVE, INSPECTION,
                    EBS_TEST, MANUAL)

# ---- layout --------------------------------------------------------------
WIDTH, HEIGHT = 1280, 820
TOPBAR_H = 48
PANEL_W = 300
MAP_RECT = pygame.Rect(0, TOPBAR_H, WIDTH - PANEL_W, HEIGHT - TOPBAR_H)
PANEL_RECT = pygame.Rect(WIDTH - PANEL_W, TOPBAR_H, PANEL_W, HEIGHT - TOPBAR_H)
FPS = 60

# ---- colours -------------------------------------------------------------
BG = (18, 20, 26)
PANEL = (28, 31, 40)
TOPBAR = (24, 27, 35)
GRID = (32, 36, 46)
BLUE = (54, 130, 246)
YELLOW = (245, 196, 56)
GT_FAINT = (66, 72, 86)
RACE = (235, 74, 74)
CENTER = (150, 152, 165)
LOCALP = (74, 210, 130)
APEX = (120, 245, 120)
CAR = (230, 232, 240)
TEXT = (228, 230, 238)
DIM = (150, 154, 168)
ACCENT = (90, 170, 255)
GOOD = (90, 210, 130)
WARN = (245, 170, 60)
BAD = (240, 80, 80)

ASSI_COLORS = {"off": (58, 60, 70), "yellow": (250, 212, 48), "blue": (66, 132, 255)}

MISSIONS = [(ACCELERATION, "ACCEL"), (SKIDPAD, "SKIDPAD"), (AUTOCROSS, "AUTOX"),
            (TRACKDRIVE, "TRACK"), (INSPECTION, "INSPECT"), (EBS_TEST, "EBS"),
            (MANUAL, "MANUAL")]

STATE_COLOR = {AS_OFF: DIM, AS_READY: WARN, AS_DRIVING: GOOD,
               AS_EMERGENCY: BAD, AS_FINISHED: ACCENT}


# --------------------------------------------------------------------------
# small button helper
# --------------------------------------------------------------------------
class Button:
    def __init__(self, rect, label, on_click, kind="action"):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.on_click = on_click
        self.kind = kind            # action / mission / toggle
        self.active = False         # highlighted (selected mission / toggle on)

    def draw(self, surf, font):
        base = PANEL
        border = (70, 75, 90)
        if self.kind == "go":
            base = (32, 90, 52) if not self.active else (44, 150, 80)
        elif self.kind == "stop":
            base = (110, 40, 40)
        elif self.active:
            base = (40, 70, 120)
            border = ACCENT
        pygame.draw.rect(surf, base, self.rect, border_radius=6)
        pygame.draw.rect(surf, border, self.rect, width=1, border_radius=6)
        txt = font.render(self.label, True, TEXT)
        surf.blit(txt, txt.get_rect(center=self.rect.center))

    def hit(self, pos):
        if self.rect.collidepoint(pos):
            self.on_click()
            return True
        return False


# --------------------------------------------------------------------------
# car sprite: use an image whose name is "TOR" if present, else draw one
# --------------------------------------------------------------------------
def load_car_sprite():
    here = os.path.dirname(os.path.abspath(__file__))
    for folder in (here, os.path.join(here, "assets")):
        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".gif", ""):
            path = os.path.join(folder, "TOR" + ext)
            if os.path.isfile(path):
                try:
                    return pygame.image.load(path).convert_alpha()
                except pygame.error:
                    pass
    return None


def draw_vector_car(length_px, width_px):
    """Fallback top-down formula car if no TOR image is supplied."""
    s = pygame.Surface((length_px, width_px), pygame.SRCALPHA)
    L, W = length_px, width_px
    body = [(L * 0.04, W * 0.40), (L * 0.62, W * 0.30), (L * 0.96, W * 0.42),
            (L * 0.96, W * 0.58), (L * 0.62, W * 0.70), (L * 0.04, W * 0.60)]
    pygame.draw.polygon(s, (40, 44, 54), body)
    pygame.draw.polygon(s, CAR, body, width=2)
    # nose cone + cockpit
    pygame.draw.polygon(s, ACCENT, [(L * 0.96, W * 0.45), (L * 1.0, W * 0.5),
                                    (L * 0.96, W * 0.55)])
    pygame.draw.circle(s, (20, 22, 28), (int(L * 0.45), int(W * 0.5)), int(W * 0.13))
    wheel = (16, 18, 24)
    for wx, wy in [(0.22, 0.08), (0.22, 0.92), (0.78, 0.10), (0.78, 0.90)]:
        r = pygame.Rect(0, 0, L * 0.16, W * 0.16)
        r.center = (L * wx, W * wy)
        pygame.draw.rect(s, wheel, r, border_radius=2)
    return s


# --------------------------------------------------------------------------
# world <-> screen
# --------------------------------------------------------------------------
class View:
    def __init__(self, track):
        pts = [track.left, track.right, track.centerline]
        pts = [p for p in pts if len(p)]
        allp = np.vstack(pts) if pts else np.array([[0, 0], [10, 10]])
        lo = allp.min(axis=0) - 6.0
        hi = allp.max(axis=0) + 6.0
        span = np.maximum(hi - lo, 1.0)
        self.scale = min(MAP_RECT.width / span[0], MAP_RECT.height / span[1])
        # centre the track in the map rect
        self.cx, self.cy = (lo + hi) / 2.0
        self.lo, self.hi = lo, hi

    def w2s(self, p):
        x = MAP_RECT.centerx + (p[0] - self.cx) * self.scale
        y = MAP_RECT.centery - (p[1] - self.cy) * self.scale
        return int(x), int(y)


# --------------------------------------------------------------------------
# the app
# --------------------------------------------------------------------------
class App:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("FSD Path Planning -- live pipeline")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        # a comma-separated list: pygame picks the first font available, so the
        # UI looks the same on Windows (Consolas) and on Linux/Jetson (DejaVu).
        mono = "consolas,dejavusansmono,liberationmono,freemono,monospace"
        self.font = pygame.font.SysFont(mono, 16)
        self.font_s = pygame.font.SysFont(mono, 13)
        self.font_b = pygame.font.SysFont(mono, 22, bold=True)

        self.sim = Simulation(dt=0.04)
        self.sim.select_mission(AUTOCROSS)
        self.view = View(self.sim.track)
        self.sprite = load_car_sprite()
        self.trail = []
        self.paused = False
        self.time_scale = 2
        self.buttons = []
        self._build_buttons()

    # ---- buttons ----------------------------------------------------------
    def _build_buttons(self):
        self.buttons.clear()
        n = len(MISSIONS)
        bw = (MAP_RECT.width - 8) / n
        for i, (mid, label) in enumerate(MISSIONS):
            b = Button((4 + i * bw, 8, bw - 6, TOPBAR_H - 16), label,
                       lambda m=mid: self.choose_mission(m), kind="mission")
            b.active = (mid == self.sim.mission)
            self.buttons.append(b)
        self.mission_buttons = list(self.buttons)

        px = PANEL_RECT.x + 16
        pw = PANEL_W - 32
        y = HEIGHT - 250
        self.btn_asms = Button((px, y, pw, 40), "ASMS: OFF", self.toggle_asms, "toggle")
        self.btn_go = Button((px, y + 48, pw // 2 - 4, 44), "GO", self.go, "go")
        self.btn_stop = Button((px + pw // 2 + 4, y + 48, pw // 2 - 4, 44),
                               "EMERGENCY", self.emergency, "stop")
        self.btn_reset = Button((px, y + 100, pw, 36), "RESET", self.reset, "action")
        self.btn_pause = Button((px, y + 142, pw // 2 - 4, 32), "PAUSE",
                                self.toggle_pause, "action")
        self.btn_speed = Button((px + pw // 2 + 4, y + 142, pw // 2 - 4, 32),
                                f"x{self.time_scale}", self.cycle_speed, "action")
        self.buttons += [self.btn_asms, self.btn_go, self.btn_stop,
                         self.btn_reset, self.btn_pause, self.btn_speed]

    # ---- actions ----------------------------------------------------------
    def choose_mission(self, mission):
        self.sim.select_mission(mission)
        self.view = View(self.sim.track)
        self.trail.clear()
        for b in self.mission_buttons:
            b.active = False
        for b, (mid, _) in zip(self.mission_buttons, MISSIONS):
            b.active = (mid == mission)
        self.btn_asms.label = "ASMS: OFF"
        self.btn_asms.active = False

    def toggle_asms(self):
        new = not self.sim.AS.asms_on
        self.sim.set_asms(new)
        self.btn_asms.active = new
        self.btn_asms.label = f"ASMS: {'ON' if new else 'OFF'}"

    def go(self):
        self.sim.press_go()
        self.btn_go.active = True

    def emergency(self):
        self.sim.emergency()

    def reset(self):
        self.sim.reset()
        self.trail.clear()
        self.btn_asms.label = "ASMS: OFF"
        self.btn_asms.active = False
        self.btn_go.active = False

    def toggle_pause(self):
        self.paused = not self.paused
        self.btn_pause.label = "RESUME" if self.paused else "PAUSE"

    def cycle_speed(self):
        order = [1, 2, 4, 8]
        self.time_scale = order[(order.index(self.time_scale) + 1) % len(order)]
        self.btn_speed.label = f"x{self.time_scale}"

    # ---- event loop -------------------------------------------------------
    def run(self):
        running = True
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    for b in self.buttons:
                        if b.hit(e.pos):
                            break
                elif e.type == pygame.KEYDOWN:
                    self._key(e.key)
            if not self.paused:
                for _ in range(self.time_scale):
                    self.sim.tick()
            self._record_trail()
            self.draw()
            self.clock.tick(FPS)
        pygame.quit()

    def _key(self, key):
        if pygame.K_1 <= key <= pygame.K_7:
            self.choose_mission(MISSIONS[key - pygame.K_1][0])
        elif key == pygame.K_a:
            self.toggle_asms()
        elif key == pygame.K_g:
            self.go()
        elif key == pygame.K_e:
            self.emergency()
        elif key == pygame.K_r:
            self.reset()
        elif key == pygame.K_p:
            self.toggle_pause()
        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_MINUS):
            self.cycle_speed()

    def _record_trail(self):
        x, y, _ = self.sim.vehicle.pose
        if not self.trail or np.hypot(x - self.trail[-1][0], y - self.trail[-1][1]) > 0.3:
            self.trail.append((x, y))
            if len(self.trail) > 1500:
                self.trail.pop(0)

    # ---- drawing ----------------------------------------------------------
    def draw(self):
        self.screen.fill(BG)
        snap = self.sim.snapshot()
        self._draw_map(snap)
        self._draw_topbar()
        self._draw_panel(snap)
        pygame.display.flip()

    def _draw_map(self, snap):
        self.screen.set_clip(MAP_RECT)
        pygame.draw.rect(self.screen, BG, MAP_RECT)
        self._draw_grid()
        track = snap["track"]
        w2s = self.view.w2s

        # ground-truth cones, faint -- the part of the track not yet discovered
        for c in track.left:
            pygame.draw.circle(self.screen, GT_FAINT, w2s(c), 3, 1)
        for c in track.right:
            pygame.draw.circle(self.screen, GT_FAINT, w2s(c), 3, 1)

        # planner output ----------------------------------------------------
        race = snap["race"]
        if race is not None:
            cl = np.vstack([race["centerline"], race["centerline"][0]])
            self._dashed(cl, CENTER)
            rl = np.vstack([race["raceline"], race["raceline"][0]])
            pygame.draw.lines(self.screen, RACE, False, [w2s(p) for p in rl], 3)
            for idx in race["apex_indices"]:
                pygame.draw.circle(self.screen, APEX, w2s(race["raceline"][idx]), 5)
        if snap["skidpad_loops"]:
            for loop in snap["skidpad_loops"]:
                lp = np.vstack([loop, loop[0]])
                pygame.draw.lines(self.screen, RACE, False, [w2s(p) for p in lp], 2)
        lp = snap["local_path"]
        if lp is not None and len(lp) > 1 and race is None:
            pygame.draw.lines(self.screen, LOCALP, False, [w2s(p) for p in lp], 3)

        # trail -------------------------------------------------------------
        if len(self.trail) > 1:
            pygame.draw.lines(self.screen, (70, 90, 120), False,
                              [w2s(p) for p in self.trail], 2)

        # mapped cones (the live dataset) -----------------------------------
        for c in snap["map_left"]:
            pygame.draw.circle(self.screen, BLUE, w2s(c), 4)
        for c in snap["map_right"]:
            pygame.draw.circle(self.screen, YELLOW, w2s(c), 4)

        self._draw_perception(snap)
        self._draw_car(snap)
        self.screen.set_clip(None)

    def _draw_grid(self):
        step = max(5.0, round(10.0))
        # light grid every ~10 m
        x = self.view.lo[0]
        while x <= self.view.hi[0]:
            p1 = self.view.w2s((x, self.view.lo[1]))
            p2 = self.view.w2s((x, self.view.hi[1]))
            pygame.draw.line(self.screen, GRID, p1, p2, 1)
            x += 10.0
        y = self.view.lo[1]
        while y <= self.view.hi[1]:
            p1 = self.view.w2s((self.view.lo[0], y))
            p2 = self.view.w2s((self.view.hi[0], y))
            pygame.draw.line(self.screen, GRID, p1, p2, 1)
            y += 10.0

    def _dashed(self, pts, color, dash=6):
        scr = [self.view.w2s(p) for p in pts]
        for i in range(0, len(scr) - 1, 2):
            pygame.draw.line(self.screen, color, scr[i], scr[i + 1], 1)

    def _draw_perception(self, snap):
        pose = snap["pose"]
        px, py, theta = pose
        car_s = self.view.w2s((px, py))
        rng = self.sim.perception.max_range
        half = self.sim.perception.half_fov
        # FOV wedge
        wedge = [car_s]
        for a in np.linspace(-half, half, 24):
            wp = (px + rng * np.cos(theta + a), py + rng * np.sin(theta + a))
            wedge.append(self.view.w2s(wp))
        fov = pygame.Surface((MAP_RECT.width, MAP_RECT.height), pygame.SRCALPHA)
        off = [(p[0] - MAP_RECT.x, p[1] - MAP_RECT.y) for p in wedge]
        pygame.draw.polygon(fov, (90, 150, 230, 22), off)
        self.screen.blit(fov, MAP_RECT.topleft)
        # rays + rings to the cones detected this tick
        for local, col in ((snap["det_left_local"], BLUE),
                           (snap["det_right_local"], YELLOW)):
            world = local_to_world(local, pose)
            if world is None:
                continue
            for c in world:
                cs = self.view.w2s(c)
                pygame.draw.line(self.screen, (90, 110, 140), car_s, cs, 1)
                pygame.draw.circle(self.screen, (255, 255, 255), cs, 6, 1)

    def _draw_car(self, snap):
        px, py, theta = snap["pose"]
        car_len_px = max(14, int(2.9 * self.view.scale))
        car_w_px = max(7, int(1.5 * self.view.scale))
        if self.sprite is not None:
            base = pygame.transform.smoothscale(self.sprite, (car_len_px, car_w_px))
        else:
            base = draw_vector_car(car_len_px, car_w_px)
        rot = pygame.transform.rotate(base, np.degrees(theta))
        self.screen.blit(rot, rot.get_rect(center=self.view.w2s((px, py))))

    # ---- top bar ----------------------------------------------------------
    def _draw_topbar(self):
        pygame.draw.rect(self.screen, TOPBAR, (0, 0, WIDTH, TOPBAR_H))
        for b in self.mission_buttons:
            b.draw(self.screen, self.font_s)

    # ---- side panel -------------------------------------------------------
    def _draw_panel(self, snap):
        pygame.draw.rect(self.screen, PANEL, PANEL_RECT)
        x = PANEL_RECT.x + 16
        y = PANEL_RECT.y + 14

        # AS state
        state = snap["state"]
        self.screen.blit(self.font_s.render("AUTONOMOUS SYSTEM", True, DIM), (x, y))
        y += 20
        self.screen.blit(self.font_b.render(state, True, STATE_COLOR[state]), (x, y))
        y += 34

        # ASSI lamps
        color, blink = snap["assi"]
        on = (pygame.time.get_ticks() // 250) % 2 == 0 if blink else True
        lit = ASSI_COLORS[color] if on else (40, 42, 50)
        self.screen.blit(self.font_s.render("ASSI", True, DIM), (x, y + 6))
        for i in range(3):
            c = (x + 70 + i * 34, y + 14)
            pygame.draw.circle(self.screen, lit if color != "off" else ASSI_COLORS["off"], c, 11)
            pygame.draw.circle(self.screen, (15, 16, 20), c, 11, 2)
        y += 42

        # AMI / mission
        ami = dict(MISSIONS).get(snap["mission"], snap["mission"]).upper()
        self._row(x, y, "AMI / MISSION", ami); y += 24
        self._row(x, y, "PHASE", snap["phase"].upper()); y += 24
        asms = "ON" if snap["asms"] else "OFF"
        self._row(x, y, "ASMS", asms, GOOD if snap["asms"] else DIM); y += 24

        # telemetry
        y += 6
        pygame.draw.line(self.screen, GRID, (x, y), (PANEL_RECT.right - 16, y)); y += 10
        self._row(x, y, "SPEED", f"{snap['speed']:5.1f} m/s"); y += 24
        self._row(x, y, "STEER", f"{np.degrees(snap['steer']):5.1f} deg"); y += 24
        self._row(x, y, "LAP", f"{snap['lap']} / {snap['laps_required']}"); y += 24
        self._row(x, y, "MAP CONES", str(snap['map_left'].shape[0] + snap['map_right'].shape[0])); y += 24
        self._row(x, y, "SIM TIME", f"{snap['sim_time']:5.1f} s"); y += 24
        if snap["ebs_decel"] > 0.1:
            self._row(x, y, "EBS DECEL", f"{snap['ebs_decel']:4.1f} m/s2",
                      GOOD if snap['ebs_decel'] > 10 else WARN); y += 24

        # message line
        y += 6
        pygame.draw.line(self.screen, GRID, (x, y), (PANEL_RECT.right - 16, y)); y += 8
        self._wrap(x, y, snap["message"], PANEL_W - 32)

        for b in self.buttons:
            if b not in self.mission_buttons:
                b.draw(self.screen, self.font)

    def _row(self, x, y, label, value, vcolor=TEXT):
        self.screen.blit(self.font_s.render(label, True, DIM), (x, y))
        v = self.font.render(value, True, vcolor)
        self.screen.blit(v, (PANEL_RECT.right - 16 - v.get_width(), y - 2))

    def _wrap(self, x, y, text, width):
        words = text.split()
        line = ""
        for w in words:
            test = (line + " " + w).strip()
            if self.font_s.size(test)[0] > width and line:
                self.screen.blit(self.font_s.render(line, True, DIM), (x, y))
                y += 18
                line = w
            else:
                line = test
        if line:
            self.screen.blit(self.font_s.render(line, True, DIM), (x, y))


if __name__ == "__main__":
    App().run()
