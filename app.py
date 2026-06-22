import os
import sys
import numpy as np
import pygame

from simulation import Simulation, local_to_world
from autonomous_system import (AS_OFF, AS_READY, AS_DRIVING, AS_EMERGENCY,
                               AS_FINISHED)
from tracks import ACCELERATION, SKIDPAD, AUTOCROSS, TRACKDRIVE

WIDTH, HEIGHT = 1280, 820
TOPBAR_H = 48
PANEL_W = 300
MAP_RECT = pygame.Rect(0, TOPBAR_H, WIDTH - PANEL_W, HEIGHT - TOPBAR_H)
PANEL_RECT = pygame.Rect(WIDTH - PANEL_W, TOPBAR_H, PANEL_W, HEIGHT - TOPBAR_H)
FPS = 60

BG = (12, 14, 17)
PANEL = (19, 22, 27)
PANEL2 = (26, 30, 37)
TOPBAR = (15, 17, 21)
GRID = (26, 30, 37)
BLUE = (60, 132, 246)
YELLOW = (244, 196, 60)
GT_FAINT = (50, 56, 68)
RACE = (232, 28, 64)
CENTER = (120, 126, 140)
LOCALP = (176, 230, 64)
APEX = (255, 96, 124)
CAR = (236, 238, 244)
TEXT = (236, 238, 244)
DIM = (116, 122, 136)
ACCENT = (232, 28, 64)
LIVE = (176, 230, 64)
GOOD = (176, 230, 64)
WARN = (245, 190, 70)
BAD = (232, 28, 64)
FINISH = (84, 178, 255)
OBST = (255, 122, 40)

ASSI_COLORS = {"off": (52, 56, 66), "yellow": (250, 212, 48), "blue": (66, 132, 255)}

MISSIONS = [(ACCELERATION, "ACCEL"), (SKIDPAD, "SKIDPAD"),
            (AUTOCROSS, "AUTOX"), (TRACKDRIVE, "TRACK")]

STATE_COLOR = {AS_OFF: DIM, AS_READY: WARN, AS_DRIVING: LIVE,
               AS_EMERGENCY: BAD, AS_FINISHED: FINISH}


class Button:
    def __init__(self, rect, label, on_click, kind="action"):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.on_click = on_click
        self.kind = kind
        self.active = False

    def draw(self, surf, font):
        base = (28, 32, 39)
        border = (52, 58, 70)
        txtcol = TEXT
        if self.kind == "go":
            base = (118, 174, 48) if self.active else (38, 56, 24)
            border = LIVE
            txtcol = (14, 18, 10) if self.active else LIVE
        elif self.kind == "stop":
            base = (66, 22, 32)
            border = ACCENT
            txtcol = (255, 208, 214)
        elif self.active:
            base = (52, 24, 32)
            border = ACCENT
            txtcol = TEXT
        pygame.draw.rect(surf, base, self.rect, border_radius=2)
        pygame.draw.rect(surf, border, self.rect, width=2, border_radius=2)
        txt = font.render(self.label, True, txtcol)
        surf.blit(txt, txt.get_rect(center=self.rect.center))

    def hit(self, pos):
        if self.rect.collidepoint(pos):
            self.on_click()
            return True
        return False


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
    s = pygame.Surface((length_px, width_px), pygame.SRCALPHA)
    L, W = length_px, width_px
    body = [(L * 0.04, W * 0.40), (L * 0.62, W * 0.30), (L * 0.96, W * 0.42),
            (L * 0.96, W * 0.58), (L * 0.62, W * 0.70), (L * 0.04, W * 0.60)]
    pygame.draw.polygon(s, (40, 44, 54), body)
    pygame.draw.polygon(s, CAR, body, width=2)
    pygame.draw.polygon(s, ACCENT, [(L * 0.96, W * 0.45), (L * 1.0, W * 0.5),
                                    (L * 0.96, W * 0.55)])
    pygame.draw.circle(s, (20, 22, 28), (int(L * 0.45), int(W * 0.5)), int(W * 0.13))
    wheel = (16, 18, 24)
    for wx, wy in [(0.22, 0.08), (0.22, 0.92), (0.78, 0.10), (0.78, 0.90)]:
        r = pygame.Rect(0, 0, L * 0.16, W * 0.16)
        r.center = (L * wx, W * wy)
        pygame.draw.rect(s, wheel, r, border_radius=2)
    return s


class View:
    def __init__(self, track):
        pts = [track.left, track.right, track.centerline]
        pts = [p for p in pts if len(p)]
        allp = np.vstack(pts) if pts else np.array([[0, 0], [10, 10]])
        lo = allp.min(axis=0) - 6.0
        hi = allp.max(axis=0) + 6.0
        span = np.maximum(hi - lo, 1.0)
        self.scale = min(MAP_RECT.width / span[0], MAP_RECT.height / span[1])
        self.cx, self.cy = (lo + hi) / 2.0
        self.lo, self.hi = lo, hi

    def w2s(self, p):
        x = MAP_RECT.centerx + (p[0] - self.cx) * self.scale
        y = MAP_RECT.centery - (p[1] - self.cy) * self.scale
        return int(x), int(y)

    def s2w(self, sx, sy):
        wx = (sx - MAP_RECT.centerx) / self.scale + self.cx
        wy = -(sy - MAP_RECT.centery) / self.scale + self.cy
        return np.array([wx, wy])


class App:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("TOR27 -- Autonomous Path Planning")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        mono = "consolas,dejavusansmono,liberationmono,freemono,monospace"
        self.font = pygame.font.SysFont(mono, 16)
        self.font_s = pygame.font.SysFont(mono, 13)
        self.font_n = pygame.font.SysFont(mono, 16, bold=True)
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

    def _build_buttons(self):
        self.buttons.clear()
        logo_w = 150
        n = len(MISSIONS)
        bw = (MAP_RECT.width - logo_w - 8) / n
        for i, (mid, label) in enumerate(MISSIONS):
            b = Button((logo_w + i * bw, 9, bw - 6, TOPBAR_H - 18), label,
                       lambda m=mid: self.choose_mission(m), kind="mission")
            b.active = (mid == self.sim.mission)
            self.buttons.append(b)
        self.mission_buttons = list(self.buttons)

        px = PANEL_RECT.x + 16
        pw = PANEL_W - 32
        y = HEIGHT - 250
        self.btn_asms = Button((px, y, pw, 38), "ASMS: OFF", self.toggle_asms, "toggle")
        self.btn_go = Button((px, y + 44, pw // 2 - 4, 42), "GO", self.go, "go")
        self.btn_stop = Button((px + pw // 2 + 4, y + 44, pw // 2 - 4, 42),
                               "EMERGENCY", self.emergency, "stop")
        self.btn_race = Button((px, y + 92, pw // 2 - 4, 34), "RACE",
                               self.race_mode, "action")
        self.btn_continue = Button((px + pw // 2 + 4, y + 92, pw // 2 - 4, 34),
                                   "CONTINUE", self.continue_drive, "go")
        self.btn_stoprun = Button((px, y + 132, pw // 2 - 4, 32), "STOP",
                                  self.stop_run, "action")
        self.btn_end = Button((px + pw // 2 + 4, y + 132, pw // 2 - 4, 32), "END",
                              self.end_run, "action")
        self.btn_reset = Button((px, y + 170, pw // 2 - 4, 32), "RESET",
                                self.reset, "action")
        self.btn_pause = Button((px + pw // 2 + 4, y + 170, pw // 2 - 4, 32),
                                "PAUSE", self.toggle_pause, "action")
        self.buttons += [self.btn_asms, self.btn_go, self.btn_stop, self.btn_race,
                         self.btn_continue, self.btn_stoprun, self.btn_end,
                         self.btn_reset, self.btn_pause]

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
        self.btn_race.active = False

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

    def stop_run(self):
        self.sim.request_stop()

    def reset(self):
        self.sim.reset()
        self.trail.clear()
        self.btn_asms.label = "ASMS: OFF"
        self.btn_asms.active = False
        self.btn_go.active = False
        self.btn_race.active = False

    def toggle_pause(self):
        self.paused = not self.paused
        self.btn_pause.label = "RESUME" if self.paused else "PAUSE"

    def end_run(self):
        self.sim.request_end()

    def race_mode(self):
        self.sim.request_race()
        self.btn_race.active = True

    def continue_drive(self):
        self.sim.continue_drive()

    def place_obstacle(self, screen_pos):
        world = self.view.s2w(*screen_pos)
        obs = self.sim.obstacle
        if obs is not None and np.linalg.norm(world - obs) < 2.5:
            self.sim.clear_obstacle()
        else:
            self.sim.set_obstacle(world)

    def run(self):
        running = True
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    if not any(b.hit(e.pos) for b in self.buttons) \
                            and MAP_RECT.collidepoint(e.pos):
                        self.place_obstacle(e.pos)
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
        if pygame.K_1 <= key <= pygame.K_4:
            self.choose_mission(MISSIONS[key - pygame.K_1][0])
        elif key == pygame.K_a:
            self.toggle_asms()
        elif key == pygame.K_g:
            self.go()
        elif key == pygame.K_e:
            self.emergency()
        elif key == pygame.K_s:
            self.stop_run()
        elif key == pygame.K_f:
            self.end_run()
        elif key == pygame.K_m:
            self.race_mode()
        elif key == pygame.K_c:
            self.continue_drive()
        elif key == pygame.K_r:
            self.reset()
        elif key == pygame.K_p:
            self.toggle_pause()

    def _record_trail(self):
        x, y, _ = self.sim.vehicle.pose
        if not self.trail or np.hypot(x - self.trail[-1][0], y - self.trail[-1][1]) > 0.3:
            self.trail.append((x, y))
            if len(self.trail) > 1500:
                self.trail.pop(0)

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

        for c in track.left:
            pygame.draw.circle(self.screen, GT_FAINT, w2s(c), 3, 1)
        for c in track.right:
            pygame.draw.circle(self.screen, GT_FAINT, w2s(c), 3, 1)

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

        if len(self.trail) > 1:
            pygame.draw.lines(self.screen, (70, 90, 120), False,
                              [w2s(p) for p in self.trail], 2)

        for c in snap["map_left"]:
            pygame.draw.circle(self.screen, BLUE, w2s(c), 4)
        for c in snap["map_right"]:
            pygame.draw.circle(self.screen, YELLOW, w2s(c), 4)

        self._draw_perception(snap)
        self._draw_car(snap)
        self._draw_obstacle(snap)
        self._draw_lap_counter(snap)
        self._draw_legend()
        self.screen.set_clip(None)

    def _draw_obstacle(self, snap):
        obs = snap["obstacle"]
        if obs is None:
            return
        c = self.view.w2s(obs)
        r = max(9, int(1.2 * self.view.scale))
        pygame.draw.circle(self.screen, OBST, c, r)
        pygame.draw.circle(self.screen, (255, 232, 210), c, r, 2)
        o = int(r * 0.5)
        pygame.draw.line(self.screen, (28, 10, 10), (c[0] - o, c[1] - o), (c[0] + o, c[1] + o), 2)
        pygame.draw.line(self.screen, (28, 10, 10), (c[0] - o, c[1] + o), (c[0] + o, c[1] - o), 2)
        if snap["obstacle_blocked"]:
            banner = self.font_b.render("OBSTACLE -- STOPPED", True, (255, 210, 210))
            bx = MAP_RECT.centerx - banner.get_width() // 2
            box = pygame.Rect(bx - 14, MAP_RECT.y + 14, banner.get_width() + 28,
                              banner.get_height() + 10)
            pygame.draw.rect(self.screen, (120, 40, 40), box, border_radius=6)
            pygame.draw.rect(self.screen, BAD, box, width=1, border_radius=6)
            self.screen.blit(banner, (bx, MAP_RECT.y + 18))

    def _draw_legend(self):
        items = [(RACE, "raceline"), (LOCALP, "local path"),
                 (CENTER, "centreline"), (BLUE, "left cone"),
                 (YELLOW, "right cone"), (OBST, "obstacle")]
        lh, pad, w = 18, 8, 132
        h = pad * 2 + lh * len(items)
        x0 = MAP_RECT.x + 12
        y0 = MAP_RECT.bottom - h - 12
        panel = pygame.Surface((w, h), pygame.SRCALPHA)
        panel.fill((22, 25, 32, 215))
        self.screen.blit(panel, (x0, y0))
        pygame.draw.rect(self.screen, (70, 75, 90),
                         pygame.Rect(x0, y0, w, h), width=1, border_radius=6)
        for i, (col, label) in enumerate(items):
            cy = y0 + pad + i * lh + lh // 2
            if "cone" in label:
                pygame.draw.circle(self.screen, col, (x0 + 16, cy), 4)
            elif label == "obstacle":
                pygame.draw.circle(self.screen, col, (x0 + 16, cy), 6)
            else:
                pygame.draw.line(self.screen, col, (x0 + 8, cy), (x0 + 26, cy), 3)
            self.screen.blit(self.font_s.render(label, True, DIM),
                             (x0 + 34, cy - 8))

    def _draw_lap_counter(self, snap):
        if snap["mission"] not in (AUTOCROSS, TRACKDRIVE, SKIDPAD) or snap["lap"] < 1:
            return
        txt = self.font_b.render(f"LAP {snap['lap']}", True, TEXT)
        box = pygame.Rect(MAP_RECT.x + 12, MAP_RECT.y + 12,
                          txt.get_width() + 24, txt.get_height() + 12)
        pygame.draw.rect(self.screen, (24, 27, 35), box, border_radius=6)
        pygame.draw.rect(self.screen, ACCENT, box, width=1, border_radius=6)
        self.screen.blit(txt, (box.x + 12, box.y + 6))

    def _draw_grid(self):
        step = max(5.0, round(10.0))
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
        wedge = [car_s]
        for a in np.linspace(-half, half, 24):
            wp = (px + rng * np.cos(theta + a), py + rng * np.sin(theta + a))
            wedge.append(self.view.w2s(wp))
        fov = pygame.Surface((MAP_RECT.width, MAP_RECT.height), pygame.SRCALPHA)
        off = [(p[0] - MAP_RECT.x, p[1] - MAP_RECT.y) for p in wedge]
        pygame.draw.polygon(fov, (90, 150, 230, 22), off)
        self.screen.blit(fov, MAP_RECT.topleft)
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

    def _draw_topbar(self):
        pygame.draw.rect(self.screen, TOPBAR, (0, 0, WIDTH, TOPBAR_H))
        pygame.draw.rect(self.screen, ACCENT, (0, 0, 6, TOPBAR_H))
        logo = self.font_b.render("TOR27", True, TEXT)
        self.screen.blit(logo, (18, 11))
        tag = self.font_s.render("FSD", True, ACCENT)
        self.screen.blit(tag, (18 + logo.get_width() + 8, 20))
        pygame.draw.line(self.screen, ACCENT, (0, TOPBAR_H - 1),
                         (WIDTH, TOPBAR_H - 1), 1)
        for b in self.mission_buttons:
            b.draw(self.screen, self.font_s)

    def _draw_panel(self, snap):
        pygame.draw.rect(self.screen, PANEL, PANEL_RECT)
        pygame.draw.rect(self.screen, ACCENT,
                         (PANEL_RECT.x, PANEL_RECT.y, 4, PANEL_RECT.height))
        x = PANEL_RECT.x + 18
        right = PANEL_RECT.right - 16
        w = right - x
        y = PANEL_RECT.y + 14

        state = snap["state"]
        sc = STATE_COLOR[state]
        self.screen.blit(self.font_s.render("AUTONOMOUS SYSTEM", True, DIM), (x, y))
        y += 20
        pygame.draw.rect(self.screen, PANEL2, (x, y, w, 30), border_radius=3)
        pygame.draw.rect(self.screen, sc, (x, y, 5, 30))
        self.screen.blit(self.font_b.render(state, True, sc), (x + 14, y + 3))
        y += 40

        color, blink = snap["assi"]
        on = (pygame.time.get_ticks() // 250) % 2 == 0 if blink else True
        lit = ASSI_COLORS[color] if (on and color != "off") else ASSI_COLORS["off"]
        self.screen.blit(self.font_s.render("ASSI", True, DIM), (x, y + 4))
        for i in range(3):
            c = (x + 62 + i * 30, y + 10)
            pygame.draw.circle(self.screen, lit, c, 9)
            pygame.draw.circle(self.screen, (10, 11, 14), c, 9, 2)
        y += 36

        ami = dict(MISSIONS).get(snap["mission"], snap["mission"]).upper()
        self._row(x, y, "MISSION", ami); y += 22
        self._row(x, y, "PHASE", snap["phase"].upper()); y += 22
        self._row(x, y, "ASMS", "ON" if snap["asms"] else "OFF",
                  LIVE if snap["asms"] else DIM); y += 28

        self.screen.blit(self.font_s.render("SPEED", True, DIM), (x, y))
        spd = self.font_b.render(f"{snap['speed']:.1f}", True, TEXT)
        self.screen.blit(spd, (right - spd.get_width() - 30, y - 5))
        self.screen.blit(self.font_s.render("m/s", True, DIM), (right - 26, y + 3))
        y += 22
        self._seg_bar(x, y, w, 12, snap['speed'] / 16.0); y += 26

        deg = float(np.degrees(snap['steer']))
        self.screen.blit(self.font_s.render("STEER", True, DIM), (x, y))
        st = self.font_n.render(f"{deg:+.0f} deg", True, TEXT)
        self.screen.blit(st, (right - st.get_width(), y - 2)); y += 20
        self._steer_bar(x, y, w, 12, max(-1.0, min(1.0, deg / 28.0))); y += 26

        pygame.draw.line(self.screen, GRID, (x, y), (right, y)); y += 10
        if snap["mission"] in (AUTOCROSS, TRACKDRIVE, SKIDPAD):
            self._row(x, y, "LAP", str(snap['lap']))
        else:
            self._row(x, y, "LAP", f"{snap['lap']} / {snap['laps_required']}")
        y += 22
        cones = snap['map_left'].shape[0] + snap['map_right'].shape[0]
        self._row(x, y, "CONES", str(cones)); y += 22
        self._row(x, y, "TIME", f"{snap['sim_time']:.1f} s"); y += 22
        if snap["ebs_decel"] > 0.1:
            self._row(x, y, "EBS DECEL", f"{snap['ebs_decel']:.1f} m/s2",
                      LIVE if snap['ebs_decel'] > 10 else WARN); y += 22

        y += 6
        pygame.draw.line(self.screen, GRID, (x, y), (right, y)); y += 8
        self._wrap(x, y, snap["message"], w)

        for b in self.buttons:
            if b not in self.mission_buttons:
                b.draw(self.screen, self.font)

    def _seg_bar(self, x, y, w, h, frac, segs=16):
        frac = max(0.0, min(1.0, frac))
        gap = 2
        sw = (w - (segs - 1) * gap) / segs
        lit = int(round(frac * segs))
        for i in range(segs):
            rx = int(x + i * (sw + gap))
            if i < lit:
                col = ACCENT if i >= segs - 3 else LIVE
            else:
                col = PANEL2
            pygame.draw.rect(self.screen, col, (rx, y, int(sw) + 1, h))

    def _steer_bar(self, x, y, w, h, frac):
        pygame.draw.rect(self.screen, PANEL2, (x, y, w, h))
        cx = x + w // 2
        fill = int(abs(frac) * (w // 2))
        if frac >= 0:
            pygame.draw.rect(self.screen, ACCENT, (cx, y, fill, h))
        else:
            pygame.draw.rect(self.screen, ACCENT, (cx - fill, y, fill, h))
        pygame.draw.line(self.screen, (90, 96, 110), (cx, y - 1), (cx, y + h + 1), 1)

    def _row(self, x, y, label, value, vcolor=TEXT):
        self.screen.blit(self.font_s.render(label, True, DIM), (x, y + 2))
        v = self.font_n.render(value, True, vcolor)
        self.screen.blit(v, (PANEL_RECT.right - 16 - v.get_width(), y))

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
    app = App()
    if len(sys.argv) > 2 and sys.argv[1].lower() == "f1":
        app.sim.select_f1(sys.argv[2])
        app.view = View(app.sim.track)
        app.trail.clear()
    app.run()
