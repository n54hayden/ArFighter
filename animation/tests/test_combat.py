#!/usr/bin/env python3
"""Hit-registration tests: the real Game code driven by synthetic pose landmarks (no camera).

    python tests/test_combat.py

Checks, using the hit log as the record of what registered:
  1. repeated punches: every punch that reaches the body registers exactly once
  2. very fast punches (the fist jumps past the body between camera frames) still register
  3. a fist held inside the body registers once, not every frame
  4. strikes from out of range never register
  5. boss attacks against a standing player: every attack resolves exactly once (hit, block,
     out of range or miss) and nothing is logged twice
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pygame  # noqa: E402

import ar_fighter  # noqa: E402
from fighter import config as C  # noqa: E402
from fighter.camera import CameraFrame  # noqa: E402
from fighter.enemy import EnemyState  # noqa: E402

W, H = C.SCREEN_W, C.SCREEN_H
# Standing pose (normalised image coords): x offset from the body centre, y.
BASE = {0: (0.0, 0.25), 11: (-0.045, 0.35), 12: (0.045, 0.35), 13: (-0.06, 0.45), 14: (0.06, 0.45),
        15: (-0.05, 0.40), 16: (0.05, 0.40), 19: (-0.05, 0.38), 20: (0.05, 0.38), 23: (-0.03, 0.56),
        24: (0.03, 0.56), 25: (-0.035, 0.72), 26: (0.035, 0.72), 27: (-0.035, 0.87), 28: (0.035, 0.87),
        31: (-0.02, 0.89), 32: (0.05, 0.89)}


class FakeCam:
    error = None
    camera_fps = 30.0
    notice = None

    def __init__(self):
        self.frame = None

    def start(self): pass
    def stop(self): pass
    def join(self, timeout=None): pass
    def latest(self): return self.frame


class Rig:
    """Owns a Game with a fake camera and steps it at 60 FPS with 30 FPS pose frames."""

    def __init__(self):
        pygame.init()
        orig = ar_fighter.CameraThread
        ar_fighter.CameraThread = lambda *a, **k: FakeCam()
        self.g = ar_fighter.Game(0, 1, False)
        ar_fighter.CameraThread = orig
        self.g.music.available = False
        self.g.sfx.available = False
        self.g.hitlog._folder = None  # don't write CSVs from tests
        self.cam = self.g.camera
        self.t = time.perf_counter()
        self.fid = 0
        self.cx, self.r, self.fist = 0.42, 1.0, 0.0

    def frame(self):
        lm = np.zeros((33, 4), np.float32)
        for i, (dx, y) in BASE.items():
            lm[i] = (self.cx + dx * self.r, 0.5 + (y - 0.5) * self.r, 0, 0.99)
        for i in (16, 20):  # right fist (screen right, toward the boss)
            lm[i, 0] += self.fist / W
        img = np.full((H, W, 3), 60, np.uint8)
        return CameraFrame(self.fid, img, lm, self.t, 20.0, np.full((360, 640, 3), 90, np.uint8), None, None, None)

    def step(self, n=1):
        g = self.g
        for _ in range(n):
            self.t += 1 / 60
            if self.fid % 2 == 0:
                self.cam.frame = self.frame()
            self.fid += 1
            pygame.event.pump()
            g._poll_camera(self.t)
            g._update(1 / 60, self.t)
            g.dt = 1 / 60
            g._draw()

    def start_fight(self, boss=0):
        g = self.g
        g.boss_index = boss
        g._start_countdown()
        self.step(int(6.5 * 60))  # countdown + calibration
        assert g.mode == "fight", g.mode

    def freeze_boss(self, gap_px=150):
        """Boss stands still (no AI, no dodges / blocks), body centred gap_px right of the player."""
        e = self.g.enemy
        e.boss = dict(e.boss, dodge_chance=0.0, kick_block_chance=0.0)
        e.hp = 1e6
        e.state = EnemyState.IDLE
        e.idle_time = 1e9
        e.wx = e.floor.world_x(self.cx * W + gap_px, 1.0)
        e.vx = e.mvx = 0.0
        e.stun_immunity = 0.0


def ramp(rig, r, frames=8):
    """Step toward / away from the camera over `frames` camera frames, like a real step."""
    r0 = rig.r
    for i in range(frames):
        rig.r = r0 + (r - r0) * (i + 1) / frames
        rig.step(2)


def player_entries(g):
    return [e for e in g.hitlog.events if e.attacker == "PLAYER"]


def punch(rig, reach_px, out_frames, hold_frames=2, back_frames=4):
    """One punch, in camera frames (2 game frames each)."""
    for i in range(out_frames):
        rig.fist = reach_px * (i + 1) / out_frames
        rig.step(2)
    rig.step(2 * hold_frames)
    for i in range(back_frames):
        rig.fist = reach_px * (1 - (i + 1) / back_frames)
        rig.step(2)
    rig.fist = 0.0
    rig.step(8)


def test_repeated(rig, n=25):
    g = rig.g
    before = len(player_entries(g))
    for _ in range(n):
        rig.freeze_boss()
        punch(rig, 175, out_frames=3)
    got = len(player_entries(g)) - before
    return got == n, f"{n} punches -> {got} registered"


def test_fast(rig, n=15):
    """The fist covers its whole reach in ONE camera frame and goes past the body."""
    g = rig.g
    before = len(player_entries(g))
    for _ in range(n):
        rig.freeze_boss(gap_px=110)
        punch(rig, 260, out_frames=1, hold_frames=0, back_frames=2)
    got = len(player_entries(g)) - before
    return got == n, f"{n} one-frame punches (fist jumps 260 px past the body) -> {got} registered"


def test_held(rig):
    g = rig.g
    rig.freeze_boss()
    before = len(player_entries(g))
    for i in range(3):
        rig.fist = 175 * (i + 1) / 3
        rig.step(2)
    rig.step(60)  # hold the fist inside the body for a second
    rig.fist = 0.0
    rig.step(10)
    new = player_entries(g)[before:]
    got = len(new)
    detail = "" if got == 1 else "  " + "; ".join(f"#{e.strike_id} {e.attack} {e.result} @{e.t}s" for e in new)
    return got == 1, f"fist held in the body for 1 s -> {got} registered{detail}"


def test_out_of_range(rig, n=10):
    """Step back quickly and punch at once: you're out of reach before the shadow follows you."""
    g = rig.g
    before = len(player_entries(g))
    for _ in range(n):
        ramp(rig, 1.0)
        rig.step(120)           # level with the shadow again
        rig.freeze_boss()
        ramp(rig, 0.7, 6)       # a quick step back (0.2 s)...
        punch(rig, 175, out_frames=3)   # ...and punch straight away
    ramp(rig, 1.0)
    rig.step(60)
    got = len(player_entries(g)) - before
    return got == 0, f"{n} punches right after a quick step back -> {got} registered"


def test_size_match(rig):
    """The shadow matches the player's on-screen height and stays put sideways while doing so."""
    g = rig.g
    e = g.enemy
    rig.freeze_boss()
    out = []
    for r in (1.0, 0.8, 1.15):
        x0 = e.x
        ramp(rig, r)
        rig.step(180)
        player_h = g.floor.H0 * g.player.depth_ratio
        out.append((r, e.H / player_h, abs(e.x - x0)))
    ramp(rig, 1.0)
    rig.step(120)
    ok = all(abs(ratio - 1) < 0.07 and dx < 2 for _, ratio, dx in out)
    return ok, "  ".join(f"depth {r:.2f}: shadow/player height {ratio:.2f}, x moved {dx:.0f}px" for r, ratio, dx in out)


def test_boss_attacks(rig, boss, seconds=60):
    g = rig.g
    rig.start_fight(boss)
    e = g.enemy
    e.hp = 1e6
    rig.cx, rig.fist = 0.42, 0.0
    dup_before = g.hitlog.duplicates
    started = {}  # attack id -> spec, as each attack begins
    for _ in range(seconds * 60):
        rig.step(1)
        g.player.hp = 100.0
        if e.state is EnemyState.ATTACK and e.attack is not None:
            started.setdefault(e.attack_id, e.attack)
        if g.mode != "fight":
            break
    if e.state is EnemyState.ATTACK:
        started.pop(e.attack_id, None)  # still in progress when the test stopped
    melee = {i for i, s in started.items() if not s.ranged and not s.spell}
    entries = [x for x in g.hitlog.events if x.attacker == e.name and x.strike_id.startswith("A")
               and int(x.strike_id[1:]) in started]
    ids = [int(x.strike_id[1:]) for x in entries]
    dup = {i for i in ids if ids.count(i) > 1}
    missing = melee - set(ids)
    results = {}
    for x in entries:
        results[x.result] = results.get(x.result, 0) + 1
    others = len(started) - len(melee)
    ok = not dup and not missing and g.hitlog.duplicates == dup_before
    return ok, (f"{C.BOSSES[boss]['name']}: {len(melee)} melee attacks -> {len(entries)} entries {results}; "
                f"duplicates {len(dup)}, unresolved {len(missing)}; {others} ranged/spell (logged via projectile)")


def test_interrupted(rig, n=20):
    """Hit the boss in the middle of his strike window (the crash seen in play): nothing may break
    and the interrupted attack must not land afterwards."""
    g = rig.g
    rig.start_fight(0)
    e = g.enemy
    e.hp = 1e6
    done = 0
    for _ in range(60 * 60):
        rig.step(1)
        g.player.hp = 100.0
        if e.state is EnemyState.ATTACK and e.phase_i == 1:
            e.take_hit(5, "body", -e.facing, 0.6)        # interrupt mid-strike
            if e.strike_colliders():
                return False, "strike still live after the boss was hit"
            rig.step(2)
            done += 1
            if done >= n:
                break
    return done >= n, f"{done} strikes interrupted mid-window, no errors, nothing left live"


def main():
    rig = Rig()
    rig.start_fight(0)
    results = [("repeated punches", *test_repeated(rig)),
               ("fast punches", *test_fast(rig)),
               ("held fist", *test_held(rig)),
               ("out of range", *test_out_of_range(rig)),
               ("size match", *test_size_match(rig)),
               ("interrupted", *test_interrupted(rig))]
    for boss in range(len(C.BOSSES)):
        results.append((f"boss attacks {boss + 1}", *test_boss_attacks(rig, boss)))
    print()
    for name, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:18s} {detail}")
    print(f"\nhit log: {len(rig.g.hitlog.events)} events, duplicates rejected by the log: {rig.g.hitlog.duplicates}")
    pygame.quit()
    sys.exit(0 if all(ok for _, ok, _ in results) else 1)


if __name__ == "__main__":
    main()
