"""PlayerTracker: turns MediaPipe landmarks into colliders, punch speed, depth and jumps."""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pygame

from . import config as C
from .geometry import Capsule, Circle, Vec, draw_shape

LANDMARKS = {
    "nose": 0,
    "l_shoulder": 11, "r_shoulder": 12,
    "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16,
    "l_index": 19, "r_index": 20,
    "l_hip": 23, "r_hip": 24,
    "l_knee": 25, "r_knee": 26,
    "l_ankle": 27, "r_ankle": 28,
    "l_foot": 31, "r_foot": 32,
}
CORE = ("l_shoulder", "r_shoulder", "l_hip", "r_hip")
SKELETON = [
    ("l_shoulder", "r_shoulder"), ("l_hip", "r_hip"),
    ("l_shoulder", "l_hip"), ("r_shoulder", "r_hip"),
    ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"), ("l_wrist", "l_index"),
    ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"), ("r_wrist", "r_index"),
    ("l_hip", "l_knee"), ("l_knee", "l_ankle"),
    ("r_hip", "r_knee"), ("r_knee", "r_ankle"),
    ("l_ankle", "l_foot"), ("r_ankle", "r_foot"),
]
SIDES = ("l", "r")


def _mid(a: Vec, b: Vec) -> Vec:
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


def _dist(a: Vec, b: Vec) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class PlayerTracker:
    def __init__(self, screen_w: int, screen_h: int):
        self.sw, self.sh = screen_w, screen_h
        self.reset_calibration()
        self.reset_fight()

    # --- lifecycle -----------------------------------------------------------
    def reset_fight(self) -> None:
        self.hp = float(C.PLAYER_MAX_HP)
        self.punch_cooldown = {s: 0.0 for s in SIDES}
        self.kick_cooldown = {s: 0.0 for s in SIDES}

    def reset_calibration(self) -> None:
        self.pts: Dict[str, Vec] = {}             # smoothed screen-space landmarks
        self.fist_vel: Dict[str, Vec] = {s: (0.0, 0.0) for s in SIDES}
        self.foot_vel: Dict[str, Vec] = {s: (0.0, 0.0) for s in SIDES}
        self._prev_limb_raw: Dict[tuple, Vec] = {}
        self._prev_capture_t: Optional[float] = None
        self.last_seen = -1e9
        self._now = 0.0

        self.calibrated = False
        self.calibration_enabled = False          # the game turns this on once you're in position
        self.calib_time = 0.0
        self._calib_samples: List[dict] = []
        self.base_torso = 1.0
        self.base_shoulder = 1.0
        self._head_ref = self.sh * 0.3
        self.shield_enabled = False                # only for bosses that hand you a shield
        self.shield_side = "l"                    # landmark side of your real left arm (set in calibration)
        self.depth_ratio = 1.0                    # current size / calibrated size
        self.unit = 100.0                         # current torso length in pixels
        self.ground_y = float(self.sh)            # floor line for the enemy
        self.body_height = self.sh * 0.75
        self.full_body_visible = False

        self._ground_ref: Dict[str, float] = {}   # calibrated y of 'ankle' / 'hip'
        self._ground_offset: Dict[str, float] = {}
        self.expected_ground: Optional[float] = None
        self.rise = 0.0                           # torso units above the floor baseline
        self.airborne = False
        self._land_time = -1e9

    # --- per camera frame ----------------------------------------------------
    def ingest(self, landmarks: Optional[np.ndarray], capture_time: float, now: float) -> None:
        if landmarks is None:
            self._prev_limb_raw.clear()
            return

        raw: Dict[str, Vec] = {}
        for name, idx in LANDMARKS.items():
            x, y, _z, vis = landmarks[idx]
            if vis >= C.LANDMARK_VISIBILITY:
                raw[name] = (float(x) * self.sw, float(y) * self.sh)
        if not all(k in raw for k in CORE):
            self._prev_limb_raw.clear()
            return

        dt_cap = 0.0 if self._prev_capture_t is None else capture_time - self._prev_capture_t
        self._prev_capture_t = capture_time

        a = C.LANDMARK_SMOOTHING
        smoothed = {}
        for name, p in raw.items():
            old = self.pts.get(name)
            smoothed[name] = p if old is None else (old[0] + (p[0] - old[0]) * a, old[1] + (p[1] - old[1]) * a)
        self.pts = smoothed
        self.last_seen = now

        # Body scale -> depth. Take the max of two cues so turning sideways
        # (shoulders shrink) or leaning toward the camera (torso shrinks) doesn't
        # read as stepping back.
        torso = _dist(_mid(raw["l_shoulder"], raw["r_shoulder"]), _mid(raw["l_hip"], raw["r_hip"]))
        shoulder = _dist(raw["l_shoulder"], raw["r_shoulder"])
        if self.calibrated:
            ratio = max(torso / self.base_torso, shoulder / self.base_shoulder)
            self.depth_ratio += (ratio - self.depth_ratio) * C.DEPTH_SMOOTHING
            self.unit = self.base_torso * self.depth_ratio
        else:
            self.unit = torso if self.unit <= 0 else self.unit + (torso - self.unit) * 0.3

        # Fist and foot velocity from raw (unsmoothed) positions so smoothing doesn't
        # dull punches and kicks.
        for kind, point_fn, vel in (("fist", self._fist_point, self.fist_vel),
                                    ("foot", self._foot_point, self.foot_vel)):
            for side in SIDES:
                p = point_fn(raw, side)
                prev = self._prev_limb_raw.get((kind, side))
                if p is not None and prev is not None and 0.0 < dt_cap < 0.25:
                    vel[side] = ((p[0] - prev[0]) / dt_cap, (p[1] - prev[1]) / dt_cap)
                else:
                    vel[side] = (0.0, 0.0)
                if p is None:
                    self._prev_limb_raw.pop((kind, side), None)
                else:
                    self._prev_limb_raw[(kind, side)] = p

        if self.calibrated:
            self._update_jump(raw, dt_cap, now)
        elif self.calibration_enabled:
            self._collect_calibration_sample(raw, torso, shoulder)

    # --- per game frame ------------------------------------------------------
    def update(self, dt: float, now: float) -> None:
        self._now = now
        for side in SIDES:
            self.punch_cooldown[side] = max(0.0, self.punch_cooldown[side] - dt)
            self.kick_cooldown[side] = max(0.0, self.kick_cooldown[side] - dt)
        if not self.calibrated and self.calibration_enabled and self.tracked:
            self.calib_time += dt
            if self.calib_time >= C.CALIBRATION_SECONDS and len(self._calib_samples) >= 8:
                self._finish_calibration()

    @property
    def tracked(self) -> bool:
        return self._now - self.last_seen < C.TRACKING_LOST_TIMEOUT

    @property
    def calibration_progress(self) -> float:
        return min(1.0, self.calib_time / C.CALIBRATION_SECONDS)

    # --- calibration ---------------------------------------------------------
    def _collect_calibration_sample(self, raw: Dict[str, Vec], torso: float, shoulder: float) -> None:
        ankles = [raw[k][1] for k in ("l_ankle", "r_ankle") if k in raw]
        self._calib_samples.append({
            "torso": torso,
            "shoulder": shoulder,
            "hip": _mid(raw["l_hip"], raw["r_hip"])[1],
            "ankle": max(ankles) if ankles else None,
            "nose": raw["nose"][1] if "nose" in raw else None,
            "lr_dx": raw["l_shoulder"][0] - raw["r_shoulder"][0],
        })

    def _finish_calibration(self) -> None:
        s = self._calib_samples
        med = lambda key: float(np.median([x[key] for x in s if x[key] is not None]))  # noqa: E731
        self.base_torso = med("torso")
        self.base_shoulder = med("shoulder")
        hip = med("hip")
        self._ground_ref = {"hip": hip}
        ankle_count = sum(1 for x in s if x["ankle"] is not None)
        self.full_body_visible = ankle_count >= len(s) * 0.6
        if self.full_body_visible:
            ankle = med("ankle")
            self._ground_ref["ankle"] = ankle
            self.ground_y = ankle + 0.16 * self.base_torso       # sole sits a little below the ankle
        else:
            self.ground_y = hip + 1.65 * self.base_torso          # floor is off-screen; estimate it
        nose_count = sum(1 for x in s if x["nose"] is not None)
        if nose_count:
            nose = med("nose")
            self.body_height = (self.ground_y - nose) * 1.1
        else:
            nose = hip - 1.55 * self.base_torso
            self.body_height = self.base_torso * 3.4
        self._head_ref = nose - 0.1 * self.base_torso  # same point as head().center
        self.body_height = max(self.sh * 0.3, min(self.sh * 1.2, self.body_height))
        self._ground_offset = {k: 0.0 for k in self._ground_ref}
        # The camera image is mirrored, so your real left arm is the one on the left of the
        # screen. Decide it from where the shoulders were, not from MediaPipe's labels.
        self.shield_side = "l" if med("lr_dx") < 0 else "r"
        self.depth_ratio = 1.0
        self.unit = self.base_torso
        self.calibrated = True

    # --- jump detection ------------------------------------------------------
    def _update_jump(self, raw: Dict[str, Vec], dt: float, now: float) -> None:
        """Detect jumps by comparing the floor-contact landmarks to a depth-compensated baseline.

        Stepping back makes your feet move up the image as well, which would look
        like a jump. With a pinhole camera, a point at fixed world height sits at
        horizon + (y0 - horizon) * scale, so the expected floor y is predicted from
        the current depth ratio, and a slow drift term absorbs horizon error.
        """
        signals: Dict[str, float] = {"hip": _mid(raw["l_hip"], raw["r_hip"])[1]}
        ankles = [raw[k][1] for k in ("l_ankle", "r_ankle") if k in raw]
        if ankles:
            signals["ankle"] = max(ankles)  # the lower foot, so lifting one knee isn't a jump

        horizon = self.sh * C.HORIZON_Y_FRAC
        rise = None
        for key in ("ankle", "hip"):  # ankles take priority when visible
            y = signals.get(key)
            ref = self._ground_ref.get(key)
            if y is None or ref is None:
                continue
            expected = horizon + (ref - horizon) * self.depth_ratio + self._ground_offset[key]
            if rise is None:
                rise = (expected - y) / self.unit
                self.expected_ground = expected
            if not self.airborne and dt > 0:
                k = 1.0 - math.exp(-dt / C.GROUND_ADAPT_TAU)
                self._ground_offset[key] += (y - expected) * k
        if rise is None:
            return
        self.rise = rise
        if not self.airborne and rise > C.JUMP_THRESHOLD:
            self.airborne = True
        elif self.airborne and rise < C.JUMP_RELEASE:
            self.airborne = False
            self._land_time = now

    @property
    def is_jumping(self) -> bool:
        return self.airborne or (self._now - self._land_time) < C.JUMP_GRACE

    def standing_head_y(self) -> float:
        """Where your head is when standing upright at your current distance (stars aim here)."""
        horizon = self.sh * C.HORIZON_Y_FRAC
        return horizon + (self._head_ref - horizon) * self.depth_ratio

    def in_range(self, min_depth: float) -> bool:
        return self.depth_ratio >= min_depth

    # --- body queries --------------------------------------------------------
    def _fist_point(self, pts: Dict[str, Vec], side: str) -> Optional[Vec]:
        wrist = pts.get(f"{side}_wrist")
        if wrist is None:
            return None
        index = pts.get(f"{side}_index")
        if index is not None:
            return _mid(wrist, index)
        elbow = pts.get(f"{side}_elbow")
        if elbow is not None:  # extrapolate a little past the wrist
            return (wrist[0] + (wrist[0] - elbow[0]) * 0.15, wrist[1] + (wrist[1] - elbow[1]) * 0.15)
        return wrist

    def _foot_point(self, pts: Dict[str, Vec], side: str) -> Optional[Vec]:
        ankle = pts.get(f"{side}_ankle")
        if ankle is None:
            return None
        toe = pts.get(f"{side}_foot")
        return _mid(ankle, toe) if toe is not None else ankle

    def foot_speed(self, side: str) -> float:
        vx, vy = self.foot_vel[side]
        return math.hypot(vx, vy) / max(self.unit, 1.0)

    def fist_speed(self, side: str) -> float:
        vx, vy = self.fist_vel[side]
        return math.hypot(vx, vy) / max(self.unit, 1.0)

    def shoulder_mid(self) -> Vec:
        return _mid(self.pts["l_shoulder"], self.pts["r_shoulder"])

    def hip_mid(self) -> Vec:
        return _mid(self.pts["l_hip"], self.pts["r_hip"])

    def com_x(self) -> float:
        if not all(k in self.pts for k in CORE):
            return self.sw * 0.5
        return (self.shoulder_mid()[0] + self.hip_mid()[0]) * 0.5

    def target_point(self, target: str) -> Vec:
        """Where the enemy aims: 'head', 'torso' or 'legs'."""
        head = self.head()
        if target == "head" and head is not None:
            return head.center
        sh, hip = self.shoulder_mid(), self.hip_mid()
        if target == "legs":
            for a, b in (("l_knee", "r_knee"), ("l_ankle", "r_ankle")):
                if a in self.pts and b in self.pts:
                    return _mid(self.pts[a], self.pts[b])
            return (hip[0], hip[1] + 1.2 * self.unit)
        return (sh[0] + (hip[0] - sh[0]) * 0.25, sh[1] + (hip[1] - sh[1]) * 0.25)

    # --- colliders -----------------------------------------------------------
    def head(self) -> Optional[Circle]:
        if "nose" in self.pts:
            nx, ny = self.pts["nose"]
        elif all(k in self.pts for k in CORE):
            nx, ny = self.shoulder_mid()
            ny -= 0.55 * self.unit
        else:
            return None
        return Circle((nx, ny - 0.1 * self.unit), C.HEAD_RADIUS * self.unit)

    def torso(self) -> Optional[Capsule]:
        if not all(k in self.pts for k in CORE):
            return None
        return Capsule(self.shoulder_mid(), self.hip_mid(), C.TORSO_RADIUS * self.unit)

    def forearms(self) -> List[Capsule]:
        out = []
        for side in SIDES:
            e, w = self.pts.get(f"{side}_elbow"), self.pts.get(f"{side}_wrist")
            if e is not None and w is not None:
                out.append(Capsule(e, w, C.FOREARM_RADIUS * self.unit))
        return out

    def fists(self) -> Dict[str, Circle]:
        out = {}
        for side in SIDES:
            p = self._fist_point(self.pts, side)
            if p is not None:
                out[side] = Circle(p, C.FIST_RADIUS * self.unit)
        return out

    def feet(self) -> Dict[str, Circle]:
        out = {}
        for side in SIDES:
            p = self._foot_point(self.pts, side)
            if p is not None:
                out[side] = Circle(p, C.FOOT_RADIUS * self.unit)
        return out

    def shield(self) -> Optional[Circle]:
        """Round shield strapped to your left forearm (when this fight gives you one)."""
        if not self.shield_enabled:
            return None
        e, w = self.pts.get(f"{self.shield_side}_elbow"), self.pts.get(f"{self.shield_side}_wrist")
        if e is None or w is None:
            return None
        centre = (e[0] + (w[0] - e[0]) * 0.9, e[1] + (w[1] - e[1]) * 0.9)  # gripped near the hand
        return Circle(centre, C.SHIELD_RADIUS * self.unit)

    def draw_shield(self, surf: pygame.Surface) -> None:
        sh = self.shield()
        if sh is None:
            return
        c, r = (int(sh.center[0]), int(sh.center[1])), int(sh.radius)
        pygame.draw.circle(surf, (25, 18, 8), (c[0] + 4, c[1] + 4), r)               # drop shadow
        pygame.draw.circle(surf, C.SHIELD_COLOR, c, r)
        pygame.draw.circle(surf, C.SHIELD_RIM_COLOR, c, r, max(3, r // 9))
        pygame.draw.circle(surf, C.SHIELD_RIM_COLOR, c, int(r * 0.62), max(2, r // 18))
        pygame.draw.circle(surf, C.SHIELD_RIM_COLOR, c, max(4, r // 5))                # central boss
        pygame.draw.circle(surf, (110, 75, 25), c, max(2, r // 9))

    def legs(self) -> List[Capsule]:
        r = C.LEG_RADIUS * self.unit
        out = []
        for side in SIDES:
            hip, knee, ankle = (self.pts.get(f"{side}_{j}") for j in ("hip", "knee", "ankle"))
            if hip is not None and knee is not None:
                out.append(Capsule(hip, knee, r))
                if ankle is not None:
                    out.append(Capsule(knee, ankle, r))
        if not out and all(k in self.pts for k in CORE):
            hip = self.hip_mid()
            out.append(Capsule(hip, (hip[0], max(hip[1], self.ground_y - 0.1 * self.unit)), r * 1.5))
        return out

    # --- debug ---------------------------------------------------------------
    def draw_debug(self, surf: pygame.Surface, font: pygame.font.Font, colliders: bool = True) -> None:
        for a, b in SKELETON:
            if a in self.pts and b in self.pts:
                pygame.draw.line(surf, (240, 240, 240), self.pts[a], self.pts[b], 2)
        for p in self.pts.values():
            pygame.draw.circle(surf, (255, 255, 255), (int(p[0]), int(p[1])), 4)
        if not colliders:
            return

        head, torso = self.head(), self.torso()
        if head:
            draw_shape(surf, head, (60, 230, 90))
        if torso:
            draw_shape(surf, torso, (60, 230, 90))
        for leg in self.legs():
            draw_shape(surf, leg, (255, 160, 40))
        for fa in self.forearms():
            draw_shape(surf, fa, (60, 200, 255), 3)
        sh = self.shield()
        if sh:
            draw_shape(surf, sh, (255, 210, 90), 2)
        for side, fist in self.fists().items():
            fast = self.fist_speed(side) >= C.PUNCH_SPEED_THRESHOLD
            draw_shape(surf, fist, (255, 60, 60) if fast else (255, 230, 60), 0 if fast else 2)
        for side, foot in self.feet().items():
            fast = self.foot_speed(side) >= C.KICK_SPEED_THRESHOLD
            draw_shape(surf, foot, (255, 60, 60) if fast else (255, 120, 220), 0 if fast else 2)
        if self.expected_ground is not None:
            y = int(self.expected_ground)
            pygame.draw.line(surf, (255, 160, 40), (0, y), (self.sw, y), 1)

        if torso:
            x, y = int(torso.a[0] + torso.radius + 12), int(torso.a[1])
            lines = [
                f"depth {self.depth_ratio:.2f}",
                f"rise {self.rise:+.2f}{'  AIR' if self.airborne else ''}",
                f"fist L {self.fist_speed('l'):.1f}  R {self.fist_speed('r'):.1f}",
                f"foot L {self.foot_speed('l'):.1f}  R {self.foot_speed('r'):.1f}",
            ]
            for i, text in enumerate(lines):
                surf.blit(font.render(text, True, (255, 255, 255)), (x, y + i * 18))
