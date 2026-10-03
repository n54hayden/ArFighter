"""ShadowEnemy: a procedurally animated silhouette fighter with a simple state machine.

The body is a 2D skeleton driven by joint angles (forward kinematics). Angles are
in degrees measured from "straight down", with positive values rotating toward
the direction the enemy faces (90 = pointing forward, 180 = straight up). Knee
and elbow angles are relative to the parent bone.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import pygame

from . import config as C
from .geometry import Capsule, Circle, Vec, draw_shape

# Body proportions as fractions of total height H.
UPPER_ARM, FOREARM, THIGH, SHIN = 0.17, 0.16, 0.245, 0.245
TORSO, HEAD_OFFSET, HEAD_R, FOOT_LEN, FOOT_H = 0.30, 0.11, 0.065, 0.07, 0.02

Pose = Dict[str, float]

GUARD: Pose = dict(lean=8, fs=40, fe=100, rs=20, re=130, fh=18, fk=-24, rh=-16, rk=-6, dx=0.0)


def _pose(**overrides: float) -> Pose:
    p = dict(GUARD)
    p.update(overrides)
    return p


JAB_WIND = _pose(fs=55, fe=80, lean=4)
JAB_EXT = _pose(fs=86, fe=4, lean=14, dx=0.05)
CROSS_WIND = _pose(rs=35, re=110, fs=45, fe=95, lean=0)
CROSS_EXT = _pose(rs=86, re=4, fs=35, fe=110, lean=22, dx=0.08)
UPPER_WIND = _pose(lean=24, rs=-10, re=60, fh=35, fk=-60, rh=-5, rk=-45)
UPPER_EXT = _pose(lean=-4, rs=105, re=15, dx=0.06)
KICK_WIND = _pose(lean=-6, fh=85, fk=-115, rh=-8, rk=-4, fs=30, fe=110)
KICK_EXT = _pose(lean=-28, fh=140, fk=-8, rh=-10, rk=-2, fs=-20, fe=60, rs=10, re=100, dx=0.25)  # lunges in
SWEEP_WIND = _pose(lean=20, rh=60, rk=-115, fh=60, fk=-120, fs=20, fe=60, rs=-10, re=40)
SWEEP_EXT = _pose(lean=35, rh=65, rk=-130, fh=80, fk=-2, fs=-30, fe=40, rs=-50, re=30, dx=0.04)
STUN_POSE = _pose(lean=-22, fs=-30, fe=30, rs=-45, re=20, fh=25, fk=-35, dx=-0.04)


def lerp_pose(a: Pose, b: Pose, t: float) -> Pose:
    return {k: a[k] + (b[k] - a[k]) * t for k in a}


def _ease(t: float, kind: str) -> float:
    if kind == "out":  # snappy strikes
        return 1.0 - (1.0 - t) ** 3
    return t * t * (3.0 - 2.0 * t)


@dataclass(frozen=True)
class Phase:
    pose: Pose
    duration: float
    active: bool = False   # strike collider live during this phase...
    ease: str = "inout"
    active_from: float = 0.0  # ...once the limb has covered this fraction of its motion


@dataclass(frozen=True)
class AttackSpec:
    name: str
    label: str
    limb: str              # front_hand | rear_hand | front_foot
    target: str            # head | torso | legs (what the AI lines up with)
    damage: float
    block_chip: float      # fraction of damage that still goes through a block
    min_depth: float       # player depth ratio needed to be in range of this attack
    strike_radius: float   # fraction of H
    phases: Tuple[Phase, ...]
    unblockable: bool = False


def _attack(name, label, limb, target, damage, chip, min_depth, radius, wind, ext, t_wind, t_strike, t_hold, t_rec,
            live_from=0.5, unblockable=False) -> AttackSpec:
    return AttackSpec(name, label, limb, target, damage, chip, min_depth, radius, (
        Phase(wind, t_wind),
        # Only the later part of the strike motion is live, so a fist or foot swinging
        # up from the wind-up doesn't clip your body or lowered arms on the way.
        Phase(ext, t_strike, active=True, ease="out", active_from=live_from),
        Phase(ext, t_hold, active=True),
        Phase(GUARD, t_rec),
    ), unblockable)


ATTACKS: Dict[str, AttackSpec] = {a.name: a for a in (
    _attack("jab", "Jab", "front_hand", "head", 6, 0.0, 0.85, 0.045, JAB_WIND, JAB_EXT, 0.30, 0.09, 0.08, 0.28),
    _attack("cross", "Cross", "rear_hand", "head", 9, 0.0, 0.85, 0.045, CROSS_WIND, CROSS_EXT, 0.38, 0.10, 0.08, 0.34),
    _attack("uppercut", "Uppercut", "rear_hand", "head", 18, 0.25, 0.88, 0.05, UPPER_WIND, UPPER_EXT,
            0.65, 0.14, 0.10, 0.50, live_from=0.65),
    _attack("high_kick", "High Kick", "front_foot", "head", 14, 0.20, 0.80, 0.06, KICK_WIND, KICK_EXT,
            0.55, 0.16, 0.12, 0.55, live_from=0.75),
    _attack("low_sweep", "Low Sweep", "front_foot", "legs", 12, 1.0, 0.80, 0.065, SWEEP_WIND, SWEEP_EXT,
            0.60, 0.20, 0.15, 0.55, live_from=0.3, unblockable=True),
)}


class EnemyState(Enum):
    IDLE = auto()
    APPROACH = auto()
    ATTACK = auto()
    STUNNED = auto()
    DEAD = auto()


def _dir(angle_deg: float, facing: int) -> Vec:
    a = math.radians(angle_deg)
    return (facing * math.sin(a), math.cos(a))


def _add(p: Vec, d: Vec, s: float) -> Vec:
    return (p[0] + d[0] * s, p[1] + d[1] * s)


def _ipt(p: Vec) -> Tuple[int, int]:
    return int(p[0]), int(p[1])


def _fill_capsule(surf: pygame.Surface, color, a: Vec, b: Vec, r: float) -> None:
    r = max(1.0, r)
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    if length > 1e-3:
        nx, ny = -dy / length * r, dx / length * r
        pygame.draw.polygon(surf, color, [(a[0] + nx, a[1] + ny), (b[0] + nx, b[1] + ny),
                                          (b[0] - nx, b[1] - ny), (a[0] - nx, a[1] - ny)])
    pygame.draw.circle(surf, color, _ipt(a), int(r))
    pygame.draw.circle(surf, color, _ipt(b), int(r))


def _make_glow(radius: int, color) -> pygame.Surface:
    surf = pygame.Surface((radius * 2, radius * 2))
    for i in range(radius, 0, -2):
        k = (1.0 - i / radius) ** 1.6
        pygame.draw.circle(surf, tuple(int(c * k) for c in color), (radius, radius), i)
    return surf


class ShadowEnemy:
    def __init__(self, screen_w: int, screen_h: int):
        self.sw, self.sh = screen_w, screen_h
        self.font = pygame.font.Font(None, 40)
        self.debug_font = pygame.font.Font(None, 22)
        self.reset(ground_y=screen_h - 10, height=screen_h * 0.75, x=screen_w * 0.75)

    def reset(self, ground_y: float, height: float, x: float) -> None:
        self.ground_y = ground_y
        self.H = height
        self.x = x
        self.facing = -1
        self.hp = float(C.ENEMY_MAX_HP)
        self.state = EnemyState.IDLE
        self.state_t = 0.0
        self.idle_time = C.ENEMY_FIRST_ATTACK_DELAY
        self.pose: Pose = dict(GUARD)
        self.attack: Optional[AttackSpec] = None
        self.next_attack: Optional[AttackSpec] = None
        self.queued: Optional[AttackSpec] = None
        self.phase_i = 0
        self.phase_t = 0.0
        self.phase_progress = 0.0
        self.phase_from: Pose = dict(GUARD)
        self.attack_resolved = False
        self._strike_prev: Optional[Vec] = None
        self._strike_cur: Optional[Vec] = None
        self._was_live = False
        self.approach_t = 0.0
        self.stun_t = 0.0
        self.stun_immunity = 0.0
        self.vx = 0.0
        self.flash_t = 0.0
        self.walk_phase = 0.0
        self.anim_t = 0.0
        self._reach = {name: self._strike_offset(spec) for name, spec in ATTACKS.items()}
        glow_r = max(8, int(self.H * 0.09))
        self._glow = _make_glow(glow_r, (255, 50, 30))
        self._shadow = pygame.Surface((int(self.H * 0.5), int(self.H * 0.06)), pygame.SRCALPHA)
        pygame.draw.ellipse(self._shadow, (0, 0, 0, 110), self._shadow.get_rect())
        self.sk = self._skeleton(self.pose, self.x, self.facing)

    # --- kinematics ----------------------------------------------------------
    def _skeleton(self, p: Pose, x: float, facing: int) -> Dict[str, Vec]:
        H, f = self.H, facing

        def leg(hip_a: float, knee_a: float):
            knee = _add((0.0, 0.0), _dir(hip_a, f), THIGH * H)
            shin_a = hip_a + knee_a
            ankle = _add(knee, _dir(shin_a, f), SHIN * H)
            toe = _add(ankle, _dir(shin_a + 90, f), FOOT_LEN * H)
            return knee, ankle, toe

        fk, fa, ft = leg(p["fh"], p["fk"])
        rk, ra, rt = leg(p["rh"], p["rk"])
        drop = max(fa[1], ra[1])  # whichever foot is lowest stands on the ground
        hip = (x + f * p["dx"] * H, self.ground_y - drop - FOOT_H * H)
        r_hip = (hip[0] - f * 0.012 * H, hip[1])
        spine = _dir(180 - p["lean"], f)
        shoulder = _add(hip, spine, TORSO * H)
        r_shoulder = (shoulder[0] - f * 0.02 * H, shoulder[1])
        head = _add(shoulder, spine, HEAD_OFFSET * H)
        f_elbow = _add(shoulder, _dir(p["fs"], f), UPPER_ARM * H)
        f_hand = _add(f_elbow, _dir(p["fs"] + p["fe"], f), FOREARM * H)
        r_elbow = _add(r_shoulder, _dir(p["rs"], f), UPPER_ARM * H)
        r_hand = _add(r_elbow, _dir(p["rs"] + p["re"], f), FOREARM * H)
        off = lambda v: (hip[0] + v[0], hip[1] + v[1])  # noqa: E731
        roff = lambda v: (r_hip[0] + v[0], r_hip[1] + v[1])  # noqa: E731
        return {
            "hip": hip, "shoulder": shoulder, "head": head, "spine": spine,
            "f_elbow": f_elbow, "f_hand": f_hand,
            "r_hip": r_hip, "r_shoulder": r_shoulder, "r_elbow": r_elbow, "r_hand": r_hand,
            "f_knee": off(fk), "f_ankle": off(fa), "f_toe": off(ft),
            "r_knee": roff(rk), "r_ankle": roff(ra), "r_toe": roff(rt),
        }

    @staticmethod
    def _limb_end(sk: Dict[str, Vec], limb: str) -> Vec:
        if limb == "front_hand":
            return sk["f_hand"]
        if limb == "rear_hand":
            return sk["r_hand"]
        a, t = sk["f_ankle"], sk["f_toe"]
        return ((a[0] + t[0]) * 0.5, (a[1] + t[1]) * 0.5)

    def _strike_offset(self, spec: AttackSpec) -> float:
        """Furthest forward distance the striking limb reaches while its hitbox is live."""
        i = next(i for i, ph in enumerate(spec.phases) if ph.active)
        start, ph = spec.phases[i - 1].pose, spec.phases[i]
        best = 0.0
        for k in range(11):
            u = ph.active_from + (1.0 - ph.active_from) * k / 10
            sk = self._skeleton(lerp_pose(start, ph.pose, u), 0.0, 1)
            best = max(best, self._limb_end(sk, spec.limb)[0])
        return best

    # --- colliders -----------------------------------------------------------
    def hurtboxes(self) -> List:
        return [
            Capsule(self.sk["hip"], self.sk["shoulder"], 0.08 * self.H),
            Circle(self.sk["head"], HEAD_R * self.H * 1.1),
        ]

    def center(self) -> Vec:
        h, s = self.sk["hip"], self.sk["shoulder"]
        return ((h[0] + s[0]) * 0.5, (h[1] + s[1]) * 0.5)

    @property
    def phase(self) -> Optional[Phase]:
        if self.state is EnemyState.ATTACK and self.attack is not None:
            return self.attack.phases[self.phase_i]
        return None

    @property
    def in_windup(self) -> bool:
        return self.state is EnemyState.ATTACK and self.phase_i == 0

    def strike_collider(self) -> Optional[Capsule]:
        """The striking fist/foot swept over this frame's motion.

        Using the swept path instead of a single circle stops fast strikes from
        tunnelling through a guard between frames.
        """
        if self.attack_resolved or not self._strike_live() or self._strike_cur is None:
            return None
        return Capsule(self._strike_prev or self._strike_cur, self._strike_cur, self.attack.strike_radius * self.H)

    def _strike_live(self) -> bool:
        ph = self.phase
        return ph is not None and ph.active and self.phase_progress >= ph.active_from

    # --- combat events -------------------------------------------------------
    def take_hit(self, damage: float) -> str:
        """Returns 'ko', 'stun' or 'armor' (damaged but not stunned)."""
        if self.state is EnemyState.DEAD:
            return "armor"
        self.hp = max(0.0, self.hp - damage)
        self.flash_t = 0.1
        if self.hp <= 0:
            self.state = EnemyState.DEAD
            self.attack = None
            return "ko"
        if self.stun_immunity > 0 or self.state is EnemyState.STUNNED:
            return "armor"
        self.state = EnemyState.STUNNED
        self.attack = None
        self.stun_t = C.ENEMY_STUN_TIME
        self.vx = -self.facing * C.ENEMY_KNOCKBACK * self.H
        return "stun"

    def on_blocked(self) -> None:
        self.attack_resolved = True
        self.x -= self.facing * 0.04 * self.H

    # --- AI ------------------------------------------------------------------
    def _pick_attack(self) -> AttackSpec:
        names = list(C.ENEMY_ATTACK_WEIGHTS)
        weights = [C.ENEMY_ATTACK_WEIGHTS[n] for n in names]
        return ATTACKS[random.choices(names, weights)[0]]

    def _enter_idle(self, duration: float) -> None:
        self.state = EnemyState.IDLE
        self.state_t = 0.0
        self.idle_time = duration
        self.attack = None

    def _desired_x(self, player, spec: AttackSpec) -> float:
        tx = player.target_point(spec.target)[0]
        # Stand so the fully extended strike lands just in front of the target's centre,
        # which keeps our guard hand outside the player's guard until we strike.
        stand_off = self._reach[spec.name] + C.ENEMY_AIM_OFFSET * player.unit
        side = -self.facing  # stay on the side we're currently on
        d = tx + side * stand_off
        if not self.sw * 0.06 <= d <= self.sw * 0.94:
            d = tx - side * stand_off  # no room against the wall: go around to the other side
        return d

    def _start_attack(self, spec: AttackSpec) -> None:
        self.state = EnemyState.ATTACK
        self.attack = spec
        self.phase_i = 0
        self.phase_t = 0.0
        self.phase_progress = 0.0
        self.phase_from = dict(self.pose)
        self.attack_resolved = False
        self._strike_prev = self._strike_cur = None
        self._was_live = False

    def _update_attack(self, dt: float, player, paused: bool) -> None:
        spec = self.attack
        ph = spec.phases[self.phase_i]
        duration = ph.duration / C.ENEMY_ATTACK_SPEED
        self.phase_t += dt
        t = min(1.0, self.phase_t / duration)
        self.phase_progress = _ease(t, ph.ease)
        self.pose = lerp_pose(self.phase_from, ph.pose, self.phase_progress)

        if self.phase_i == 0 and not paused:  # keep tracking the player during the wind-up
            diff = self._desired_x(player, spec) - self.x
            max_step = C.ENEMY_WALK_SPEED * 0.5 * self.H * dt
            self.x += max(-max_step, min(max_step, diff))

        if self.phase_t >= duration:
            self.phase_t -= duration
            self.phase_progress = 0.0
            self.phase_from = ph.pose
            self.phase_i += 1
            if self.phase_i >= len(spec.phases):
                if spec.name == "jab" and random.random() < C.ENEMY_COMBO_CHANCE:
                    self.queued = ATTACKS["cross"]
                    self._enter_idle(0.05)
                else:
                    aggression = 1.0 + 0.6 * (1.0 - self.hp / C.ENEMY_MAX_HP)
                    self._enter_idle(random.uniform(C.ENEMY_IDLE_MIN, C.ENEMY_IDLE_MAX) / aggression)

    def _idle_pose(self) -> Pose:
        p = dict(GUARD)
        b = math.sin(self.anim_t * 2.6)
        p["lean"] += 2 * b
        p["fe"] += 4 * b
        p["re"] -= 3 * b
        return p

    def _walk_pose(self) -> Pose:
        s = math.sin(self.walk_phase)
        p = dict(GUARD)
        p["fh"] += 20 * s
        p["rh"] -= 20 * s
        p["fk"] -= 18 * max(0.0, -s)
        p["rk"] -= 18 * max(0.0, s)
        p["lean"] += 3
        return p

    def update(self, dt: float, player) -> None:
        self.anim_t += dt
        self.flash_t = max(0.0, self.flash_t - dt)
        self.stun_immunity = max(0.0, self.stun_immunity - dt)
        if self.state is EnemyState.DEAD:
            return

        paused = not (player.calibrated and player.tracked)
        target: Optional[Pose] = None

        if self.state is EnemyState.ATTACK:
            self._update_attack(dt, player, paused)
        elif self.state is EnemyState.STUNNED:
            self.x += self.vx * dt
            self.vx *= math.exp(-7.0 * dt)
            self.stun_t -= dt
            target = _pose(**{**STUN_POSE, "lean": STUN_POSE["lean"] + 6 * math.sin(self.anim_t * 18)})
            if self.stun_t <= 0:
                self._enter_idle(random.uniform(0.25, 0.5))
                self.stun_immunity = C.ENEMY_STUN_IMMUNITY
        elif paused:
            target = self._idle_pose()
        else:
            self.facing = 1 if player.com_x() >= self.x else -1
            if self.state is EnemyState.IDLE:
                self.state_t += dt
                target = self._idle_pose()
                if self.state_t >= self.idle_time:
                    self.next_attack = self.queued or self._pick_attack()
                    self.queued = None
                    self.state = EnemyState.APPROACH
                    self.approach_t = 0.0
            elif self.state is EnemyState.APPROACH:
                self.approach_t += dt
                diff = self._desired_x(player, self.next_attack) - self.x
                if abs(diff) <= 0.035 * self.H or self.approach_t > C.ENEMY_APPROACH_TIMEOUT:
                    self._start_attack(self.next_attack)
                else:
                    step = math.copysign(min(abs(diff), C.ENEMY_WALK_SPEED * self.H * dt), diff)
                    self.x += step
                    self.walk_phase += dt * 10.0 * (1 if step * self.facing > 0 else -1)
                    target = self._walk_pose()

        self.x = max(self.sw * 0.04, min(self.sw * 0.96, self.x))
        if target is not None:
            k = 1.0 - math.exp(-dt * C.ENEMY_POSE_BLEND)
            self.pose = lerp_pose(self.pose, target, k)
        self.sk = self._skeleton(self.pose, self.x, self.facing)
        if self.state is EnemyState.ATTACK:
            cur = self._limb_end(self.sk, self.attack.limb)
            # Sweep only between live frames, never back into the wind-up.
            live = self._strike_live()
            self._strike_prev = self._strike_cur if (live and self._was_live) else cur
            self._strike_cur = cur
            self._was_live = live

    # --- rendering -----------------------------------------------------------
    def draw(self, surf: pygame.Surface) -> None:
        if self.state is EnemyState.DEAD:
            return
        sk, H = self.sk, self.H
        surf.blit(self._shadow, self._shadow.get_rect(center=(int(self.x), int(self.ground_y))))

        f = self.facing
        sp = sk["spine"]
        n = (-sp[1], sp[0])
        torso_poly = [_add(sk["shoulder"], n, 0.075 * H), _add(sk["hip"], n, 0.055 * H),
                      _add(sk["hip"], n, -0.055 * H), _add(sk["shoulder"], n, -0.075 * H)]
        back = [
            ("cap", sk["r_shoulder"], sk["r_elbow"], 0.030), ("cap", sk["r_elbow"], sk["r_hand"], 0.026),
            ("circ", sk["r_hand"], 0.036),
            ("cap", sk["r_hip"], sk["r_knee"], 0.045), ("cap", sk["r_knee"], sk["r_ankle"], 0.036),
            ("cap", sk["r_ankle"], sk["r_toe"], 0.022),
        ]
        front = [
            ("poly", torso_poly), ("circ", sk["shoulder"], 0.07), ("circ", sk["hip"], 0.06),
            ("cap", sk["shoulder"], sk["head"], 0.03), ("circ", sk["head"], HEAD_R),
            ("cap", sk["hip"], sk["f_knee"], 0.048), ("cap", sk["f_knee"], sk["f_ankle"], 0.038),
            ("cap", sk["f_ankle"], sk["f_toe"], 0.022),
            ("cap", sk["shoulder"], sk["f_elbow"], 0.032), ("cap", sk["f_elbow"], sk["f_hand"], 0.028),
            ("circ", sk["f_hand"], 0.038),
        ]

        rim = max(2.0, 0.006 * H)
        flashing = self.flash_t > 0

        def paint(parts, color, grow):
            for part in parts:
                if part[0] == "cap":
                    _fill_capsule(surf, color, part[1], part[2], part[3] * H + grow)
                elif part[0] == "circ":
                    pygame.draw.circle(surf, color, _ipt(part[1]), int(part[2] * H + grow))
                else:
                    pygame.draw.polygon(surf, color, part[1])
                    if grow:
                        pygame.draw.polygon(surf, color, part[1], int(grow * 2))

        paint(back + front, C.ENEMY_RIM, rim)
        paint(back, C.ENEMY_FLASH if flashing else C.ENEMY_BACK, 0)
        paint(front, C.ENEMY_FLASH if flashing else C.ENEMY_BODY, 0)

        # Eye: white normally, red while attacking.
        angry = self.state is EnemyState.ATTACK
        eye = _add(sk["head"], (f, 0), 0.03 * H)
        eye = (eye[0], eye[1] - 0.01 * H)
        ew, eh = max(4, int(0.024 * H)), max(2, int(0.009 * H))
        pygame.draw.ellipse(surf, C.ENEMY_EYES_ANGRY if angry else C.ENEMY_EYES,
                            pygame.Rect(int(eye[0] - ew / 2), int(eye[1] - eh / 2), ew, eh))

        # Telegraph: glowing limb during the wind-up.
        if self.in_windup:
            progress = min(1.0, self.phase_t / (self.attack.phases[0].duration / C.ENEMY_ATTACK_SPEED))
            glow = self._glow.copy()
            glow.fill((int(255 * (0.4 + 0.6 * progress)),) * 3, special_flags=pygame.BLEND_MULT)
            p = self._limb_end(sk, self.attack.limb)
            surf.blit(glow, glow.get_rect(center=_ipt(p)), special_flags=pygame.BLEND_ADD)
            if self.attack.unblockable and int(self.anim_t * 8) % 2 == 0:
                label = self.font.render("JUMP!", True, (255, 220, 80))
                surf.blit(label, label.get_rect(center=(int(self.x), int(self.ground_y - 0.55 * H))))

    def draw_debug(self, surf: pygame.Surface) -> None:
        if self.state is EnemyState.DEAD:
            return
        for hb in self.hurtboxes():
            draw_shape(surf, hb, (255, 80, 255))
        strike = self.strike_collider()
        if strike:
            draw_shape(surf, strike, (255, 40, 40), 0)
        elif self.in_windup:
            p = self._limb_end(self.sk, self.attack.limb)
            draw_shape(surf, Circle(p, self.attack.strike_radius * self.H), (255, 150, 40))
        label = self.state.name
        if self.state is EnemyState.ATTACK:
            label += f" {self.attack.label} [{('windup', 'strike', 'hold', 'recover')[self.phase_i]}]"
        elif self.state is EnemyState.APPROACH and self.next_attack:
            label += f" -> {self.next_attack.label}"
        text = self.debug_font.render(label, True, (255, 200, 255))
        head = self.sk["head"]
        surf.blit(text, text.get_rect(midbottom=(int(head[0]), int(head[1] - 0.09 * self.H))))
