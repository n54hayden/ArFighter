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
from . import arcade
from .effects import render_outlined
from .geometry import Capsule, Circle, Vec, draw_shape

# Body proportions as fractions of total height H.
UPPER_ARM, FOREARM, THIGH, SHIN = 0.17, 0.16, 0.245, 0.245
TORSO, HEAD_OFFSET, HEAD_R, FOOT_LEN, FOOT_H = 0.30, 0.11, 0.065, 0.07, 0.02
STAFF_LEN, STAFF_R, STAFF_HAND_GAP = 0.95, 0.016, 0.09
# Khopesh: a short straight part from the hand, then a hooked curve.
BLADE_STRAIGHT, BLADE_CURVE_SEGS, BLADE_SEG, BLADE_BEND, BLADE_W = 0.14, 4, 0.055, -15.0, 0.016

Pose = Dict[str, float]

# Staff keys (only used by bosses with a staff; their arms reach for the staff by IK):
#   sa = staff angle (same convention as limbs, 90 = pointing forward)
#   sx, sy = grip centre relative to the shoulder (H units, +x forward, +y down)
#   so = how far the staff is slid forward through the grip (H units)
GUARD: Pose = dict(lean=8, fs=40, fe=100, rs=20, re=130, fh=18, fk=-24, rh=-16, rk=-6, dx=0.0,
                   sa=172, sx=0.05, sy=0.2, so=0.0,  # staff held upright beside the body (not shielding it)
                   wa=150)  # sword blade angle (absolute, same convention), for bosses with a sword


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
BLOCK_POSE = _pose(lean=-6, fs=70, fe=105, rs=55, re=115, fh=75, fk=-105)  # knee up to check the kick
FALL_POSE = _pose(lean=-10, fs=150, fe=20, rs=120, re=30, fh=40, fk=-30, rh=0, rk=-10)
DOWN_POSE = _pose(lean=0, fs=170, fe=10, rs=150, re=20, fh=25, fk=-50, rh=10, rk=-25)
GETUP_POSE = _pose(lean=30, rh=60, rk=-110, fh=70, fk=-120, fs=50, fe=90, rs=40, re=110)
TUCK_POSE = _pose(lean=55, fs=100, fe=125, rs=85, re=135, fh=125, fk=-150, rh=115, rk=-150)  # rolled into a ball
DODGE_POSE = _pose(lean=-30, fs=60, fe=110, rs=45, re=120, fh=35, fk=-60, rh=-25, rk=-30)
STAR_WIND = _pose(lean=-8, rs=-70, re=100, fs=70, fe=60)          # throwing hand cocked back
STAR_THROW = _pose(lean=18, rs=92, re=5, fs=20, fe=120, dx=0.04)  # arm whips forward
# Staff attacks. Horizontal sweeps are drawn side-on, so the staff slides from behind
# the body to the front at a fixed height (what a horizontal swing looks like from the side).
STAFF_SWIPE_WIND = _pose(lean=-8, sa=200, sx=-0.02, sy=-0.14, so=0.0)              # raised behind the head
STAFF_SWIPE_EXT = _pose(lean=18, sa=100, sx=0.18, sy=0.07, so=0.22, dx=0.05)       # chopped down forward
# Ground sweep: crouched, staff angled down so the tip skims the floor.
STAFF_LOW_WIND = _pose(lean=30, rh=60, rk=-115, fh=60, fk=-120, sa=75, sx=-0.12, sy=0.3, so=-0.25)
STAFF_LOW_EXT = _pose(lean=40, rh=65, rk=-130, fh=50, fk=-110, sa=74, sx=0.05, sy=0.27, so=0.40, dx=0.04)
# High sweep: staff level at neck height, lower than the throwing stars, so you must duck deeper.
STAFF_HIGH_WIND = _pose(lean=-4, sa=90, sx=-0.18, sy=-0.05, so=-0.25)
STAFF_HIGH_EXT = _pose(lean=6, sa=90, sx=0.14, sy=-0.05, so=0.35, dx=0.04)
STAFF_POKE_WIND = _pose(lean=-12, sa=91, sx=-0.12, sy=0.07, so=-0.25, dx=-0.06)    # pulled back
STAFF_POKE_EXT = _pose(lean=22, sa=90, sx=0.22, sy=0.07, so=0.42, dx=0.14)         # lunging thrust
# Egyptian Soldier's khopesh (front hand).
SWORD_SLASH_WIND = _pose(lean=-6, fs=130, fe=25, wa=215)                 # raised up and back
SWORD_SLASH_EXT = _pose(lean=18, fs=105, fe=0, wa=105, dx=0.06)          # comes over the top at head height
SWORD_CUT_WIND = _pose(lean=2, fs=10, fe=130, wa=-60)                    # blade low behind
SWORD_CUT_EXT = _pose(lean=16, fs=88, fe=5, wa=114, dx=0.07)             # rising cut, ends at the chest
SWORD_FLUNG_POSE = _pose(lean=-24, fs=175, fe=30, wa=240, rs=-40, re=20, fh=25, fk=-35, dx=-0.05)  # parried
FLIP_CROUCH_POSE = _pose(lean=25, rh=55, rk=-100, fh=60, fk=-110, fs=20, fe=90)
# Mage casting poses.
CAST_UP_WIND = _pose(lean=-4, fs=120, fe=30, rs=110, re=40)
CAST_UP = _pose(lean=-10, fs=165, fe=10, rs=155, re=15)                              # meteor / icicles
CAST_LOW_WIND = _pose(lean=10, fs=30, fe=60, rs=20, re=70, fh=30, fk=-50, rh=-5, rk=-35)
CAST_LOW = _pose(lean=30, fs=60, fe=10, rs=50, re=15, fh=40, fk=-70, rh=0, rk=-50, dx=0.03)  # low fire wall
CAST_HIGH_WIND = _pose(lean=-6, fs=60, fe=110, rs=50, re=120)
CAST_HIGH = _pose(lean=10, fs=105, fe=5, rs=100, re=10, dx=0.03)                       # high fire wall
MAGE_TIRED = _pose(lean=28, fs=10, fe=20, rs=0, re=25, fh=25, fk=-40, rh=-10, rk=-25)  # drained after a spell
TELEPORT_POSE = _pose(lean=0, fs=100, fe=130, rs=95, re=135)                           # arms crossed, shimmering
SPELL_COLORS = {"meteor": (255, 120, 30), "icicle_rain": (120, 220, 255),
                "fire_wall_low": (255, 90, 30), "fire_wall_high": (170, 110, 255)}
KNOCKDOWN_TILT = 88.0  # degrees the body rotates backward when it falls


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
    ranged: bool = False   # throws a projectile at the end of the strike phase instead of hitting
    cue: str = ""          # warning shown during the wind-up ("JUMP!", "DUCK!", ...)
    spell: str = ""        # Mage: spell cast at the start of the attack (hazards handled by the game)


def _attack(name, label, limb, target, damage, chip, min_depth, radius, wind, ext, t_wind, t_strike, t_hold, t_rec,
            live_from=0.5, unblockable=False, cue="") -> AttackSpec:
    return AttackSpec(name, label, limb, target, damage, chip, min_depth, radius, (
        Phase(wind, t_wind),
        # Only the later part of the strike motion is live, so a fist or foot swinging
        # up from the wind-up doesn't clip your body or lowered arms on the way.
        Phase(ext, t_strike, active=True, ease="out", active_from=live_from),
        Phase(ext, t_hold, active=True),
        Phase(GUARD, t_rec),
    ), unblockable, cue=cue)


ATTACKS: Dict[str, AttackSpec] = {a.name: a for a in (
    _attack("jab", "Jab", "front_hand", "head", 6, 0.0, 0.85, 0.045, JAB_WIND, JAB_EXT, 0.30, 0.09, 0.08, 0.28),
    _attack("cross", "Cross", "rear_hand", "head", 9, 0.0, 0.85, 0.045, CROSS_WIND, CROSS_EXT, 0.38, 0.10, 0.08, 0.34),
    _attack("uppercut", "Uppercut", "rear_hand", "head", 18, 0.25, 0.88, 0.05, UPPER_WIND, UPPER_EXT,
            0.65, 0.14, 0.10, 0.50, live_from=0.65),
    _attack("high_kick", "High Kick", "front_foot", "head", 14, 0.20, 0.80, 0.06, KICK_WIND, KICK_EXT,
            0.55, 0.16, 0.12, 0.55, live_from=0.75),
    _attack("low_sweep", "Low Sweep", "front_foot", "legs", 12, 1.0, 0.80, 0.065, SWEEP_WIND, SWEEP_EXT,
            0.60, 0.20, 0.15, 0.55, live_from=0.3, unblockable=True, cue="JUMP!"),
    AttackSpec("throwing_star", "Throwing Star", "rear_hand", "head", C.STAR_DAMAGE, 1.0, 0.0, C.STAR_RADIUS, (
        Phase(STAR_WIND, 0.5),
        Phase(STAR_THROW, 0.12, ease="out"),
        Phase(STAR_THROW, 0.10),
        Phase(GUARD, 0.40),
    ), unblockable=True, ranged=True, cue="DUCK!"),
    # Ninja Monk's staff. Blockable ones still chip through (block_chip).
    _attack("staff_swipe", "Staff Swipe", "staff", "head", 12, 0.35, 0.82, 0.03, STAFF_SWIPE_WIND,
            STAFF_SWIPE_EXT, 0.50, 0.15, 0.10, 0.45, live_from=0.45),
    _attack("staff_ground", "Ground Sweep", "staff", "legs", 12, 1.0, 0.80, 0.03, STAFF_LOW_WIND,
            STAFF_LOW_EXT, 0.55, 0.22, 0.12, 0.50, live_from=0.3, unblockable=True, cue="JUMP!"),
    _attack("staff_high", "High Sweep", "staff", "head", 14, 0.35, 0.80, 0.03, STAFF_HIGH_WIND,
            STAFF_HIGH_EXT, 0.55, 0.22, 0.12, 0.50, live_from=0.3, cue="DUCK LOW!"),
    _attack("staff_poke", "Staff Poke", "staff", "torso", 12, 0.35, 0.90, 0.03, STAFF_POKE_WIND,
            STAFF_POKE_EXT, 0.60, 0.12, 0.15, 0.50, live_from=0.4, cue="JUMP BACK!"),
    # Egyptian Soldier's sword: hits hardest. Your shield parries it; a forearm block only reduces it.
    _attack("sword_slash", "Sword Slash", "sword", "head", 24, 0.3, 0.82, 0.03, SWORD_SLASH_WIND,
            SWORD_SLASH_EXT, 0.45, 0.14, 0.10, 0.45, live_from=0.45),
    _attack("sword_cut", "Sword Cut", "sword", "torso", 20, 0.3, 0.85, 0.03, SWORD_CUT_WIND,
            SWORD_CUT_EXT, 0.40, 0.12, 0.10, 0.40, live_from=0.8),
    # Same slash with almost no wind-up, used right after flipping over you.
    _attack("sword_flip_slash", "Flip Slash", "sword", "head", 24, 0.3, 0.82, 0.03, SWORD_SLASH_WIND,
            SWORD_SLASH_EXT, 0.22, 0.14, 0.10, 0.45, live_from=0.45),
)}


def _spell(name: str, label: str, wind: Pose, cast: Pose, target: str, cue: str) -> AttackSpec:
    """A Mage spell: wind-up, channel while the warnings count down, then a long drained recovery."""
    return AttackSpec(name, label, "front_hand", target, 0.0, 1.0, 0.0, 0.03, (
        Phase(wind, C.MAGE_CAST_WINDUP),
        Phase(cast, C.MAGE_CAST_CHANNEL),
        Phase(MAGE_TIRED, 0.3),
        Phase(MAGE_TIRED, C.MAGE_RECOVERY),  # vulnerable: no dodging or teleporting mid-attack
    ), cue=cue, spell=name)


ATTACKS.update({a.name: a for a in (
    _spell("meteor", "Meteor", CAST_UP_WIND, CAST_UP, "torso", "MOVE!"),
    _spell("icicle_rain", "Icicle Rain", CAST_UP_WIND, CAST_UP, "torso", "MOVE!"),
    _spell("fire_wall_low", "Low Fire Wall", CAST_LOW_WIND, CAST_LOW, "legs", "JUMP!"),
    _spell("fire_wall_high", "High Fire Wall", CAST_HIGH_WIND, CAST_HIGH, "head", "DUCK!"),
)})


class EnemyState(Enum):
    IDLE = auto()
    APPROACH = auto()
    ATTACK = auto()
    STUNNED = auto()
    DODGE = auto()
    FLIP = auto()
    TELEPORT = auto()
    KNOCKDOWN = auto()
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
        self.cue_font = arcade.pixel(int(C.CUE_TEXT_SIZE * 0.55))
        self._cue_cache: Dict[Tuple[str, Tuple[int, int, int]], pygame.Surface] = {}
        self.debug_font = pygame.font.Font(None, 22)
        self.reset(ground_y=screen_h - 10, height=screen_h * 0.75, x=screen_w * 0.75, boss=C.BOSSES[0])

    def reset(self, ground_y: float, height: float, x: float, boss: dict) -> None:
        self.boss = boss
        self.name = boss["name"]
        self.max_hp = float(boss["max_hp"])
        self.ground_y = ground_y
        self.H = height
        self.x = x
        self.facing = -1
        self.hp = self.max_hp
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
        self.attack_seq = 0           # increments every time an attack starts (lets the game tell attacks apart)
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
        self.tilt = 0.0          # knockdown rotation (degrees, backward)
        self.down_t = 0.0
        self.block_t = 0.0
        self.dodge_t = 0.0
        self.dodge_cooldown = 0.0
        self.sweep_cooldown = 0.0     # after a leg-sweep knockdown he checks leg kicks until this runs out
        self.rolling = False
        self.spin = 0.0          # roll rotation (degrees, backward)
        self.flip_t = 0.0
        self.flip_from = self.flip_to = 0.0
        self.flip_angle = 0.0    # front flip rotation (degrees)
        self.lift = 0.0          # height off the floor while flipping (pixels)
        self._flip_started = False
        self.stun_pose: Pose = STUN_POSE
        self.hazards_active = False   # set by the game: a spell's hazards are still live
        self.last_attack: Optional[str] = None
        self._casts: List[dict] = []
        self.pressure = 0.0           # Mage: accumulated time the player has spent close by
        self.teleport_cd = 0.0
        self.tele_t = 0.0
        self.tele_dest = x
        self.visible = True
        self._tele_events: List[Tuple[str, Vec]] = []
        self._spell_glows = {k: _make_glow(max(8, int(height * 0.07)), c) for k, c in SPELL_COLORS.items()}
        self._landed = False
        self.death_t = 0.0
        self._fade_layer: Optional[pygame.Surface] = None
        self._thrown: List[Tuple[Vec, Vec]] = []
        moves = list(boss["attack_weights"]) + (["sword_flip_slash"] if boss.get("flip_chance") else [])
        self._reach = {name: self._strike_offset(ATTACKS[name]) for name in moves}
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
        staff = None
        if self.boss.get("staff"):
            d = _dir(p["sa"], f)
            grip = (shoulder[0] + f * p["sx"] * H, shoulder[1] + p["sy"] * H)
            centre = _add(grip, d, p["so"] * H)
            staff = (_add(centre, d, -STAFF_LEN * H / 2), _add(centre, d, STAFF_LEN * H / 2))
            f_elbow, f_hand = self._reach_for(shoulder, _add(grip, d, STAFF_HAND_GAP * H))
            r_elbow, r_hand = self._reach_for(r_shoulder, _add(grip, d, -STAFF_HAND_GAP * H))
        else:
            f_elbow = _add(shoulder, _dir(p["fs"], f), UPPER_ARM * H)
            f_hand = _add(f_elbow, _dir(p["fs"] + p["fe"], f), FOREARM * H)
            r_elbow = _add(r_shoulder, _dir(p["rs"], f), UPPER_ARM * H)
            r_hand = _add(r_elbow, _dir(p["rs"] + p["re"], f), FOREARM * H)
        off = lambda v: (hip[0] + v[0], hip[1] + v[1])  # noqa: E731
        roff = lambda v: (r_hip[0] + v[0], r_hip[1] + v[1])  # noqa: E731
        sk = {
            "hip": hip, "shoulder": shoulder, "head": head, "spine": spine,
            "f_elbow": f_elbow, "f_hand": f_hand,
            "r_hip": r_hip, "r_shoulder": r_shoulder, "r_elbow": r_elbow, "r_hand": r_hand,
            "f_knee": off(fk), "f_ankle": off(fa), "f_toe": off(ft),
            "r_knee": roff(rk), "r_ankle": roff(ra), "r_toe": roff(rt),
        }
        if staff:
            sk["staff_back"], sk["staff_tip"] = staff
        if self.boss.get("sword"):
            # sword_0 is the hilt (in the hand); the blade runs straight, then hooks.
            angle = p["wa"]
            sk["sword_0"] = f_hand
            pt = _add(f_hand, _dir(angle, f), BLADE_STRAIGHT * H)
            sk["sword_1"] = pt
            for i in range(BLADE_CURVE_SEGS):
                angle += BLADE_BEND
                pt = _add(pt, _dir(angle, f), BLADE_SEG * H)
                sk[f"sword_{i + 2}"] = pt
        return sk

    def _reach_for(self, shoulder: Vec, target: Vec) -> Tuple[Vec, Vec]:
        """Two-bone arm IK: (elbow, hand) reaching from the shoulder toward target, elbow bent downward."""
        a, b = UPPER_ARM * self.H, FOREARM * self.H
        dx, dy = target[0] - shoulder[0], target[1] - shoulder[1]
        dist = max(abs(a - b) + 1e-3, min(a + b - 1e-3, math.hypot(dx, dy)))
        base = math.atan2(dy, dx)
        bend = math.acos(max(-1.0, min(1.0, (a * a + dist * dist - b * b) / (2 * a * dist))))
        elbows = [(shoulder[0] + a * math.cos(base + s * bend), shoulder[1] + a * math.sin(base + s * bend))
                  for s in (1, -1)]
        elbow = max(elbows, key=lambda e: e[1])
        hand = (shoulder[0] + dist * math.cos(base), shoulder[1] + dist * math.sin(base))
        return elbow, hand

    def _apply_tilt(self, sk: Dict[str, Vec]) -> Dict[str, Vec]:
        """Rotate the whole body backward around its feet (used for knockdowns)."""
        if self.tilt < 0.01:
            return sk
        a = math.radians(self.tilt) * self.facing
        ca, sa = math.cos(a), math.sin(a)
        px, py = self.x, self.ground_y
        lift = 0.05 * self.H * math.sin(math.radians(self.tilt))  # body thickness off the floor
        out = {}
        for k, (x, y) in sk.items():
            if k == "spine":  # a direction, not a point
                out[k] = (x * ca + y * sa, -x * sa + y * ca)
            else:
                dx, dy = x - px, y - py
                out[k] = (px + dx * ca + dy * sa, py - dx * sa + dy * ca - lift)
        return out

    def _apply_spin(self, sk: Dict[str, Vec]) -> Dict[str, Vec]:
        """Backward roll: rotate the curled-up body around its middle, kept touching the floor."""
        if self.spin < 0.01 or self.spin > 359.99:
            return sk
        a = math.radians(self.spin) * self.facing
        ca, sa = math.cos(a), math.sin(a)
        cx = (sk["hip"][0] + sk["shoulder"][0]) * 0.5
        cy = (sk["hip"][1] + sk["shoulder"][1]) * 0.5
        out = {}
        for k, (x, y) in sk.items():
            if k == "spine":
                out[k] = (x * ca + y * sa, -x * sa + y * ca)
            else:
                dx, dy = x - cx, y - cy
                out[k] = (cx + dx * ca + dy * sa, cy - dx * sa + dy * ca)
        lowest = max(y for k, (x, y) in out.items() if k != "spine") + 0.04 * self.H
        shift = self.ground_y - lowest  # roll along the floor: lowest point touches the ground
        return {k: (v if k == "spine" else (v[0], v[1] + shift)) for k, v in out.items()}

    @staticmethod
    def _limb_end(sk: Dict[str, Vec], limb: str) -> Vec:
        if limb == "front_hand":
            return sk["f_hand"]
        if limb == "rear_hand":
            return sk["r_hand"]
        if limb == "staff":
            return sk["staff_tip"]
        if limb == "sword":
            return sk[f"sword_{BLADE_CURVE_SEGS + 1}"]
        a, t = sk["f_ankle"], sk["f_toe"]
        return ((a[0] + t[0]) * 0.5, (a[1] + t[1]) * 0.5)

    def _strike_offset(self, spec: AttackSpec) -> float:
        """Furthest forward distance the striking limb reaches while its hitbox is live."""
        if spec.ranged or spec.spell:
            return 0.0
        i = next(i for i, ph in enumerate(spec.phases) if ph.active)
        start, ph = spec.phases[i - 1].pose, spec.phases[i]
        best = 0.0
        for k in range(11):
            u = ph.active_from + (1.0 - ph.active_from) * k / 10
            sk = self._skeleton(lerp_pose(start, ph.pose, u), 0.0, 1)
            best = max(best, self._limb_end(sk, spec.limb)[0])
        return best

    # --- colliders -----------------------------------------------------------
    @property
    def can_be_hit(self) -> bool:
        return self.state not in (EnemyState.DEAD, EnemyState.KNOCKDOWN, EnemyState.DODGE, EnemyState.FLIP,
                                  EnemyState.TELEPORT)

    def hurtboxes(self) -> List:
        if not self.can_be_hit:
            return []
        return [
            Capsule(self.sk["hip"], self.sk["shoulder"], 0.08 * self.H),
            Circle(self.sk["head"], HEAD_R * self.H * 1.1),
        ]

    def kick_targets(self) -> List[Tuple[str, object]]:
        """Hurtboxes a kick can land on, labelled 'head', 'feet' or 'torso' (in priority order)."""
        if not self.can_be_hit:
            return []
        sk, H = self.sk, self.H
        out = [("head", Circle(sk["head"], HEAD_R * H * 1.1))]
        for p in ("f", "r"):
            knee, ankle, toe = sk[p + "_knee"], sk[p + "_ankle"], sk[p + "_toe"]
            shin_mid = ((knee[0] + ankle[0]) * 0.5, (knee[1] + ankle[1]) * 0.5)
            out.append(("feet", Capsule(shin_mid, ankle, 0.04 * H)))
            out.append(("feet", Capsule(ankle, toe, 0.04 * H)))
        out.append(("torso", Capsule(sk["hip"], sk["shoulder"], 0.08 * H)))
        return out

    def staff_collider(self) -> Optional[Capsule]:
        """The staff itself, which the player can punch or kick to parry."""
        if not self.can_be_hit or "staff_tip" not in self.sk:
            return None
        return Capsule(self.sk["staff_back"], self.sk["staff_tip"], STAFF_R * self.H * 1.6)

    def sword_capsules(self) -> List[Capsule]:
        """The blade, segment by segment (hilt to tip)."""
        if "sword_1" not in self.sk:
            return []
        pts = [self.sk[f"sword_{i}"] for i in range(0, BLADE_CURVE_SEGS + 2)]
        return [Capsule(a, b, BLADE_W * self.H) for a, b in zip(pts, pts[1:])]

    def sword_parried(self) -> None:
        """His sword hit your shield: it's flung back and he staggers, wide open."""
        self.attack_resolved = True
        self.state = EnemyState.STUNNED
        self.attack = None
        self.stun_t = C.SWORD_PARRY_STAGGER
        self.stun_pose = SWORD_FLUNG_POSE
        self.vx = -self.facing * C.ENEMY_KNOCKBACK * self.H * 0.5

    def pop_casts(self) -> List[dict]:
        """Spells started since the last call, each with a snapshot of the player at cast time."""
        casts, self._casts = self._casts, []
        return casts

    def pop_teleport_events(self) -> List[Tuple[str, Vec]]:
        events, self._tele_events = self._tele_events, []
        return events

    def pop_flip_started(self) -> bool:
        started, self._flip_started = self._flip_started, False
        return started

    def staff_struck(self) -> str:
        """Player hit the staff. Mid-attack it's a parry (attack cancelled, monk staggered),
        otherwise the staff just gets knocked aside. Returns 'parry' or 'deflect'."""
        if self.state is EnemyState.ATTACK:
            self.state = EnemyState.STUNNED
            self.attack = None
            self.stun_t = C.STAFF_PARRY_STUN
            self.vx = -self.facing * C.ENEMY_KNOCKBACK * self.H * 0.6
            return "parry"
        self.x -= self.facing * 0.03 * self.H
        return "deflect"

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

    @property
    def strike_window_open(self) -> bool:
        """The current attack's hitbox is live this frame (whether or not it has already connected)."""
        return self._strike_live()

    # --- combat events -------------------------------------------------------
    def take_hit(self, damage: float) -> str:
        """Returns 'ko', 'stun' or 'armor' (damaged but not stunned)."""
        if self.state is EnemyState.DEAD:
            return "armor"
        self.hp = max(0.0, self.hp - damage)
        self.flash_t = 0.1
        if self.hp <= 0:
            self._die()
            return "ko"
        if self.stun_immunity > 0 or self.state is EnemyState.STUNNED:
            return "armor"
        self.state = EnemyState.STUNNED
        self.attack = None
        self.stun_t = C.ENEMY_STUN_TIME
        self.stun_pose = STUN_POSE
        self.vx = -self.facing * C.ENEMY_KNOCKBACK * self.H
        return "stun"

    def _die(self) -> None:
        """K.O.: knocked onto his back, then fades away (see _update_death)."""
        self.state = EnemyState.DEAD
        self.attack = None
        self.death_t = 0.0
        self.spin = self.flip_angle = self.lift = 0.0
        self.visible = True
        self._landed = False
        self.vx = -self.facing * C.ENEMY_KNOCKDOWN_KNOCKBACK * self.H * 0.7

    @property
    def fade(self) -> float:
        """1 = fully visible; drops to 0 once the K.O.'d body has lain on the floor for a moment."""
        if self.state is not EnemyState.DEAD:
            return 1.0
        return max(0.0, min(1.0, 1.0 - (self.death_t - C.ENEMY_DEATH_FADE_START) / C.ENEMY_DEATH_FADE_TIME))

    @property
    def sweep_ready(self) -> bool:
        """A leg kick would knock him down right now (he isn't still wary from the last one)."""
        return self.sweep_cooldown <= 0

    def take_kick(self, damage: float, part: str = "torso") -> str:
        """Returns 'ko', 'knockdown', 'knockback', 'blocked', 'checked' or 'none'.

        A kick to the feet (leg sweep) always knocks the boss down, then he checks leg kicks
        ('checked': chip damage only) for LEG_SWEEP_COOLDOWN. An unblocked head kick knocks him
        down too; one to the body only knocks him back.
        """
        if not self.can_be_hit:
            return "none"
        sweep = part == "feet"
        checked = sweep and not self.sweep_ready
        blocked = checked or (not sweep and self.state in (EnemyState.IDLE, EnemyState.APPROACH)
                              and random.random() < self.boss["kick_block_chance"])
        self.hp = max(0.0, self.hp - (damage * C.ENEMY_KICK_BLOCK_DAMAGE if blocked else damage))
        self.flash_t = 0.1
        if self.hp <= 0:
            self._die()
            return "ko"
        if blocked:
            self.block_t = 0.35  # knee up: checks the kick
            self.x -= self.facing * 0.03 * self.H
            return "checked" if checked else "blocked"
        if sweep:
            self.sweep_cooldown = C.LEG_SWEEP_COOLDOWN
        if part == "torso":
            self.state = EnemyState.STUNNED
            self.attack = None
            self.stun_t = C.ENEMY_STUN_TIME
            self.stun_pose = STUN_POSE
            self.vx = -self.facing * C.ENEMY_KICK_KNOCKBACK * self.H
            return "knockback"
        self.state = EnemyState.KNOCKDOWN
        self.attack = None
        self.down_t = 0.0
        self._landed = False
        self.vx = -self.facing * C.ENEMY_KNOCKDOWN_KNOCKBACK * self.H
        return "knockdown"

    def try_dodge(self) -> bool:
        """Called when a punch or kick is about to land. Returns True if the boss dodged it."""
        if (self.state not in (EnemyState.IDLE, EnemyState.APPROACH) or self.block_t > 0
                or self.dodge_cooldown > 0 or random.random() >= self.boss["dodge_chance"]):
            return False
        self.state = EnemyState.DODGE
        self.dodge_cooldown = self.boss["dodge_cooldown"]
        self.rolling = random.random() < self.boss["roll_chance"]
        if self.rolling:
            self.dodge_t = C.ENEMY_ROLL_TIME
            self.vx = -self.facing * C.ENEMY_ROLL_SPEED * self.H
        else:
            self.dodge_t = C.ENEMY_DODGE_TIME
            self.vx = -self.facing * C.ENEMY_DODGE_SPEED * self.H
        return True

    def pop_thrown(self) -> List[Tuple[Vec, Vec]]:
        """Projectiles released since the last call, as (start, aim point) pairs."""
        thrown, self._thrown = self._thrown, []
        return thrown

    def pop_landed(self) -> bool:
        """True once, on the frame the shadow hits the floor after a knockdown."""
        landed, self._landed = self._landed, False
        return landed

    def on_blocked(self) -> None:
        self.attack_resolved = True
        self.x -= self.facing * 0.04 * self.H

    # --- AI ------------------------------------------------------------------
    def _pick_attack(self) -> AttackSpec:
        names = list(self.boss["attack_weights"])
        if self.boss.get("no_repeat") and len(names) > 1 and self.last_attack in names:
            names.remove(self.last_attack)
        weights = [self.boss["attack_weights"][n] for n in names]
        return ATTACKS[random.choices(names, weights)[0]]

    def _enter_idle(self, duration: float) -> None:
        self.state = EnemyState.IDLE
        self.state_t = 0.0
        self.idle_time = duration
        self.attack = None

    def _desired_x(self, player, spec: AttackSpec) -> float:
        if spec.spell:  # spells are cast from wherever the Mage stands
            return self.x
        tx = player.target_point(spec.target)[0]
        if spec.ranged:  # back off to throwing distance
            stand_off = C.STAR_THROW_DISTANCE * self.sw
            side = -self.facing
            d = tx + side * stand_off
            if not self.sw * 0.06 <= d <= self.sw * 0.94:
                d = tx - side * stand_off
            return max(self.sw * 0.06, min(self.sw * 0.94, d))
        # Stand so the fully extended strike lands just in front of the target's centre,
        # which keeps our guard hand outside the player's guard until we strike.
        stand_off = self._reach[spec.name] + C.ENEMY_AIM_OFFSET * player.unit
        side = -self.facing  # stay on the side we're currently on
        d = tx + side * stand_off
        if not self.sw * 0.06 <= d <= self.sw * 0.94:
            d = tx - side * stand_off  # no room against the wall: go around to the other side
        return d

    def _start_attack(self, spec: AttackSpec, player=None) -> None:
        self.last_attack = spec.name
        if spec.spell and player is not None:
            # Capture the player's position NOW; the hazards are placed from this and never move.
            self._casts.append({
                "spell": spec.spell, "x": player.hip_mid()[0], "unit": player.unit,
                "ground_y": self.ground_y, "head_y": player.standing_head_y(),
                "origin_x": self.x + self.facing * 0.15 * self.H,
            })
        self.state = EnemyState.ATTACK
        self.attack = spec
        self.attack_seq += 1
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
        duration = ph.duration / self.boss["attack_speed"]
        self.phase_t += dt
        t = min(1.0, self.phase_t / duration)
        self.phase_progress = _ease(t, ph.ease)
        self.pose = lerp_pose(self.phase_from, ph.pose, self.phase_progress)

        if self.phase_i == 0 and not paused:  # keep tracking the player during the wind-up
            diff = self._desired_x(player, spec) - self.x
            max_step = C.ENEMY_WALK_SPEED * 0.5 * self.H * dt
            self.x += max(-max_step, min(max_step, diff))

        if self.phase_t >= duration:
            if spec.ranged and self.phase_i == 1:  # release at the end of the throwing motion
                hand = self._limb_end(self.sk, spec.limb)
                aim = (player.target_point("head")[0], player.standing_head_y())
                self._thrown.append((hand, aim))
            self.phase_t -= duration
            self.phase_progress = 0.0
            self.phase_from = ph.pose
            self.phase_i += 1
            if self.phase_i >= len(spec.phases):
                if spec.name == "jab" and random.random() < C.ENEMY_COMBO_CHANCE:
                    self.queued = ATTACKS["cross"]
                    self._enter_idle(0.05)
                else:
                    aggression = 1.0 + 0.6 * (1.0 - self.hp / self.max_hp)
                    self._enter_idle(random.uniform(self.boss["idle_min"], self.boss["idle_max"]) / aggression)

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
        self.block_t = max(0.0, self.block_t - dt)
        self.dodge_cooldown = max(0.0, self.dodge_cooldown - dt)
        self.sweep_cooldown = max(0.0, self.sweep_cooldown - dt)

        paused = not (player.calibrated and player.tracked)
        target: Optional[Pose] = None
        if self.boss.get("teleports") and not paused and self.state is not EnemyState.DEAD:
            self._track_pressure(dt, player)

        if self.state is EnemyState.DEAD:
            target = self._update_death(dt)
        elif self.state is EnemyState.ATTACK:
            self._update_attack(dt, player, paused)
        elif self.state is EnemyState.STUNNED:
            self.x += self.vx * dt
            self.vx *= math.exp(-7.0 * dt)
            self.stun_t -= dt
            target = dict(self.stun_pose)
            target["lean"] += 6 * math.sin(self.anim_t * 18)
            if self.stun_t <= 0:
                self._enter_idle(random.uniform(0.25, 0.5))
                self.stun_immunity = C.ENEMY_STUN_IMMUNITY
        elif self.state is EnemyState.KNOCKDOWN:
            target = self._update_knockdown(dt)
        elif self.state is EnemyState.FLIP:
            target = self._update_flip(dt, player)
        elif self.state is EnemyState.TELEPORT:
            target = self._update_teleport(dt, player)
        elif self.state is EnemyState.DODGE:
            self.x += self.vx * dt
            self.dodge_t -= dt
            if self.rolling:
                self.vx *= math.exp(-1.5 * dt)
                u = 1.0 - max(0.0, self.dodge_t) / C.ENEMY_ROLL_TIME
                self.spin = 360.0 * _ease(min(1.0, u / 0.85), "inout")  # finish the turn, then stand
                target = TUCK_POSE if u < 0.8 else GUARD
            else:
                self.vx *= math.exp(-6.0 * dt)
                target = DODGE_POSE
            if self.dodge_t <= 0:
                self.spin = 0.0
                self.rolling = False
                self._enter_idle(random.uniform(0.1, 0.3))
        elif self.block_t > 0:
            target = BLOCK_POSE
        elif paused:
            target = self._idle_pose()
        else:
            self.facing = 1 if player.com_x() >= self.x else -1
            if self.state is EnemyState.IDLE:
                self.state_t += dt
                target = self._idle_pose()
                if self._should_teleport():
                    self._start_teleport(player)
                elif self.state_t >= self.idle_time and not self.hazards_active:
                    if not self.queued and random.random() < self.boss.get("flip_chance", 0.0) \
                            and self._try_flip(player):
                        pass
                    else:
                        self.next_attack = self.queued or self._pick_attack()
                        self.queued = None
                        self.state = EnemyState.APPROACH
                        self.approach_t = 0.0
            elif self.state is EnemyState.APPROACH:
                self.approach_t += dt
                diff = self._desired_x(player, self.next_attack) - self.x
                if abs(diff) <= 0.035 * self.H or self.approach_t > C.ENEMY_APPROACH_TIMEOUT:
                    self._start_attack(self.next_attack, player)
                else:
                    step = math.copysign(min(abs(diff), C.ENEMY_WALK_SPEED * self.H * dt), diff)
                    self.x += step
                    self.walk_phase += dt * 10.0 * (1 if step * self.facing > 0 else -1)
                    target = self._walk_pose()

        # Keep the body on screen, including while it's lying on the floor behind its feet.
        lying = 1.15 * self.H * math.sin(math.radians(self.tilt))  # feet to outstretched hands
        lo = self.sw * 0.04 + (lying if self.facing > 0 else 0.0)
        hi = self.sw * 0.96 - (lying if self.facing < 0 else 0.0)
        self.x = max(lo, min(hi, self.x))
        if target is not None:
            k = 1.0 - math.exp(-dt * C.ENEMY_POSE_BLEND)
            self.pose = lerp_pose(self.pose, target, k)
        self.sk = self._apply_flip(self._apply_spin(self._apply_tilt(self._skeleton(self.pose, self.x, self.facing))))
        if self.state is EnemyState.ATTACK:
            cur = self._limb_end(self.sk, self.attack.limb)
            # Sweep only between live frames, never back into the wind-up.
            live = self._strike_live()
            self._strike_prev = self._strike_cur if (live and self._was_live) else cur
            self._strike_cur = cur
            self._was_live = live

    # --- Mage: defensive teleport ----------------------------------------------
    def _track_pressure(self, dt: float, player) -> None:
        """Build up 'pressure' while the player stays close; drain it slowly while they're away."""
        self.teleport_cd = max(0.0, self.teleport_cd - dt)
        if self.state is EnemyState.TELEPORT:
            return
        if abs(self.x - player.com_x()) < C.MAGE_TELEPORT_TRIGGER_DIST * self.H:
            self.pressure += dt
        else:
            self.pressure = max(0.0, self.pressure - C.MAGE_TELEPORT_DECAY * dt)

    def _should_teleport(self) -> bool:
        # Only from IDLE (never mid-cast, stunned or knocked down), only after sustained pressure,
        # and never while on cooldown. Getting hit doesn't add pressure by itself.
        return (bool(self.boss.get("teleports")) and self.teleport_cd <= 0
                and self.pressure >= C.MAGE_TELEPORT_PRESSURE)

    def _start_teleport(self, player) -> None:
        self.state = EnemyState.TELEPORT
        self.tele_t = 0.0
        com = player.com_x()
        d = C.MAGE_TELEPORT_DISTANCE * self.sw
        options = [x for x in (com - d, com + d) if self.sw * 0.1 <= x <= self.sw * 0.9]
        if not options:  # squeezed against a wall: take whichever spot is further from the player
            options = [max(self.sw * 0.1, min(self.sw * 0.9, com + d * (1 if com < self.sw / 2 else -1)))]
        self.tele_dest = random.choice(options)
        self._tele_events.append(("tell", self.center()))

    def _update_teleport(self, dt: float, player) -> Pose:
        self.tele_t += dt
        tell, gone = C.MAGE_TELEPORT_TELL, C.MAGE_TELEPORT_GONE
        if self.visible and self.tele_t >= tell:
            self.visible = False
            self._tele_events.append(("out", self.center()))
        if not self.visible and self.tele_t >= tell + gone:
            self.x = self.tele_dest
            self.facing = 1 if player.com_x() >= self.x else -1
            self.visible = True
            self.pressure = 0.0
            self.teleport_cd = C.MAGE_TELEPORT_COOLDOWN
            self._enter_idle(0.6)
            self.sk = self._skeleton(GUARD, self.x, self.facing)
            self._tele_events.append(("in", self.center()))
            return GUARD
        return TELEPORT_POSE

    def _try_flip(self, player) -> bool:
        """Leap over the player, landing at slashing distance on their other side."""
        spec = ATTACKS["sword_flip_slash"]
        side = 1 if self.x > player.com_x() else -1          # which side of the player we're on now
        tx = player.target_point(spec.target)[0]
        land = tx - side * (self._reach[spec.name] + C.ENEMY_AIM_OFFSET * player.unit)
        if not self.sw * 0.08 <= land <= self.sw * 0.92:
            return False
        self.state = EnemyState.FLIP
        self.flip_t = 0.0
        self.flip_from, self.flip_to = self.x, land
        return True

    def _update_flip(self, dt: float, player) -> Pose:
        """Short crouch, front flip over the player, land facing them and slash at once."""
        self.flip_t += dt
        u = self.flip_t / C.FLIP_TIME
        crouch = 0.18
        if u < crouch:
            return FLIP_CROUCH_POSE
        if not self._flip_started and u >= crouch:
            self._flip_started = True
        v = min(1.0, (u - crouch) / (1.0 - crouch))
        self.x = self.flip_from + (self.flip_to - self.flip_from) * _ease(v, "inout")
        self.lift = math.sin(math.pi * v) * C.FLIP_HEIGHT * self.H
        self.flip_angle = 360.0 * _ease(v, "inout")
        if v >= 1.0:
            self.lift = 0.0
            self.flip_angle = 0.0
            self.facing = -self.facing  # now on the other side, facing back at the player
            self._start_attack(ATTACKS["sword_flip_slash"])
            return GUARD
        return TUCK_POSE if 0.1 < v < 0.85 else GUARD

    def _apply_flip(self, sk: Dict[str, Vec]) -> Dict[str, Vec]:
        """Front flip: rotate forward around the body's middle and lift off the floor."""
        if self.flip_angle < 0.01 and self.lift < 0.5:
            return sk
        a = -math.radians(self.flip_angle) * self.facing  # forward, the opposite way to a backward roll
        ca, sa = math.cos(a), math.sin(a)
        cx = (sk["hip"][0] + sk["shoulder"][0]) * 0.5
        cy = (sk["hip"][1] + sk["shoulder"][1]) * 0.5
        out = {}
        for k, (x, y) in sk.items():
            if k == "spine":
                out[k] = (x * ca + y * sa, -x * sa + y * ca)
            else:
                dx, dy = x - cx, y - cy
                out[k] = (cx + dx * ca + dy * sa, cy - dx * sa + dy * ca - self.lift)
        return out

    def _update_death(self, dt: float) -> Pose:
        """Fall backward like a knockdown, but stay down (and fade, see `fade`)."""
        self.death_t += dt
        self.x += self.vx * dt
        self.vx *= math.exp(-5.0 * dt)
        fall, t = C.ENEMY_FALL_TIME, self.death_t
        if t < fall:
            self.tilt = KNOCKDOWN_TILT * (t / fall) ** 2
            return FALL_POSE
        if t - dt < fall:  # first frame on the floor: the game adds the impact
            self._landed = True
        bounce = t - fall
        self.tilt = KNOCKDOWN_TILT - (8.0 * math.sin(math.pi * bounce / 0.2) if bounce < 0.2 else 0.0)
        return DOWN_POSE

    def walk_in(self, dt: float, target_x: float, speed: float) -> None:
        """Boss intro: walk toward target_x without any AI (he can't attack yet)."""
        self.anim_t += dt
        diff = target_x - self.x
        step = math.copysign(min(abs(diff), speed * dt), diff)
        self.x += step
        if abs(step) > 1e-6:
            self.walk_phase += dt * 10.0
            target = self._walk_pose()
        else:
            target = self._idle_pose()
        self.pose = lerp_pose(self.pose, target, 1.0 - math.exp(-dt * C.ENEMY_POSE_BLEND))
        self.sk = self._skeleton(self.pose, self.x, self.facing)

    def _update_knockdown(self, dt: float) -> Pose:
        """Fall backward, lie on the floor, then get back up."""
        self.down_t += dt
        self.x += self.vx * dt
        self.vx *= math.exp(-5.0 * dt)
        fall, lie, rise = C.ENEMY_FALL_TIME, C.ENEMY_DOWN_TIME, C.ENEMY_GETUP_TIME
        t = self.down_t
        if t < fall:
            u = t / fall
            self.tilt = KNOCKDOWN_TILT * u * u  # accelerate into the floor
            return FALL_POSE
        if t < fall + lie:
            if t - dt < fall:  # first frame on the floor
                self._landed = True
            bounce = t - fall
            self.tilt = KNOCKDOWN_TILT - (8.0 * math.sin(math.pi * bounce / 0.2) if bounce < 0.2 else 0.0)
            return DOWN_POSE
        if t < fall + lie + rise:
            self.tilt = KNOCKDOWN_TILT * (1.0 - _ease((t - fall - lie) / rise, "inout"))
            return GETUP_POSE
        self.tilt = 0.0
        self._enter_idle(0.2)
        self.stun_immunity = C.ENEMY_STUN_IMMUNITY
        return GUARD

    # --- rendering -----------------------------------------------------------
    def draw(self, surf: pygame.Surface) -> None:
        fade = self.fade
        if fade <= 0.0 or not self.visible:
            return
        if fade < 1.0:  # dissolving after a K.O.: draw onto a transparent layer, then blend it in
            if self._fade_layer is None or self._fade_layer.get_size() != surf.get_size():
                self._fade_layer = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
            self._fade_layer.fill((0, 0, 0, 0))
            self._draw_body(self._fade_layer)
            self._fade_layer.set_alpha(int(255 * fade))
            surf.blit(self._fade_layer, (0, 0))
            return
        self._draw_body(surf)

    def _draw_body(self, surf: pygame.Surface) -> None:
        sk, H = self.sk, self.H
        shadow_x = self.x - self.facing * 0.45 * H * math.sin(math.radians(self.tilt))
        surf.blit(self._shadow, self._shadow.get_rect(center=(int(shadow_x), int(self.ground_y))))

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
        if self.boss.get("robe"):
            # Robe over the thighs (inserted before the arms) and a pointed hood.
            knees = ((sk["f_knee"][0] + sk["r_knee"][0]) / 2, (sk["f_knee"][1] + sk["r_knee"][1]) / 2)
            hem = (knees[0] + (knees[0] - sk["hip"][0]) * 0.35, knees[1] + (knees[1] - sk["hip"][1]) * 0.35)
            robe = [_add(sk["shoulder"], n, 0.08 * H), _add(hem, n, 0.15 * H), _add(hem, n, -0.15 * H),
                    _add(sk["shoulder"], n, -0.08 * H)]
            fwd = (-sp[1] * f, sp[0] * f)
            hd, R = sk["head"], HEAD_R * H
            hood = [_add(_add(hd, sp, 1.25 * R), fwd, 0.3 * R), _add(_add(hd, sp, 1.4 * R), fwd, -1.9 * R),
                    _add(_add(hd, sp, -0.9 * R), fwd, -1.1 * R), _add(_add(hd, sp, -0.7 * R), fwd, 0.35 * R)]
            front[8:8] = [("poly", robe), ("poly", hood)]

        rim = max(2.0, 0.006 * H)
        flashing = self.flash_t > 0 or (self.state is EnemyState.TELEPORT and int(self.anim_t * 24) % 2 == 0)

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

        staff = [("cap", sk["staff_back"], sk["staff_tip"], STAFF_R)] if "staff_tip" in sk else []
        paint(back + staff + front, self.boss["rim"], rim)
        paint(back, C.ENEMY_FLASH if flashing else C.ENEMY_BACK, 0)
        if staff:
            paint(staff, C.ENEMY_FLASH if flashing else C.STAFF_COLOR, 0)
            for end in ("staff_back", "staff_tip"):  # metal end caps
                pygame.draw.circle(surf, C.STAFF_CAP_COLOR, _ipt(sk[end]), max(2, int(STAFF_R * H * 1.3)))
        paint(front, C.ENEMY_FLASH if flashing else C.ENEMY_BODY, 0)

        if self.boss.get("headdress"):
            self._draw_headdress(surf, sk)
        # Eye: white normally, red while attacking.
        angry = self.state is EnemyState.ATTACK
        eye = _add(sk["head"], (f, 0), 0.03 * H)
        eye = (eye[0], eye[1] - 0.01 * H)
        ew, eh = max(4, int(0.024 * H)), max(2, int(0.009 * H))
        pygame.draw.ellipse(surf, C.ENEMY_EYES_ANGRY if angry else C.ENEMY_EYES,
                            pygame.Rect(int(eye[0] - ew / 2), int(eye[1] - eh / 2), ew, eh))
        if self.boss.get("headband"):
            self._draw_headband(surf, sk, self.boss["headband"])
        if "sword_1" in sk:
            self._draw_sword(surf, sk, flashing)

        # Mage: both hands glow in the spell's colour while casting.
        spell = self.attack.spell if self.state is EnemyState.ATTACK and self.attack else ""
        if spell and self.phase_i <= 1:
            glow = self._spell_glows[spell]
            for hand in ("f_hand", "r_hand"):
                surf.blit(glow, glow.get_rect(center=_ipt(sk[hand])), special_flags=pygame.BLEND_ADD)
            self._draw_cue(surf, self.attack.cue, (self.x, sk["head"][1] - 0.15 * H))
            return

        # Telegraph: glowing limb during the wind-up.
        if self.in_windup:
            progress = min(1.0, self.phase_t / (self.attack.phases[0].duration / self.boss["attack_speed"]))
            glow = self._glow.copy()
            glow.fill((int(255 * (0.4 + 0.6 * progress)),) * 3, special_flags=pygame.BLEND_MULT)
            p = self._limb_end(sk, self.attack.limb)
            surf.blit(glow, glow.get_rect(center=_ipt(p)), special_flags=pygame.BLEND_ADD)
            if self.attack.cue:
                legs = self.attack.target == "legs"
                self._draw_cue(surf, self.attack.cue, (self.x, self.ground_y - 0.55 * H if legs
                                                       else sk["head"][1] - 0.13 * H))

    def _draw_cue(self, surf: pygame.Surface, cue: str, pos: Vec) -> None:
        """Big outlined warning ('JUMP!', 'DUCK!', ...) that flashes yellow/white; kept on screen."""
        color = (255, 220, 80) if int(self.anim_t * 8) % 2 == 0 else (255, 255, 255)
        key = (cue, color)
        label = self._cue_cache.get(key)
        if label is None:
            label = self._cue_cache[key] = render_outlined(self.cue_font, cue, color)
        hw, hh = label.get_width() / 2, label.get_height() / 2
        x = min(max(pos[0], hw + 8), surf.get_width() - hw - 8)
        y = min(max(pos[1], hh + 70), surf.get_height() - hh - 8)  # clear of the health bars
        surf.blit(label, label.get_rect(center=(int(x), int(y))))

    def _draw_sword(self, surf: pygame.Surface, sk: Dict[str, Vec], flashing: bool) -> None:
        H = self.H
        hand, first = sk["sword_0"], sk["sword_1"]
        blade = [sk[f"sword_{i}"] for i in range(0, BLADE_CURVE_SEGS + 2)]
        w = max(3, int(BLADE_W * 2 * H))
        pygame.draw.lines(surf, (20, 14, 8), False, blade, w + 4)          # outline
        pygame.draw.lines(surf, C.ENEMY_FLASH if flashing else C.BLADE_COLOR, False, blade, w)
        for pt in blade:
            pygame.draw.circle(surf, C.ENEMY_FLASH if flashing else C.BLADE_COLOR, _ipt(pt), w // 2)
        length = math.hypot(first[0] - hand[0], first[1] - hand[1]) or 1.0
        back = ((hand[0] - first[0]) / length, (hand[1] - first[1]) / length)
        pygame.draw.line(surf, (40, 25, 12), hand, _add(hand, back, 0.045 * H), max(3, w))  # handle
        pygame.draw.circle(surf, (40, 25, 12), _ipt(hand), max(2, int(0.02 * H)))           # fist on the hilt

    def _draw_headdress(self, surf: pygame.Surface, sk: Dict[str, Vec]) -> None:
        """Striped Egyptian headcloth hanging behind the head."""
        H, f = self.H, self.facing
        head, sp = sk["head"], sk["spine"]
        n = (-sp[1] * f, sp[0] * f)  # points toward the back of the head
        top = _add(head, sp, HEAD_R * H * 0.9)
        back = _add(head, n, -HEAD_R * H * 1.05)
        low = _add(_add(head, sp, -0.13 * H), n, -0.07 * H)
        front_low = _add(_add(head, sp, -0.09 * H), n, 0.0)
        poly = [_add(top, n, 0.02 * H), back, low, front_low]
        pygame.draw.polygon(surf, (205, 165, 55), poly)
        for t in (0.35, 0.65):  # blue stripes
            a = (poly[0][0] + (poly[3][0] - poly[0][0]) * t, poly[0][1] + (poly[3][1] - poly[0][1]) * t)
            b = (poly[1][0] + (poly[2][0] - poly[1][0]) * t, poly[1][1] + (poly[2][1] - poly[1][1]) * t)
            pygame.draw.line(surf, (40, 70, 160), a, b, max(2, int(0.008 * H)))
        pygame.draw.polygon(surf, (20, 14, 8), poly, 2)
        pygame.draw.circle(surf, C.ENEMY_BODY, _ipt(head), int(HEAD_R * H * 0.75))  # face stays in shadow

    def _draw_headband(self, surf: pygame.Surface, sk: Dict[str, Vec], color) -> None:
        """Ninja headband: a band across the forehead with two tails fluttering behind."""
        H, f = self.H, self.facing
        head, sp = sk["head"], sk["spine"]
        n = (-sp[1], sp[0])
        band_c = _add(head, sp, 0.012 * H)
        w = max(2, int(0.016 * H))
        pygame.draw.line(surf, color, _add(band_c, n, HEAD_R * H), _add(band_c, n, -HEAD_R * H), w)
        knot = (band_c[0] - f * HEAD_R * H * 0.9, band_c[1])
        for i, spread in enumerate((0.0, 0.025)):
            pts = [knot]
            for j in range(1, 6):
                wave = math.sin(self.anim_t * 14 + j * 0.9 + i * 1.7) * 0.012 * H
                pts.append((knot[0] - f * 0.03 * H * j, knot[1] + (spread + 0.006 * j) * H + wave))
            pygame.draw.lines(surf, color, False, pts, max(2, w - i))

    def draw_debug(self, surf: pygame.Surface) -> None:
        if self.state is EnemyState.DEAD:
            return
        for hb in self.hurtboxes():
            draw_shape(surf, hb, (255, 80, 255))
        for name, hb in self.kick_targets():
            if name == "feet":  # kick here (or the head) for a knockdown
                draw_shape(surf, hb, (255, 160, 220), 1)
        staff = self.staff_collider()
        if staff:
            draw_shape(surf, staff, (255, 200, 80))
        for cap in self.sword_capsules():
            draw_shape(surf, cap, (255, 230, 120), 1)
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
