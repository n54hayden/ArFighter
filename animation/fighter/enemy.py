"""ShadowEnemy: a skinned 3D shadow fighter animated from motion-capture and hand-keyed clips.

The character is the Quaternius mannequin (see ASSETS.md) rendered by fighter/render3d.py.
Clips cross-fade through fighter/character.Animator; a masked overlay keeps the fighting guard
on the upper body while the legs walk; additive flinches push the spine / head away from hits.

Attack timing comes from the clips themselves: tools/build_character.py measures each attack's
striking limb, contact time and active window. The wind-up is stretched (never shortened) to
the attack's `windup` time so every strike stays readable, then the clip plays at full speed.
The strike collider is the striking fist / foot / weapon tip swept between frames while the
active window is open, and each attack resolves at most once.

The shadow lives on the floor lane (see fighter/floor.py): wx (sideways, body heights) changes,
the depth wz is fixed, and screen x, height H and the floor row follow from the perspective.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame

from . import config as C
from .character import Animator, CharacterData, quat_axis_angle
from .geometry import Capsule, Circle, Vec, draw_shape
from .render3d import CharacterRenderer, Look, _rgb, placement

LIMBS = ("hand.L", "hand.R", "foot.L", "foot.R")       # order used by the clip metadata
LIMB_JOINT = {"hand.L": "DEF-hand.L", "hand.R": "DEF-hand.R", "foot.L": "DEF-toe.L", "foot.R": "DEF-toe.R"}
HEAD_R = 0.065            # head collider radius (fraction of height)
TORSO_R = 0.085


@dataclass(frozen=True)
class AttackSpec:
    name: str
    label: str
    clip: str
    target: str            # head | torso | legs (what the AI lines up with)
    damage: float
    block_chip: float      # fraction of damage that still goes through a forearm block
    min_depth: float       # the player is out of reach if they step back past this depth ratio
    strike_radius: float   # fraction of H
    windup: float          # seconds the wind-up lasts at least (the clip's own is stretched to this)
    power: float = 0.5     # how hard it hits (reaction size)
    unblockable: bool = False
    ranged: bool = False   # throws a projectile at contact time instead of striking
    weapon: bool = False   # strikes with the weapon tip instead of a fist / foot
    cue: str = ""          # warning shown during the wind-up ("JUMP!", "DUCK!", ...)
    then: str = ""         # follow-up clip (e.g. getting up after a slide)
    spell: str = ""        # Mage: spell cast at the start (hazards handled by the game)
    limb: str = "hand.R"   # filled in from the clip metadata


def _atk(name, label, clip, target, damage, chip, min_depth, radius, windup, power=0.5, **kw) -> AttackSpec:
    return AttackSpec(name, label, clip, target, damage, chip, min_depth, radius, windup, power, **kw)


ATTACKS: Dict[str, AttackSpec] = {a.name: a for a in (
    _atk("jab", "Jab", "jab", "head", 6, 0.0, 0.85, 0.05, 0.30, 0.3),
    _atk("cross", "Cross", "cross", "head", 9, 0.0, 0.85, 0.05, 0.38, 0.5),
    _atk("high_kick", "Roundhouse", "kick_round", "head", 14, 0.20, 0.80, 0.06, 0.55, 0.8),
    _atk("side_kick", "Side Kick", "kick_side", "torso", 12, 0.20, 0.80, 0.065, 0.50, 0.8),
    _atk("front_kick", "Front Kick", "kick_front", "torso", 11, 0.20, 0.82, 0.06, 0.50, 0.7),
    _atk("low_sweep", "Slide Sweep", "slide", "legs", 12, 1.0, 0.80, 0.07, 0.60, 0.6, unblockable=True,
         cue="JUMP!", then="slide_exit"),
    _atk("throwing_star", "Throwing Star", "throw", "head", C.STAR_DAMAGE, 1.0, 0.0, C.STAR_RADIUS, 0.50,
         unblockable=True, ranged=True, cue="DUCK!"),
    # Ninja Monk's staff. Blockable ones still chip through.
    _atk("staff_swipe", "Staff Swipe", "sword_attack", "head", 12, 0.35, 0.82, 0.035, 0.50, 0.6, weapon=True),
    _atk("staff_ground", "Ground Sweep", "slash_b", "legs", 12, 1.0, 0.80, 0.035, 0.55, 0.6, weapon=True,
         unblockable=True, cue="JUMP!"),
    _atk("staff_lunge", "Staff Lunge", "slash_a", "torso", 12, 0.35, 0.85, 0.035, 0.55, 0.6, weapon=True,
         cue="BACK OFF!"),
    # Egyptian Soldier's khopesh: hits hardest. Your shield parries it; a forearm block only reduces it.
    _atk("sword_slash", "Sword Slash", "sword_attack", "head", 24, 0.3, 0.82, 0.035, 0.45, 0.8, weapon=True),
    _atk("sword_cut", "Sword Cut", "slash_b", "torso", 20, 0.3, 0.85, 0.035, 0.40, 0.7, weapon=True),
    _atk("sword_flip_slash", "Flip Slash", "slash_a", "head", 24, 0.3, 0.82, 0.035, 0.22, 0.8, weapon=True),
)}
SPELL_CUES = {"meteor": ("Meteor", "MOVE!", "torso"), "icicle_rain": ("Icicle Rain", "MOVE!", "torso"),
              "fire_wall_low": ("Low Fire Wall", "JUMP!", "legs"), "fire_wall_high": ("High Fire Wall", "DUCK!", "head")}
for _n, (_label, _cue, _target) in SPELL_CUES.items():
    ATTACKS[_n] = _atk(_n, _label, "spell_enter", _target, 0.0, 1.0, 0.0, 0.03, 0.0, cue=_cue, spell=_n)
SPELL_COLORS = {"meteor": (255, 120, 30), "icicle_rain": (120, 220, 255),
                "fire_wall_low": (255, 90, 30), "fire_wall_high": (170, 110, 255)}


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


def _make_glow(radius: int, color) -> pygame.Surface:
    surf = pygame.Surface((radius * 2, radius * 2))
    for i in range(radius, 0, -2):
        k = (1.0 - i / radius) ** 1.6
        pygame.draw.circle(surf, tuple(int(c * k) for c in color), (radius, radius), i)
    return surf


class ShadowEnemy:
    _attack_counter = 0  # attack ids are unique for the whole session (the hit log keys on them)

    """Lives on the floor lane: wx (sideways, body heights) changes, the depth wz stays fixed at the
    lane where you calibrated, so the shadow keeps your calibrated size and only moves left / right."""

    def __init__(self, screen_w: int, screen_h: int, floor, data: CharacterData,
                 renderer: Optional[CharacterRenderer]):
        self.sw, self.sh = screen_w, screen_h
        self.floor = floor
        self.data = data
        self.renderer = renderer
        self.rig = data.rig
        self.J = {n: i for i, n in enumerate(self.rig.names)}
        self.font = pygame.font.Font(None, 40)
        self.debug_font = pygame.font.Font(None, 22)
        # Fighting guard on the upper body while walking.
        self.upper_mask = np.zeros(len(self.rig))
        for n, i in self.J.items():
            if any(k in n for k in ("spine.002", "spine.003", "neck", "head", "shoulder", "arm", "hand", "f_", "thumb")):
                self.upper_mask[i] = 1.0
        self.walk_speed_m = self._clip_speed("walk")
        # Additive flinch axis: the body's sideways axis, in each spine joint's local frame.
        rest = self.rig.world(self.rig.rest_r, self.rig.rest_t[self.rig.hips], root=data.above)
        self._flinch_axes = {self.J[n]: rest[self.J[n]][:3, :3].T @ np.array([1.0, 0.0, 0.0])
                             for n in ("DEF-spine.001", "DEF-spine.002", "DEF-spine.003", "DEF-neck", "DEF-head")}
        self._glow = _make_glow(24, (255, 60, 30))
        self.reset(floor.world_x(screen_w * 0.75, 1.0), floor.depth_from_k(1.0), C.BOSSES[0])

    def _clip_speed(self, name: str) -> float:
        """Ground speed (m/s) the walk cycle was animated at: planted-foot travel per second."""
        clip = self.data.clips[name]
        toe = self.J["DEF-toe.L"]
        z = []
        for f in range(clip.frames):
            W = self.rig.world(clip.rot[f], clip.hips[f], root=self.data.above)
            z.append(W[toe, 2, 3])
        return max(0.3, (max(z) - min(z)) * 2.0 / clip.duration)

    # --- placement on the floor ----------------------------------------------
    @property
    def k(self) -> float:
        return self.floor.k_from_depth(self.wz)

    @property
    def H(self) -> float:
        return self.floor.H0 * self.k

    @property
    def ground_y(self) -> float:
        return self.floor.y_from_k(self.k)

    @property
    def x(self) -> float:
        return self.floor.screen_x(self.wx, self.k)

    @x.setter
    def x(self, value: float) -> None:
        self.wx = self.floor.world_x(value, self.k)

    @property
    def px_per_m(self) -> float:
        return self.H / self.data.height

    def reset(self, wx: float, wz: float, boss: dict) -> None:
        self.boss = boss
        self.name = boss["name"]
        self.max_hp = float(boss["max_hp"])
        self.mass = float(boss.get("mass", 1.0))
        self.wx, self.wz = wx, wz
        self.hp = self.max_hp
        self.facing = -1
        self.turn = 0.0              # visual yaw offset while turning round (radians, decays to 0)
        self.back_turned = 0.0       # 0..1: facing away (rolling away from you)
        self.look = Look(_rgb(boss["rim"]), _rgb(boss["rim"]),
                         _rgb(boss.get("gear_color", (120, 90, 55))),
                         tuple(k for k in ("staff", "sword", "headband", "headdress", "robe") if boss.get(k)))
        self.idle_clip = "idle" if boss.get("robe") else "guard"
        self.anim = Animator(self.data)
        self.anim.play(self.idle_clip, fade=0.0)
        self.state = EnemyState.IDLE
        self.state_t = 0.0
        self.idle_time = C.ENEMY_FIRST_ATTACK_DELAY
        self.attack: Optional[AttackSpec] = None
        self.next_attack: Optional[AttackSpec] = None
        self.queued: Optional[AttackSpec] = None
        self.attack_id = 0
        self.attack_t = 0.0          # time into the attack clip
        self.phase_i = 0             # 0 wind-up, 1 strike (live), 3 recovery
        self.attack_resolved = False
        self.attack_missed: Optional[AttackSpec] = None  # finished without touching anything
        self._strike_prev: Optional[Vec] = None
        self._strike_cur: Optional[Vec] = None
        self._released = False
        self.approach_t = 0.0
        self.stun_t = 0.0
        self.stun_immunity = 0.0
        self.vx = 0.0                # knockback / dodge velocity (screen px / s)
        self.mvx = 0.0               # walking velocity (body heights / s)
        self.flash_t = 0.0
        self.anim_t = 0.0
        self.hitstop = 0.0
        self.flinch = self.flinch_v = 0.0   # radians: + folds forward, - snaps back
        self.flinch_part = "body"
        self.down_t = 0.0
        self.block_t = 0.0
        self.dodge_t = 0.0
        self.dodge_cooldown = 0.0
        self.rolling = False
        self.lift = 0.0              # pixels off the floor (flips, hops)
        self.roll_angle = 0.0        # body rotation in the screen plane (front flip)
        self.flip_t = 0.0
        self.flip_from = self.flip_to = (wx, wz)
        self._flip_started = False
        self.hazards_active = False
        self.last_attack: Optional[str] = None
        self._casts: List[dict] = []
        self.pressure = 0.0
        self.teleport_cd = 0.0
        self.tele_t = 0.0
        self.tele_dest = (wx, wz)
        self.visible = True
        self.alpha = 1.0
        self._tele_events: List[Tuple[str, Vec]] = []
        self._landed = False
        self._thrown: List[Tuple[Vec, Vec]] = []
        self._wall_hit: Optional[Vec] = None
        self.dead_t = 0.0
        self.get_up_at = 0.0
        self.grip_angle = 0.0
        self._sizing = False        # held weapon: 0 = along the grip axis, pi = reversed (twirls between)
        self.gear_pts = self.renderer.gear_points(self.look.gear) if self.renderer else {}
        # reach of each move (fraction of height) for spacing
        self._reach = {}
        for name in list(boss["attack_weights"]) + (["sword_flip_slash"] if boss.get("flip_chance") else []):
            spec = ATTACKS[name]
            meta = self.data.clips[spec.clip].meta
            extra = 0.2 if spec.weapon else 0.0  # part of the weapon past the fist (it rarely points straight at you)
            self._reach[name] = (meta.get("reach", 0.0) + extra) / self.data.height if meta else 0.0
        self._pose()

    # --- skeleton ------------------------------------------------------------
    def _pose(self) -> None:
        """Evaluate the animation, place it on screen and refresh the 2D joints used for collisions."""
        self.anim.additive = self._flinch_additive()
        rot, hips = self.anim.pose()
        yaw = self.turn + math.pi * self.back_turned
        M = placement(self.x, self.ground_y, self.px_per_m, self.facing, lift=self.lift, yaw_extra=yaw,
                      roll=self.roll_angle, pivot=0.95)
        self.world = self.rig.world(rot, hips, root=M @ self.data.above)
        P = self.world[:, :2, 3]
        j = self.J
        head = self.world[j["DEF-head"]]
        head_c = head[:3, 3] + head[:3, 1] / max(np.linalg.norm(head[:3, 1]), 1e-6) * 0.09 * self.px_per_m
        self.sk = {
            "hip": tuple(P[self.rig.hips]), "shoulder": tuple(P[j["DEF-spine.003"]]), "head": (head_c[0], head_c[1]),
            "f_hand": tuple(P[j["DEF-hand.R"]]), "r_hand": tuple(P[j["DEF-hand.L"]]),
            "f_elbow": tuple(P[j["DEF-forearm.R"]]), "r_elbow": tuple(P[j["DEF-forearm.L"]]),
            "f_knee": tuple(P[j["DEF-shin.R"]]), "r_knee": tuple(P[j["DEF-shin.L"]]),
            "f_ankle": tuple(P[j["DEF-foot.R"]]), "r_ankle": tuple(P[j["DEF-foot.L"]]),
            "f_toe": tuple(P[j["DEF-toe.R"]]), "r_toe": tuple(P[j["DEF-toe.L"]]),
        }
        self.gear = {}
        fix = self.renderer.gear_fix(self.grip_angle) if self.renderer and self.gear_pts else None
        for name, (joint, bind_p) in self.gear_pts.items():
            skin = self.world[joint] @ self.rig.inv_bind[joint]
            p = skin @ (fix @ np.append(bind_p, 1.0))  # held weapon, flipped as the clip holds it
            self.gear[name] = (p[0], p[1])

    def _limb_point(self, limb: str) -> Vec:
        if limb == "weapon":
            tip = "staff_tip" if "staff_tip" in self.gear else f"sword_{len([k for k in self.gear if k.startswith('sword_')]) - 1}"
            return self.gear.get(tip, self.sk["f_hand"])
        m = self.world[self.J[LIMB_JOINT[limb]]]
        p = m[:3, 3]
        if limb.startswith("hand"):  # centre of the fist, a little past the wrist
            p = p + m[:3, 1] / max(np.linalg.norm(m[:3, 1]), 1e-6) * 0.08 * self.px_per_m
        return p[0], p[1]

    def _flinch_additive(self) -> Dict[int, np.ndarray]:
        if abs(self.flinch) < 1e-3:
            return {}
        names = ("DEF-spine.003", "DEF-neck", "DEF-head") if self.flinch_part == "head" else \
            ("DEF-spine.001", "DEF-spine.002", "DEF-spine.003")
        return {self.J[n]: quat_axis_angle(self._flinch_axes[self.J[n]], self.flinch / len(names)) for n in names}

    # --- colliders -----------------------------------------------------------
    @property
    def can_be_hit(self) -> bool:
        return self.state not in (EnemyState.DEAD, EnemyState.KNOCKDOWN, EnemyState.DODGE, EnemyState.FLIP,
                                  EnemyState.TELEPORT)

    def hurtboxes(self) -> List:
        if not self.can_be_hit:
            return []
        return [Capsule(self.sk["hip"], self.sk["shoulder"], TORSO_R * self.H),
                Circle(self.sk["head"], HEAD_R * self.H * 1.1)]

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
        out.append(("torso", Capsule(sk["hip"], sk["shoulder"], TORSO_R * H)))
        return out

    def staff_collider(self) -> Optional[Capsule]:
        """The staff itself, which the player can punch or kick to parry."""
        if not self.can_be_hit or "staff_tip" not in self.gear:
            return None
        return Capsule(self.gear["staff_back"], self.gear["staff_tip"], 0.03 * self.H)

    def sword_capsules(self) -> List[Capsule]:
        pts = [self.gear[k] for k in sorted((k for k in self.gear if k.startswith("sword_")), key=lambda s: int(s[6:]))]
        return [Capsule(a, b, 0.018 * self.H) for a, b in zip(pts, pts[1:])]

    def center(self) -> Vec:
        h, s = self.sk["hip"], self.sk["shoulder"]
        return ((h[0] + s[0]) * 0.5, (h[1] + s[1]) * 0.5)

    @property
    def in_windup(self) -> bool:
        return self.state is EnemyState.ATTACK and self.phase_i == 0

    def strike_colliders(self) -> List[Capsule]:
        """The live strike: the fist / foot / weapon tip swept over this frame's motion (no tunnelling),
        plus, for weapons, the whole staff / blade from the fist out."""
        cap = self.strike_collider()
        if cap is None:
            return []
        out = [cap]
        if self.attack.weapon:  # the whole staff / blade counts, not just its tip
            if "staff_tip" in self.gear:
                out.append(Capsule(self.gear["staff_back"], self.gear["staff_tip"], cap.radius))
            else:
                out.append(Capsule(self.sk["f_hand"], self._strike_cur, cap.radius))
        return out

    def strike_collider(self) -> Optional[Capsule]:
        """The striking fist / foot / weapon tip swept over this frame's motion (no tunnelling)."""
        if self.state is not EnemyState.ATTACK or self.attack is None:  # interrupted (hit, knocked down, KO)
            return None
        if self.attack_resolved or self.phase_i != 1 or self._strike_cur is None or self.attack.ranged \
                or self.attack.spell:
            return None
        return Capsule(self._strike_prev or self._strike_cur, self._strike_cur, self.attack.strike_radius * self.H)

    # --- events ----------------------------------------------------------------
    def pop_casts(self) -> List[dict]:
        casts, self._casts = self._casts, []
        return casts

    def pop_teleport_events(self) -> List[Tuple[str, Vec]]:
        events, self._tele_events = self._tele_events, []
        return events

    def pop_flip_started(self) -> bool:
        started, self._flip_started = self._flip_started, False
        return started

    def pop_thrown(self) -> List[Tuple[Vec, Vec]]:
        thrown, self._thrown = self._thrown, []
        return thrown

    def pop_landed(self) -> bool:
        landed, self._landed = self._landed, False
        return landed

    def pop_wall_hit(self) -> Optional[Vec]:
        hit, self._wall_hit = self._wall_hit, None
        return hit

    def _miss(self) -> None:
        """The strike window closed without touching anything: resolved as a miss (once)."""
        self.attack_resolved = True
        self.attack_missed = (self.attack, self.attack_id)

    def pop_missed(self):
        """(spec, attack id) of an attack that finished without touching anything, for the hit log."""
        m, self.attack_missed = self.attack_missed, None
        return m

    # --- combat events -------------------------------------------------------
    def _react(self, part: str, direction: int, power: float) -> None:
        """Additive flinch away from the hit: head snaps back, body folds; scaled by power."""
        from_front = direction == 0 or direction == -self.facing
        self.flinch_part = part
        impulse = (5.0 + 9.0 * power) * (1 if part != "head" else -1) * (1 if from_front else -1)
        self.flinch_v += impulse
        self.hitstop = C.ENEMY_HITSTOP * (0.6 + 0.6 * power)
        self.flash_t = 0.1

    def _knockback(self, speed_h: float, direction: int) -> None:
        push = direction if direction else -self.facing
        self.vx = push * speed_h * self.H / self.mass

    def take_hit(self, damage: float, part: str = "body", direction: int = 0, power: float = 0.5) -> str:
        """A punch. Returns 'ko', 'stun' or 'armor' (damaged but not stunned)."""
        if self.state is EnemyState.DEAD:
            return "armor"
        self.hp = max(0.0, self.hp - damage)
        self._react(part, direction, power)
        if self.hp <= 0:
            self._die()
            return "ko"
        if self.stun_immunity > 0 or self.state is EnemyState.STUNNED:
            self.anim.play("hit_head" if part == "head" else "hit_chest", fade=0.06, restart=True, speed=1.4)
            return "armor"
        self.state = EnemyState.STUNNED
        self.attack = None
        self.stun_t = C.ENEMY_STUN_TIME
        self._knockback(C.ENEMY_KNOCKBACK * (0.7 + 0.6 * power), direction)
        self.anim.play("hit_head" if part == "head" else "hit_chest", fade=0.06, restart=True, speed=1.3)
        return "stun"

    def take_kick(self, damage: float, part: str = "torso", direction: int = 0, power: float = 0.7) -> str:
        """Returns 'ko', 'knockdown', 'knockback', 'blocked' or 'none'.

        An unblocked kick to the head or feet knocks the boss down; one to the body knocks it back.
        """
        if not self.can_be_hit:
            return "none"
        blocked = (self.state in (EnemyState.IDLE, EnemyState.APPROACH)
                   and random.random() < self.boss["kick_block_chance"])
        self.hp = max(0.0, self.hp - (damage * C.ENEMY_KICK_BLOCK_DAMAGE if blocked else damage))
        self.flash_t = 0.1
        if self.hp <= 0:
            self._react("head" if part == "head" else "body", direction, power)
            self._die()
            return "ko"
        if blocked:
            self.block_t = 0.45
            self.hitstop = C.ENEMY_HITSTOP * 0.5
            self.wx -= self.facing * 0.03 / self.mass
            self.anim.play("guard_enter", fade=0.05, start=0.35, speed=1.6, restart=True)
            return "blocked"
        self._react("head" if part == "head" else "body", direction, power)
        if part == "torso":
            self.state = EnemyState.STUNNED
            self.attack = None
            self.stun_t = C.ENEMY_STUN_TIME + 0.15
            self._knockback(C.ENEMY_KICK_KNOCKBACK * (0.7 + 0.5 * power), direction)
            self.anim.play("hit_chest", fade=0.05, restart=True, speed=1.1)
            return "knockback"
        self.state = EnemyState.KNOCKDOWN
        self.attack = None
        self.down_t = 0.0
        self._landed = False
        self.vx = 0.0  # the knockback clip carries him back (root motion)
        self.anim.play("hit_knockback", fade=0.05, restart=True, speed=1.1)
        return "knockdown"

    def _die(self) -> None:
        self.state = EnemyState.DEAD
        self.attack = None
        self.dead_t = 0.0
        self.anim.play("death", fade=0.1, restart=True, speed=1.25)

    def try_dodge(self) -> bool:
        """Called when a punch or kick is about to land. Returns True if the boss dodged it."""
        if (self.state not in (EnemyState.IDLE, EnemyState.APPROACH) or self.block_t > 0
                or self.dodge_cooldown > 0 or random.random() >= self.boss["dodge_chance"]):
            return False
        self.state = EnemyState.DODGE
        self.dodge_cooldown = self.boss["dodge_cooldown"]
        self.rolling = random.random() < self.boss["roll_chance"]
        if self.rolling:  # turns away and dives into a roll
            self.dodge_t = C.ENEMY_ROLL_TIME
            self.vx = -self.facing * C.ENEMY_ROLL_SPEED * self.H
            clip = self.data.clips["roll"]
            self.anim.play("roll", fade=0.08, restart=True, speed=clip.duration / C.ENEMY_ROLL_TIME * 0.9)
        else:  # hop back
            self.dodge_t = C.ENEMY_DODGE_TIME
            self.vx = -self.facing * C.ENEMY_DODGE_SPEED * self.H
            self.anim.play("jump_land", fade=0.06, restart=True, speed=1.8)
        return True

    def staff_struck(self) -> str:
        """Player hit the staff: mid-attack it's a parry (attack cancelled, monk staggered)."""
        if self.state is EnemyState.ATTACK:
            self.state = EnemyState.STUNNED
            self.attack = None
            self.stun_t = C.STAFF_PARRY_STUN
            self._knockback(C.ENEMY_KNOCKBACK * 0.6, 0)
            self._react("body", 0, 0.5)
            self.anim.play("hit_chest", fade=0.05, restart=True, speed=1.4)
            return "parry"
        self.wx -= self.facing * 0.03 / self.mass
        self.hitstop = C.ENEMY_HITSTOP * 0.4
        return "deflect"

    def sword_parried(self) -> None:
        """His sword hit your shield: flung back, he staggers wide open."""
        self.attack_resolved = True
        self.state = EnemyState.STUNNED
        self.attack = None
        self.stun_t = C.SWORD_PARRY_STAGGER
        self._knockback(C.ENEMY_KNOCKBACK * 0.5, 0)
        self._react("head", 0, 0.9)
        self.anim.play("hit_head", fade=0.04, restart=True, speed=0.8)

    def on_blocked(self) -> None:
        self.attack_resolved = True
        self.hitstop = C.ENEMY_HITSTOP * 0.6
        self.wx -= self.facing * 0.04 / self.mass

    # --- AI ------------------------------------------------------------------
    def player_world(self, player, x: Optional[float] = None) -> Vec:
        """The player's (or one of their screen x's) position projected onto the lane."""
        return self.floor.world_x(player.com_x() if x is None else x, self.k), self.wz

    def _px_to_lane(self, length: float) -> float:
        return length / self.H

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
        self.anim.play(self.idle_clip, fade=0.25)

    def _desired_x(self, player, spec: AttackSpec) -> float:
        if spec.spell:
            return self.wx
        tx = self.player_world(player, player.target_point(spec.target)[0])[0]
        if spec.ranged:
            stand_off = self._px_to_lane(C.STAR_THROW_DISTANCE * self.sw)
        else:  # the fully extended strike lands just in front of the target's centre
            stand_off = self._reach.get(spec.name, 0.4) + C.ENEMY_AIM_OFFSET * self._px_to_lane(player.unit)
        side = -self.facing
        d = tx + side * stand_off
        if not self.floor.walkable(d, self.wz) and self.floor.walkable(tx - side * stand_off, self.wz):
            d = tx - side * stand_off
        return d

    def _walk_toward(self, tx: float, dt: float) -> float:
        a = C.ENEMY_ACCEL / self.mass
        dx = tx - self.wx
        want = math.copysign(min(C.ENEMY_WALK_SPEED, math.sqrt(2.0 * a * abs(dx))), dx)
        self.mvx += max(-a * dt, min(a * dt, want - self.mvx))
        self.wx += self.mvx * dt
        return self.mvx * dt

    def _coast(self, dt: float) -> None:
        if self.mvx:
            a = 2.0 * C.ENEMY_ACCEL / self.mass * dt
            self.mvx = 0.0 if abs(self.mvx) <= a else self.mvx - math.copysign(a, self.mvx)
            self.wx += self.mvx * dt

    def _set_facing(self, facing: int) -> None:
        if facing != self.facing:
            self.facing = facing
            self.turn = math.pi if self.turn == 0 else -self.turn  # turn round through the camera side

    def _start_attack(self, spec: AttackSpec, player=None) -> None:
        meta = self.data.clips[spec.clip].meta
        if "limb" in meta:
            spec = AttackSpec(**{**spec.__dict__, "limb": "weapon" if spec.weapon else LIMBS[int(meta["limb"])]})
        self.last_attack = spec.name
        ShadowEnemy._attack_counter += 1
        self.attack_id = ShadowEnemy._attack_counter
        if spec.spell and player is not None:
            k = player.depth_ratio
            self._casts.append({
                "spell": spec.spell, "x": player.hip_mid()[0], "unit": player.unit, "k": k,
                "ground_y": self.floor.y_from_k(k), "head_y": player.standing_head_y(),
                "origin_x": self.x + self.facing * 0.15 * self.H, "boss_height": self.floor.H0 * k,
            })
        self.state = EnemyState.ATTACK
        self.attack = spec
        self.attack_t = 0.0
        self.phase_i = 0
        self.attack_resolved = False
        self._strike_prev = self._strike_cur = None
        self._released = False
        if spec.spell:
            self.anim.play("spell_enter", fade=0.15, restart=True, speed=0.5 / C.MAGE_CAST_WINDUP)
        else:
            self.anim.play(spec.clip, fade=0.12, restart=True, speed=self._windup_speed(spec))

    def _windup_speed(self, spec: AttackSpec) -> float:
        cw = self.data.clips[spec.clip].meta.get("active_start", 0.2)
        return min(1.0, cw / max(spec.windup / self.boss["attack_speed"], 1e-3)) if spec.windup else 1.0

    def _update_attack(self, dt: float, player, paused: bool) -> None:
        spec = self.attack
        if spec.spell:
            self._update_spell(dt)
            return
        tr = self.anim.current
        clip = self.data.clips[spec.clip]
        meta = clip.meta
        if tr is None or tr.clip.name not in (spec.clip, spec.then):
            return
        if tr.clip is clip:
            t = tr.time
            a0, a1 = meta.get("active_start", 0.0), meta.get("active_end", clip.duration)
            tr.speed = self._windup_speed(spec) if t < a0 else self.boss["attack_speed"]
            new_phase = 0 if t < a0 else (1 if t <= a1 else 3)
            if self.phase_i == 0 and not paused:  # keep lining up during the wind-up
                step = C.ENEMY_WALK_SPEED * 0.5 * dt
                self.wx += max(-step, min(step, self._desired_x(player, spec) - self.wx))
            if spec.ranged and not self._released and t >= meta.get("contact", a0):
                self._released = True
                aim = (player.target_point("head")[0], player.standing_head_y())
                self._thrown.append((self._limb_point(spec.limb), aim))
                self.attack_resolved = True
            if new_phase == 3 and self.phase_i == 1 and not self.attack_resolved:
                self._miss()
            self.phase_i = new_phase
            if tr.finished or t >= clip.duration - 0.02:
                if spec.then:
                    self.anim.play(spec.then, fade=0.1, restart=True, speed=1.2)
                else:
                    self._finish_attack()
        elif tr.finished:  # follow-up clip done
            self._finish_attack()

    def _finish_attack(self) -> None:
        spec = self.attack
        if spec is not None and not spec.spell and not spec.ranged and not self.attack_resolved:
            self._miss()
        if spec is not None and spec.name == "jab" and random.random() < C.ENEMY_COMBO_CHANCE:
            self.queued = ATTACKS["cross"]
            self._enter_idle(0.05)
        else:
            aggression = 1.0 + 0.6 * (1.0 - self.hp / self.max_hp)
            self._enter_idle(random.uniform(self.boss["idle_min"], self.boss["idle_max"]) / aggression)

    def _update_spell(self, dt: float) -> None:
        """Wind-up, channel (warnings count down), then a long drained recovery: your opening."""
        self.attack_t += dt
        t = self.attack_t
        w, ch, rec = C.MAGE_CAST_WINDUP, C.MAGE_CAST_CHANNEL, 0.3 + C.MAGE_RECOVERY
        cur = self.anim.current.clip.name if self.anim.current else ""
        if t < w:
            self.phase_i = 0
        elif t < w + ch:
            self.phase_i = 0
            if cur != "spell_idle":
                self.anim.play("spell_idle", fade=0.12)
        elif t < w + ch + rec:
            self.phase_i = 3
            if cur not in ("spell_exit", "idle"):
                self.anim.play("spell_exit", fade=0.1, restart=True)
            elif cur == "spell_exit" and self.anim.current.finished:
                self.anim.play("idle", fade=0.3, speed=0.5)  # drained and hunched
            self.flinch_part, self.flinch = "body", 0.35  # slumped forward
        else:
            self.flinch = 0.0
            self._finish_attack()

    # --- per frame ------------------------------------------------------------
    def update(self, dt: float, player) -> None:
        self.anim_t += dt
        self.flash_t = max(0.0, self.flash_t - dt)
        self.stun_immunity = max(0.0, self.stun_immunity - dt)
        self.block_t = max(0.0, self.block_t - dt)
        self.dodge_cooldown = max(0.0, self.dodge_cooldown - dt)
        self.turn = math.copysign(max(0.0, abs(self.turn) - math.pi * dt / C.ENEMY_TURN_TIME), self.turn) if self.turn else 0.0
        target_back = 1.0 if (self.state is EnemyState.DODGE and self.rolling) else 0.0
        self.back_turned += (target_back - self.back_turned) * min(1.0, dt * 14)
        # flinch spring
        if not (self.state is EnemyState.ATTACK and self.attack and self.attack.spell and self.phase_i == 3):
            self.flinch_v += (-C.FLINCH_STIFFNESS * self.flinch - C.FLINCH_DAMPING * self.flinch_v) * dt
            self.flinch += self.flinch_v * dt
        # hit-stop: the body freezes for a beat when a hit lands
        anim_dt = dt * (0.05 if self.hitstop > 0 else 1.0)
        self.hitstop = max(0.0, self.hitstop - dt)
        travel = self.anim.update(anim_dt)
        if self.gear_pts and self.anim.current is not None:  # twirl the weapon round if the clip holds it reversed
            want = 0.0 if self.anim.current.clip.meta.get("grip", 1.0) > 0 else math.pi
            step = math.pi * dt / 0.15
            self.grip_angle += max(-step, min(step, want - self.grip_angle))

        if self.state is EnemyState.DEAD:
            self.dead_t += dt
            if self.dead_t > 2.5:
                self.alpha = max(0.0, self.alpha - dt * 0.6)  # the shadow dissolves
            self._pose()
            return

        if player.calibrated and player.tracked and self.state not in (EnemyState.FLIP, EnemyState.TELEPORT):
            self._match_player_size(dt, player)
        if self.state is not EnemyState.ATTACK:  # an interrupted strike leaves nothing live behind
            self.phase_i = 0
            self._strike_prev = self._strike_cur = None
        paused = not (player.calibrated and player.tracked)
        prev = (self.wx, self.wz)
        dir_ = -1 if self.back_turned > 0.5 else 1
        self.wx += travel / self.data.height * self.facing * dir_  # root motion
        if self.boss.get("teleports") and not paused:
            self._track_pressure(dt, player)
        if self.state not in (EnemyState.IDLE, EnemyState.APPROACH, EnemyState.ATTACK):
            self.mvx = 0.0

        if self.state is EnemyState.ATTACK:
            self._coast(dt)
            self._update_attack(dt, player, paused)
        elif self.state is EnemyState.STUNNED:
            self.x += self.vx * dt
            self.vx *= math.exp(-7.0 * dt)
            self.stun_t -= dt
            if self.stun_t <= 0 and (self.anim.current is None or self.anim.normalized() > 0.6):
                self._enter_idle(random.uniform(0.25, 0.5))
                self.stun_immunity = C.ENEMY_STUN_IMMUNITY
        elif self.state is EnemyState.KNOCKDOWN:
            self._update_knockdown(dt)
        elif self.state is EnemyState.FLIP:
            self._update_flip(dt, player)
        elif self.state is EnemyState.TELEPORT:
            self._update_teleport(dt, player)
        elif self.state is EnemyState.DODGE:
            self.x += self.vx * dt
            self.dodge_t -= dt
            if self.rolling:
                self.vx *= math.exp(-1.5 * dt)
            else:
                self.vx *= math.exp(-6.0 * dt)
                u = 1.0 - max(0.0, self.dodge_t) / C.ENEMY_DODGE_TIME
                self.lift = 4.0 * u * (1.0 - u) * C.DODGE_HOP_HEIGHT * self.H
            if self.dodge_t <= 0:
                self.lift = 0.0
                self.rolling = False
                self._enter_idle(random.uniform(0.1, 0.3))
        elif self.block_t > 0:
            pass
        elif paused:
            if self.anim.current and self.anim.current.clip.name not in (self.idle_clip,):
                self.anim.play(self.idle_clip, fade=0.3)
        else:
            self._set_facing(1 if self.player_world(player)[0] >= self.wx else -1)
            if self.state is EnemyState.IDLE:
                self._coast(dt)
                self.state_t += dt
                self._locomotion()
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
                tx = self._desired_x(player, self.next_attack)
                if abs(tx - self.wx) <= 0.035 or self.approach_t > C.ENEMY_APPROACH_TIMEOUT:
                    self.anim.set_overlay(None)
                    self._start_attack(self.next_attack, player)
                else:
                    self._walk_toward(tx, dt)
                    self._locomotion()

        # Stay between the walls, including while lying on the floor.
        if self.state not in (EnemyState.FLIP, EnemyState.TELEPORT):
            offsets = [-C.FLOOR_HALF_FOOTPRINT, 0.0, C.FLOOR_HALF_FOOTPRINT]
            if self.state is EnemyState.KNOCKDOWN:
                offsets.append(-self.facing * 0.9)  # lying flat behind the feet
            want = self.wx
            self.wx = self.floor.clamp(prev, (self.wx, self.wz), offsets)[0]
            if abs(self.wx - want) > 1e-6:
                side = 1 if want > self.wx else -1
                real_wall = self.floor.walls[1 if side > 0 else 0] is not None
                if real_wall and abs(self.vx) / max(1.0, self.H) > C.WALL_SLAM_SPEED:
                    self._wall_hit = (self.x + side * 0.08 * self.H, self.ground_y - 0.5 * self.H)
                self.mvx = 0.0
                self.vx = 0.0
        self._pose()
        if self.state is EnemyState.ATTACK and self.attack and not self.attack.spell and not self.attack.ranged:
            cur = self._limb_point(self.attack.limb)
            self._strike_prev = self._strike_cur if self.phase_i == 1 and self._strike_cur is not None else cur
            self._strike_cur = cur

    def _match_player_size(self, dt: float, player) -> None:
        """Stand at the player's depth so both fighters are the same height on the same floor line.

        Follows smoothly with a dead zone (no wobble from tracking noise), keeps the screen x fixed
        (movement stays left / right only), and lags a little so a quick step back still dodges.
        """
        target = self.floor.depth_from_k(max(0.55, min(1.6, player.depth_ratio)))
        gap = abs(target - self.wz) / self.wz
        if gap > C.ENEMY_SIZE_DEADZONE:
            self._sizing = True
        elif gap < C.ENEMY_SIZE_DEADZONE * 0.25:
            self._sizing = False
        if self._sizing:
            x = self.x
            self.wz += (target - self.wz) * (1.0 - math.exp(-dt / C.ENEMY_SIZE_FOLLOW_TAU))
            self.x = x

    def _locomotion(self) -> None:
        """Walk cycle matched to ground speed (forward or backward), guard held on the upper body."""
        speed_m = self.mvx * self.data.height * self.facing  # + walking toward where he faces
        if abs(self.mvx) > 0.05:
            rate = speed_m / self.walk_speed_m
            tr = self.anim.play("walk", fade=0.2, speed=rate)
            tr.speed = rate
            if self.idle_clip == "guard":
                self.anim.set_overlay("guard", self.upper_mask, 0.85)
        else:
            if self.anim.current and self.anim.current.clip.name == "walk":
                self.anim.play(self.idle_clip, fade=0.25)
            self.anim.set_overlay(None)

    def _update_knockdown(self, dt: float) -> None:
        """Knocked off his feet (clip carries him back), lies there, then gets up."""
        self.down_t += dt
        cur = self.anim.current.clip.name if self.anim.current else ""
        if cur == "hit_knockback":
            if not self._landed and self.anim.current.time > 0.32:
                self._landed = True  # hits the floor
            if self.anim.current.finished:
                self.anim.play("get_up", fade=0.2, restart=True, speed=0.0)  # lying still
                self.get_up_at = self.down_t + C.ENEMY_DOWN_TIME * 0.6
        elif cur == "get_up":
            tr = self.anim.current
            if tr.speed == 0.0 and self.down_t >= self.get_up_at:
                tr.speed = 1.35
            if tr.finished:
                self._enter_idle(0.2)
                self.stun_immunity = C.ENEMY_STUN_IMMUNITY

    # --- Mage: defensive teleport ----------------------------------------------
    def _track_pressure(self, dt: float, player) -> None:
        self.teleport_cd = max(0.0, self.teleport_cd - dt)
        if self.state is EnemyState.TELEPORT:
            return
        if abs(self.wx - self.player_world(player)[0]) < C.MAGE_TELEPORT_TRIGGER_DIST:
            self.pressure += dt
        else:
            self.pressure = max(0.0, self.pressure - C.MAGE_TELEPORT_DECAY * dt)

    def _should_teleport(self) -> bool:
        return (bool(self.boss.get("teleports")) and self.teleport_cd <= 0
                and self.pressure >= C.MAGE_TELEPORT_PRESSURE)

    def _start_teleport(self, player) -> None:
        self.state = EnemyState.TELEPORT
        self.tele_t = 0.0
        px = self.player_world(player)[0]
        d = self._px_to_lane(C.MAGE_TELEPORT_DISTANCE * self.sw)
        options = [(x, self.wz) for x in (px - d, px + d) if self.floor.walkable(x, self.wz)]
        if not options:
            options = [self.floor.nearest_walkable(px + d * (1 if px < 0 else -1), self.wz)]
        self.tele_dest = random.choice(options)
        self.anim.play("spell_enter", fade=0.1, restart=True)
        self._tele_events.append(("tell", self.center()))

    def _update_teleport(self, dt: float, player) -> None:
        self.tele_t += dt
        tell, gone = C.MAGE_TELEPORT_TELL, C.MAGE_TELEPORT_GONE
        self.alpha = 1.0 if self.tele_t < tell * 0.4 else (0.35 if int(self.anim_t * 24) % 2 else 0.9)
        if self.visible and self.tele_t >= tell:
            self.visible = False
            self._tele_events.append(("out", self.center()))
        if not self.visible and self.tele_t >= tell + gone:
            self.wx, self.wz = self.tele_dest
            self.facing = 1 if self.player_world(player)[0] >= self.wx else -1
            self.turn = 0.0
            self.visible = True
            self.alpha = 1.0
            self.pressure = 0.0
            self.teleport_cd = C.MAGE_TELEPORT_COOLDOWN
            self._enter_idle(0.6)
            self._pose()
            self._tele_events.append(("in", self.center()))

    # --- Egyptian Soldier: flip over the player --------------------------------
    def _try_flip(self, player) -> bool:
        spec = ATTACKS["sword_flip_slash"]
        side = 1 if self.wx > self.player_world(player)[0] else -1
        tx = self.player_world(player, player.target_point(spec.target)[0])[0]
        land = tx - side * (self._reach.get(spec.name, 0.4) + C.ENEMY_AIM_OFFSET * self._px_to_lane(player.unit))
        if not self.floor.walkable(land, self.wz):
            return False
        self.state = EnemyState.FLIP
        self.flip_t = 0.0
        self.flip_from, self.flip_to = (self.wx, self.wz), (land, self.wz)
        self.anim.set_overlay(None)
        self.anim.play("flip_start", fade=0.08, restart=True, speed=1.0)
        return True

    def _update_flip(self, dt: float, player) -> None:
        """Crouch, front flip over the player (ballistic arc), land facing them and slash at once."""
        self.flip_t += dt
        u = self.flip_t / C.FLIP_TIME
        crouch = 0.18
        if u < crouch:
            return
        if not self._flip_started:
            self._flip_started = True
        v = min(1.0, (u - crouch) / (1.0 - crouch))
        (fx, fz), (tx, tz) = self.flip_from, self.flip_to
        self.wx = fx + (tx - fx) * v
        self.lift = 4.0 * v * (1.0 - v) * C.FLIP_HEIGHT * self.H
        self.roll_angle = -self.facing * 2 * math.pi * (v * v * (3 - 2 * v))
        if v >= 1.0:
            self.lift = 0.0
            self.roll_angle = 0.0
            self.facing = -self.facing  # now on the other side, facing back at the player
            self.turn = 0.0
            self._landed = True
            self._start_attack(ATTACKS["sword_flip_slash"])
            self.anim.tracks[-1].time = 0.0

    # --- rendering -----------------------------------------------------------
    def draw_shadow(self, surf: pygame.Surface) -> None:
        """Contact shadow under the body: follows a lying body, shrinks and fades when airborne."""
        if not self.visible or self.alpha <= 0.05:
            return
        P = self.world[:, :2, 3]
        xs = P[:, 0]
        H = self.H
        air = min(1.0, self.lift / (0.8 * H))
        width = max(0.42 * H, (xs.max() - xs.min()) * 0.9) * (1.0 - 0.45 * air)
        self.floor.draw_contact_shadow(surf, (xs.max() + xs.min()) * 0.5, self.ground_y, width,
                                       (1.0 - 0.6 * air) * self.alpha)

    def _bounds(self) -> Tuple[int, int, int, int]:
        pts = [self.world[:, :2, 3]] + ([np.array(list(self.gear.values()))] if self.gear else [])
        P = np.concatenate(pts)
        m = 0.22 * self.H
        x0, y0 = max(0, int(P[:, 0].min() - m)), max(0, int(P[:, 1].min() - m))
        x1, y1 = min(self.sw, int(P[:, 0].max() + m)), min(self.sh, int(P[:, 1].max() + m))
        return x0, y0, max(0, x1 - x0), max(0, y1 - y0)

    def draw(self, surf: pygame.Surface, rim_side: float = 0.0) -> None:
        if not self.visible or self.alpha <= 0.02:
            return
        rect = self._bounds()
        if self.renderer is not None:
            eyes = (1.0, 0.25, 0.12) if self.state is EnemyState.ATTACK else None
            look = self.look if eyes is None else Look(self.look.rim, self.look.glow, self.look.gear_color,
                                                       self.look.gear)
            img, at = self.renderer.render(self.world, rect, look, flash=1.0 if self.flash_t > 0 else 0.0,
                                           alpha=self.alpha, rim_side=rim_side, eyes=eyes,
                                           grip_angle=self.grip_angle, latency=True)
            if img is not None:
                surf.blit(img, at[:2])
        else:
            self._draw_fallback(surf)
        self._draw_telegraph(surf)

    def _draw_fallback(self, surf: pygame.Surface) -> None:
        """Flat silhouette from the skeleton (no GPU)."""
        P = self.world[:, :2, 3]
        for i, p in enumerate(self.rig.parents):
            if p >= 0 and "f_" not in self.rig.names[i] and "thumb" not in self.rig.names[i]:
                pygame.draw.line(surf, (10, 8, 16), P[p], P[i], max(3, int(0.05 * self.H)))
        pygame.draw.circle(surf, (10, 8, 16), [int(v) for v in self.sk["head"]], int(HEAD_R * self.H))

    def _draw_telegraph(self, surf: pygame.Surface) -> None:
        if self.state is not EnemyState.ATTACK or self.attack is None or self.phase_i != 0:
            return
        spec = self.attack
        if spec.spell:
            color = SPELL_COLORS[spec.spell]
            for hand in ("f_hand", "r_hand"):
                g = _make_glow(max(8, int(self.H * 0.06)), color)
                surf.blit(g, g.get_rect(center=[int(v) for v in self.sk[hand]]), special_flags=pygame.BLEND_ADD)
        else:
            g = self._glow
            surf.blit(g, g.get_rect(center=[int(v) for v in self._limb_point(spec.limb)]),
                      special_flags=pygame.BLEND_ADD)
        if spec.cue and int(self.anim_t * 8) % 2 == 0:
            label = self.font.render(spec.cue, True, (255, 220, 80))
            y = self.ground_y - 0.55 * self.H if spec.target == "legs" else self.sk["head"][1] - 0.14 * self.H
            surf.blit(label, label.get_rect(center=(int(self.x), int(y))))

    def draw_debug(self, surf: pygame.Surface) -> None:
        if self.state is EnemyState.DEAD:
            return
        for hb in self.hurtboxes():
            draw_shape(surf, hb, (255, 80, 255))
        for name, hb in self.kick_targets():
            if name == "feet":
                draw_shape(surf, hb, (255, 160, 220), 1)
        staff = self.staff_collider()
        if staff:
            draw_shape(surf, staff, (255, 200, 80))
        for cap in self.sword_capsules():
            draw_shape(surf, cap, (255, 230, 120), 1)
        strikes = self.strike_colliders()
        for strike in strikes:
            draw_shape(surf, strike, (255, 40, 40), 0)
        if not strikes and self.in_windup and self.attack and not self.attack.spell:
            draw_shape(surf, Circle(self._limb_point(self.attack.limb), self.attack.strike_radius * self.H),
                       (255, 150, 40))
        label = self.state.name
        if self.state is EnemyState.ATTACK and self.attack:
            label += f" {self.attack.label} [{('windup', 'STRIKE', '', 'recover')[self.phase_i]}] #{self.attack_id}"
        elif self.state is EnemyState.APPROACH and self.next_attack:
            label += f" -> {self.next_attack.label}"
        if self.anim.current:
            label += f"  ({self.anim.current.clip.name} {self.anim.current.time:.2f}s)"
        text = self.debug_font.render(label, True, (255, 200, 255))
        head = self.sk["head"]
        surf.blit(text, text.get_rect(midbottom=(int(head[0]), int(head[1] - 0.12 * self.H))))
