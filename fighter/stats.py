"""FightStats: one fight's combat statistics, shown on the Fight Summary screen.

The game reports lifecycle events here and this class does the counting, so every
counter lives in one place and is created fresh for each fight.

Player attacks are tracked per limb as *motions*. A motion starts when a fist or foot
crosses the existing strike-speed threshold toward the boss and ends once it slows down
or pulls back. Each motion counts as thrown at most once and landed at most once, no
matter how many frames it stays fast or overlaps a hurtbox.

Enemy attacks (melee strikes, projectiles, spell hazards...) are tracked generically as
*pending attacks*. The game opens one per real attack (an attack made of several
hazards, like an icicle storm, is one attack with several parts), reports hits, blocks
or "couldn't be judged" (player untracked), and closes each part when it expires. The
outcome is decided once, when the last part closes: hit > void > blocked > dodged.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Set, Tuple

from . import config as C


def percent(part: float, whole: float) -> float:
    """part / whole as a percentage; 0 when there is nothing to divide by."""
    return 100.0 * part / whole if whole > 0 else 0.0


def format_duration(seconds: float) -> str:
    """MM:SS (minutes keep counting past 59)."""
    s = int(max(0.0, seconds) + 1e-6)
    return f"{s // 60:02d}:{s % 60:02d}"


class _Motion:
    __slots__ = ("active", "counted", "landed")

    def __init__(self):
        self.active = False    # limb is mid-strike
        self.counted = False   # this motion has been counted as thrown
        self.landed = False    # this motion has been counted as landed


class _PendingAttack:
    __slots__ = ("parts", "closed", "hit", "blocked", "void")

    def __init__(self, parts: int):
        self.parts = parts
        self.closed: Set[int] = set()
        self.hit = False
        self.blocked = False
        self.void = False      # player wasn't tracked, so we can't say whether it was dodged


class FightStats:
    def __init__(self, boss_name: str = "", started_at: float = 0.0):
        self.boss_name = boss_name
        self.started_at = started_at
        self.ended_at: Optional[float] = None
        self.result = ""                    # VICTORY / CHAMPION / DEFEAT once finished
        self.health_remaining = 0.0         # fraction of max HP, set when finished

        self.punches_thrown = self.punches_landed = 0
        self.kicks_thrown = self.kicks_landed = 0
        self.damage_dealt = 0.0

        self.enemy_attacks_faced = 0
        self.enemy_attacks_dodged = 0
        self.enemy_attacks_blocked = 0
        self.enemy_attacks_hit = 0
        self.damage_taken = 0.0
        self.current_dodge_streak = 0
        self.longest_dodge_streak = 0

        self._motions: Dict[Tuple[str, str], _Motion] = {}
        self._pending: Dict[int, _PendingAttack] = {}
        self._next_id = 1

    @property
    def active(self) -> bool:
        return self.ended_at is None

    # --- player offence ---------------------------------------------------------
    def _motion(self, kind: str, side: str) -> _Motion:
        return self._motions.setdefault((kind, side), _Motion())

    def _count_thrown(self, kind: str) -> None:
        if kind == "kick":
            self.kicks_thrown += 1
        else:
            self.punches_thrown += 1

    def update_limb(self, kind: str, side: str, striking: bool, sustained: bool, qualifies: bool = True) -> None:
        """Called every frame for each fist ('punch') and foot ('kick').

        striking:  a new strike can start now (fast, toward the boss, off cooldown)
        sustained: the current strike is still going (fast-ish and still toward the boss)
        qualifies: the motion looks like a real attack (kicks: foot lifted); a strike that
                   doesn't qualify yet still counts if it lands
        """
        if not self.active:
            return
        m = self._motion(kind, side)
        if m.active and not sustained:
            m.active = False
        if not m.active and striking:
            m.active, m.counted, m.landed = True, False, False
        if m.active and not m.counted and qualifies:
            m.counted = True
            self._count_thrown(kind)

    def limb_landed(self, kind: str, side: str) -> bool:
        """The current motion of this limb damaged the boss. Returns True if it counted as a new landed hit."""
        if not self.active:
            return False
        m = self._motion(kind, side)
        if not m.active:
            m.active, m.counted, m.landed = True, False, False
        if not m.counted:
            m.counted = True
            self._count_thrown(kind)
        if m.landed:
            return False
        m.landed = True
        if kind == "kick":
            self.kicks_landed += 1
        else:
            self.punches_landed += 1
        return True

    def record_damage_dealt(self, amount: float) -> None:
        if self.active and amount > 0:
            self.damage_dealt += amount

    # --- enemy attacks --------------------------------------------------------------
    def open_attack(self, parts: int = 1) -> Optional[int]:
        """Start tracking one enemy attack made of `parts` pieces. Returns its id (None once the fight is over)."""
        if not self.active:
            return None
        attack_id = self._next_id
        self._next_id += 1
        self._pending[attack_id] = _PendingAttack(max(1, parts))
        return attack_id

    def attack_hit(self, attack_id: Optional[int], damage: float = 0.0) -> None:
        """The attack damaged the player (the damage is counted even if the outcome is already decided)."""
        if not self.active:
            return
        self.damage_taken += max(0.0, damage)
        a = self._pending.get(attack_id)
        if a is not None:
            a.hit = True

    def attack_blocked(self, attack_id: Optional[int], damage: float = 0.0) -> None:
        """Blocked or parried; `damage` is any chip damage that got through."""
        if not self.active:
            return
        self.damage_taken += max(0.0, damage)
        a = self._pending.get(attack_id)
        if a is not None:
            a.blocked = True

    def void_attack(self, attack_id: Optional[int]) -> None:
        """Couldn't be judged (player out of view): it won't count as faced unless it hit."""
        a = self._pending.get(attack_id)
        if a is not None:
            a.void = True

    def close_attack(self, attack_id: Optional[int], part: int = 0) -> None:
        """One part of the attack has expired. Safe to call repeatedly; the attack resolves
        once, when every part has closed."""
        if not self.active:
            return
        a = self._pending.get(attack_id)
        if a is None or part in a.closed:
            return
        a.closed.add(part)
        if len(a.closed) < a.parts:
            return
        del self._pending[attack_id]
        self._resolve(a)

    def _resolve(self, a: _PendingAttack) -> None:
        if a.hit:
            self.enemy_attacks_faced += 1
            self.enemy_attacks_hit += 1
            self.current_dodge_streak = 0
        elif a.void:
            return
        elif a.blocked:  # not a dodge, but not a hit either: the streak carries on
            self.enemy_attacks_faced += 1
            self.enemy_attacks_blocked += 1
        else:
            self.enemy_attacks_faced += 1
            self.enemy_attacks_dodged += 1
            self.current_dodge_streak += 1
            self.longest_dodge_streak = max(self.longest_dodge_streak, self.current_dodge_streak)

    def is_open(self, attack_id: Optional[int]) -> bool:
        return attack_id in self._pending

    # --- end of fight -------------------------------------------------------------
    def finish(self, now: float, result: str, player_hp: float) -> None:
        """Freeze the stats. Attacks still in flight that already hit or were blocked (e.g. the
        blow that ended the fight) are counted; ones that could still have been dodged are dropped."""
        if not self.active:
            return
        for a in self._pending.values():
            if a.hit or (a.blocked and not a.void):
                self._resolve(a)
        self._pending.clear()
        self.ended_at = max(now, self.started_at)
        self.result = result
        self.health_remaining = max(0.0, min(1.0, player_hp / C.PLAYER_MAX_HP))

    # --- derived numbers ----------------------------------------------------------
    @property
    def punch_accuracy(self) -> float:
        return percent(self.punches_landed, self.punches_thrown)

    @property
    def kick_accuracy(self) -> float:
        return percent(self.kicks_landed, self.kicks_thrown)

    @property
    def total_hits(self) -> int:
        return self.punches_landed + self.kicks_landed

    @property
    def dodge_rate(self) -> float:
        return percent(self.enemy_attacks_dodged, self.enemy_attacks_faced)

    def duration(self, now: Optional[float] = None) -> float:
        end = self.ended_at if self.ended_at is not None else now
        return 0.0 if end is None else max(0.0, end - self.started_at)

    def calories(self, now: Optional[float] = None) -> float:
        """Rough estimate: MET x 3.5 x kg / 200 per minute, with an assumed body weight."""
        kcal = C.CALORIE_MET * 3.5 * C.CALORIE_WEIGHT_KG / 200.0 * self.duration(now) / 60.0
        return kcal if math.isfinite(kcal) and kcal > 0 else 0.0

    def calories_per_hour(self, now: Optional[float] = None) -> float:
        hours = self.duration(now) / 3600.0
        rate = self.calories(now) / hours if hours > 0 else 0.0
        return rate if math.isfinite(rate) else 0.0
