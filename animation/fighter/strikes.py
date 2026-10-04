"""Player strike instances: turns fist / foot motion into discrete punches and kicks.

A strike starts when a limb moves toward the boss faster than the punch / kick threshold and
ends when it slows below 55% of it, pulls back, or after STRIKE_MAX_TIME. One strike can land
at most once (it is "consumed" by a hit, block, dodge or staff contact), however long the fist
overlaps the body. Contact uses the limb swept from its previous position, so a fast punch
that jumps past the boss between camera frames still connects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from . import config as C
from .geometry import Capsule, Circle, Vec

STRIKE_MAX_TIME = 0.45   # a strike longer than this is over (a held arm isn't a new punch)
STRIKE_GAP = 0.10        # minimum time between strikes with the same limb


@dataclass
class Strike:
    id: int
    kind: str          # 'punch' | 'kick'
    side: str          # 'l' | 'r'
    start: float
    consumed: bool = False
    peak_speed: float = 0.0


@dataclass
class LiveStrike:
    strike: Strike
    limb: Circle       # where the fist / foot is now
    swept: Capsule     # its path since the previous frame
    vel: Vec
    speed: float


class StrikeTracker:
    def __init__(self):
        self.active: Dict[Tuple[str, str], Strike] = {}
        self.prev: Dict[Tuple[str, str], Vec] = {}
        self.ended: Dict[Tuple[str, str], float] = {}
        self.next_id = 1
        self.started = 0

    def reset(self) -> None:
        self.active.clear()
        self.prev.clear()
        self.ended.clear()

    def update(self, player, target: Vec, now: float) -> List[LiveStrike]:
        """Advance every limb's strike state; returns the strikes that may still land this frame."""
        out: List[LiveStrike] = []
        limbs = [("punch", s, c, player.fist_vel[s], player.fist_speed(s)) for s, c in player.fists().items()]
        limbs += [("kick", s, c, player.foot_vel[s], player.foot_speed(s)) for s, c in player.feet().items()]
        seen = set()
        for kind, side, circle, vel, speed in limbs:
            key = (kind, side)
            seen.add(key)
            threshold = C.KICK_SPEED_THRESHOLD if kind == "kick" else C.PUNCH_SPEED_THRESHOLD
            # Direction is judged from where the limb was, so a fist that passes the target's
            # centre within one frame still counts as moving toward it on that frame.
            prev = self.prev.get(key, circle.center)
            toward = vel[0] * (target[0] - prev[0]) + vel[1] * (target[1] - prev[1]) > 0
            st = self.active.get(key)
            if st is not None and (speed < threshold * 0.55 or not toward or now - st.start > STRIKE_MAX_TIME):
                self.active.pop(key)
                self.ended[key] = now
                st = None
            if st is None and speed >= threshold and toward and now - self.ended.get(key, -1e9) >= STRIKE_GAP:
                st = Strike(self.next_id, kind, side, now)
                self.next_id += 1
                self.started += 1
                self.active[key] = st
            self.prev[key] = circle.center
            if st is not None:
                st.peak_speed = max(st.peak_speed, speed)
                if not st.consumed:
                    out.append(LiveStrike(st, circle, Capsule(prev, circle.center, circle.radius), vel, speed))
        for key in list(self.prev):
            if key not in seen:  # limb not visible: forget its path
                self.prev.pop(key)
                if key in self.active:
                    self.active.pop(key)
                    self.ended[key] = now
        return out
