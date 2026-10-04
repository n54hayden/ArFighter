"""Mage spell hazards: floor zones (meteor, icicles) and travelling fire walls.

Fairness rules baked in here:
  * Every hazard is placed from a snapshot of the player taken when the cast starts,
    and never moves afterwards (zones don't chase you).
  * Floor zones only care about horizontal position: you dodge by stepping sideways.
  * Icicle zones sit on a grid with guaranteed safe gaps between them.
"""
from __future__ import annotations

import math
import random
from typing import List

import pygame

from . import config as C
from .geometry import Capsule, Circle


class GroundStrike:
    """A locked danger zone on the floor; something falls on it after the warning."""

    def __init__(self, kind: str, x: float, radius: float, ground_y: float, warning: float,
                 fall_time: float, damage: float):
        self.kind = kind            # 'meteor' | 'icicle'
        self.x, self.radius, self.ground_y = x, radius, ground_y
        self.warning, self.fall_time, self.damage = warning, fall_time, damage
        self.t = 0.0
        self.landed = False
        self.after = 0.0            # time since impact (scorch / shards fade out)
        self.attack_id, self.attack_part = None, 0  # fight stats: which spell cast this belongs to

    @property
    def progress(self) -> float:
        return min(1.0, self.t / self.warning)

    def update(self, dt: float) -> bool:
        """Advance; returns True on the frame it lands."""
        if self.landed:
            self.after += dt
            return False
        self.t += dt
        if self.t >= self.warning:
            self.landed = True
            return True
        return False

    @property
    def finished(self) -> bool:
        return self.landed and self.after > 0.8

    def covers(self, x: float, half_width: float) -> bool:
        return abs(x - self.x) <= self.radius + half_width

    def draw_floor(self, surf: pygame.Surface, now: float) -> None:
        """Warning marker (under the bosses) or scorch after impact."""
        w = int(self.radius * 2)
        h = max(14, int(self.radius * 0.6))
        rect = pygame.Rect(0, 0, w, h)
        rect.center = (int(self.x), int(self.ground_y))
        meteor = self.kind == "meteor"
        if self.landed:
            fade = max(0.0, 1.0 - self.after / 0.8)
            col = (int(70 * fade), int(30 * fade), int(10 * fade)) if meteor else (int(90 * fade), int(140 * fade), int(170 * fade))
            pygame.draw.ellipse(surf, col, rect)
            return
        base = (255, 70, 30) if meteor else (110, 210, 255)
        pulse = 0.6 + 0.4 * math.sin(now * (14 if self.progress > 0.7 else 8))
        # A beam of light above the zone: easy to spot from across the room, even at the screen edge.
        beam = pygame.Surface((w, int(self.ground_y)), pygame.SRCALPHA)
        for i in range(0, beam.get_height(), 6):
            a = int((20 + 45 * self.progress) * (i / beam.get_height()) ** 0.7 * (0.7 + 0.3 * pulse))
            pygame.draw.line(beam, (*base, a), (0, i), (w, i), 6)
        surf.blit(beam, (rect.left, 0))
        pygame.draw.line(surf, base, (rect.left, 0), (rect.left, rect.centery), 1)
        pygame.draw.line(surf, base, (rect.right, 0), (rect.right, rect.centery), 1)
        overlay = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.ellipse(overlay, (*base, int(60 + 50 * self.progress)), overlay.get_rect())
        # inner ellipse grows to fill the zone as impact approaches
        inner = overlay.get_rect().inflate(-int(w * (1 - self.progress)), -int(h * (1 - self.progress)))
        pygame.draw.ellipse(overlay, (*base, 120), inner)
        surf.blit(overlay, rect)
        pygame.draw.ellipse(surf, tuple(int(c * pulse) for c in base), rect, 3)

    def draw_falling(self, surf: pygame.Surface, top_y: float) -> None:
        if self.landed:
            return
        t_left = self.warning - self.t
        if t_left > self.fall_time:
            return
        k = 1.0 - t_left / self.fall_time  # 0 -> 1 while falling
        if self.kind == "meteor":
            sx = self.x + self.radius * 1.5  # comes in at an angle
            x = sx + (self.x - sx) * k
            y = top_y + (self.ground_y - top_y) * k
            r = int(self.radius * 0.55)
            for i in range(6):  # fiery trail
                tk = max(0.0, k - i * 0.05)
                tx, ty = sx + (self.x - sx) * tk, top_y + (self.ground_y - top_y) * tk
                pygame.draw.circle(surf, (255, 120 + i * 15, 40), (int(tx), int(ty)), max(2, int(r * (1 - i * 0.14))))
            pygame.draw.circle(surf, (90, 45, 30), (int(x), int(y)), r)
            pygame.draw.circle(surf, (255, 170, 60), (int(x), int(y)), r, 4)
        else:
            y = top_y + (self.ground_y - top_y) * k
            L, W = self.radius * 1.6, self.radius * 0.35
            pts = [(self.x, y), (self.x - W, y - L), (self.x + W, y - L)]
            pygame.draw.polygon(surf, (200, 240, 255), pts)
            pygame.draw.polygon(surf, (90, 170, 220), pts, 2)


def make_meteor(snapshot: dict) -> List[GroundStrike]:
    u = snapshot["unit"]
    return [GroundStrike("meteor", snapshot["x"], C.METEOR_RADIUS * u, snapshot["ground_y"],
                         C.METEOR_WARNING, C.METEOR_FALL_TIME, C.METEOR_DAMAGE)]


def make_icicles(snapshot: dict, screen_w: int) -> List[GroundStrike]:
    """A dense storm of small zones around the player with one guaranteed safe gap a step away.

    The spot the player stood on is always covered (they have to move), the gap is wide
    enough to stand in, and everything outside the storm window is safe too.
    """
    u, px = snapshot["unit"], snapshot["x"]
    r = C.ICICLE_RADIUS * u
    lo, hi = screen_w * 0.06 + r, screen_w * 0.94 - r
    # Safe gap: on whichever side has room, a short step from the player.
    offset = random.uniform(*C.ICICLE_GAP_OFFSET) * u
    sides = [sd for sd in (-1, 1) if lo <= px + sd * (offset + C.ICICLE_SAFE_GAP * u / 2) <= hi + r]
    side = random.choice(sides) if sides else (1 if px < screen_w / 2 else -1)
    gap_c = px + side * offset
    gap_lo, gap_hi = gap_c - C.ICICLE_SAFE_GAP * u / 2, gap_c + C.ICICLE_SAFE_GAP * u / 2

    spacing = C.ICICLE_SPACING * u
    half = min(C.ICICLE_AREA * u, C.ICICLE_MAX_FLOOR * (hi - lo) / 2)
    w_lo, w_hi = px - half, px + half
    if w_lo < lo:  # slide the storm window off the wall instead of cutting it in half
        w_lo, w_hi = lo, lo + 2 * half
    elif w_hi > hi:
        w_lo, w_hi = hi - 2 * half, hi
    k_lo, k_hi = math.floor((w_lo - px) / spacing), math.ceil((w_hi - px) / spacing)
    centres = [px + k * spacing for k in range(k_lo, k_hi + 1)]
    centres = [x for x in centres if max(lo, w_lo) <= x <= min(hi, w_hi) and (x + r <= gap_lo or x - r >= gap_hi)]
    centres.sort(key=lambda x: abs(x - px))           # keep the storm centred on the player
    centres = centres[:random.randint(*C.ICICLE_COUNT)]
    if not any(abs(x - px) < 1e-6 for x in centres):  # always cover the spot they stood on
        centres[-1:] = [px]
    random.shuffle(centres)
    return [GroundStrike("icicle", x, r, snapshot["ground_y"], C.ICICLE_WARNING + i * C.ICICLE_STAGGER,
                         C.ICICLE_FALL_TIME, C.ICICLE_DAMAGE) for i, x in enumerate(centres)]


class FireWall:
    """A horizontal wall of fire that sweeps across the floor (low: jump) or at head height (high: duck)."""

    def __init__(self, kind: str, origin_x: float, direction: int, y_top: float, y_bottom: float,
                 width: float, speed: float, screen_w: int):
        self.kind = kind            # 'low' | 'high'
        self.direction = direction  # +1 travels right, -1 left
        self.origin_x = origin_x
        self.x = origin_x           # centre of the wall
        self.y_top, self.y_bottom = y_top, y_bottom
        self.width, self.speed, self.screen_w = width, speed, screen_w
        self.t = 0.0
        self.resolved = False       # hit or dodged (each wall affects the player once)
        self.window_open = False    # wall has reached the player's body
        self.evaded = False         # player was airborne / ducked at some point in the window
        self.attack_id, self.attack_part = None, 0  # fight stats: which spell cast this belongs to

    @property
    def launched(self) -> bool:
        return self.t >= C.FIRE_WALL_TELEGRAPH

    def update(self, dt: float) -> None:
        self.t += dt
        if self.launched:
            self.x += self.direction * self.speed * dt

    @property
    def finished(self) -> bool:
        return self.launched and (self.x < -self.width or self.x > self.screen_w + self.width)

    def rect(self) -> pygame.Rect:
        return pygame.Rect(int(self.x - self.width / 2), int(self.y_top), int(self.width),
                           int(self.y_bottom - self.y_top))

    def draw(self, surf: pygame.Surface, now: float) -> None:
        y0, y1 = int(self.y_top), int(self.y_bottom)
        x_end = self.screen_w if self.direction > 0 else 0
        low = self.kind == "low"
        if not self.launched:
            # Telegraph: the band it will sweep through glows along its path.
            k = self.t / C.FIRE_WALL_TELEGRAPH
            flick = 0.55 + 0.45 * math.sin(now * (20 if k > 0.6 else 10))
            x0, x1 = sorted((int(self.origin_x), x_end))
            band = pygame.Surface((max(1, x1 - x0), max(1, y1 - y0)), pygame.SRCALPHA)
            col = (255, 110, 30) if low else (170, 110, 255)
            band.fill((*col, int(35 + 45 * k)))
            surf.blit(band, (x0, y0))
            edge = tuple(int(c * flick) for c in col)
            step = 28
            for xs in range(x0, x1, step * 2):  # dashed edges
                pygame.draw.line(surf, edge, (xs, y0), (min(x1, xs + step), y0), 3)
                if not low:  # the high wall's bottom edge is what you duck under
                    pygame.draw.line(surf, edge, (xs, y1), (min(x1, xs + step), y1), 3)
            return
        r = self.rect()
        hot, mid, cool = ((255, 240, 160), (255, 140, 30), (210, 50, 10)) if low else \
            ((240, 235, 255), (175, 120, 255), (90, 45, 200))
        # Flames lick upward past the band (low wall) or flicker above and below it (high wall).
        pad = int(r.height * 0.45)
        flames = pygame.Surface((r.width, r.height + pad * 2), pygame.SRCALPHA)
        body = pygame.Rect(0, pad, r.width, r.height)
        pygame.draw.rect(flames, (*cool, 170), body, border_radius=int(r.width * 0.3))
        for i in range(9):
            fx = (i + 0.5) / 9 * r.width
            wob = abs(math.sin(now * 11 + i * 1.9))
            fh = r.height * (0.7 + 0.5 * wob)
            tongue = pygame.Rect(int(fx - r.width * 0.13), int(pad + r.height - fh), int(r.width * 0.26), int(fh))
            pygame.draw.ellipse(flames, (*mid, 190), tongue)
            pygame.draw.ellipse(flames, (*hot, 150), tongue.inflate(-tongue.width * 0.5, -tongue.height * 0.45))
            if not low:  # high wall also drips flame downward
                drip = pygame.Rect(tongue.x, pad, tongue.width, int(fh * 0.8))
                pygame.draw.ellipse(flames, (*mid, 140), drip)
        surf.blit(flames, (r.x, r.y - pad))


def make_fire_wall(kind: str, snapshot: dict, origin_x: float, ground_y: float, boss_height: float,
                   screen_w: int) -> FireWall:
    u = snapshot["unit"]
    direction = 1 if snapshot["x"] > origin_x else -1
    # Even at point-blank range the wall starts far enough away to be seen coming.
    lead = max(abs(snapshot["x"] - origin_x), C.FIRE_WALL_MIN_LEAD * boss_height)
    origin_x = snapshot["x"] - direction * lead
    if kind == "low":
        y_top, y_bottom = ground_y - C.FIRE_WALL_LOW_HEIGHT * boss_height, ground_y
    else:
        y_bottom = snapshot["head_y"] + C.FIRE_WALL_HIGH_CLEARANCE * u
        y_top = y_bottom - C.FIRE_WALL_HIGH_THICKNESS * u
    return FireWall(kind, origin_x, direction, y_top, y_bottom, C.FIRE_WALL_WIDTH * boss_height,
                    C.FIRE_WALL_SPEED * boss_height, screen_w)


def circle_hits_rect(c: Circle, r: pygame.Rect) -> bool:
    cx = max(r.left, min(c.center[0], r.right))
    cy = max(r.top, min(c.center[1], r.bottom))
    return math.hypot(c.center[0] - cx, c.center[1] - cy) <= c.radius


def capsule_hits_rect(cap: Capsule, r: pygame.Rect) -> bool:
    for i in range(7):  # sample the capsule as a chain of circles
        t = i / 6
        p = (cap.a[0] + (cap.b[0] - cap.a[0]) * t, cap.a[1] + (cap.b[1] - cap.a[1]) * t)
        if circle_hits_rect(Circle(p, cap.radius), r):
            return True
    return False
