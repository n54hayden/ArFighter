"""Throwing stars: fly in a straight line at the player's standing head height."""
from __future__ import annotations

import math
from typing import List

import pygame

from .geometry import Circle, Vec


class ThrowingStar:
    def __init__(self, start: Vec, aim: Vec, speed: float, radius: float):
        dx, dy = aim[0] - start[0], aim[1] - start[1]
        length = math.hypot(dx, dy) or 1.0
        self.x, self.y = start
        self.vx, self.vy = dx / length * speed, dy / length * speed
        self.radius = radius
        self.angle = 0.0
        self.trail: List[Vec] = []
        self.passed_player = False  # set once it has flown past the player unharmed
        self.attack_id = None       # fight stats: one throw = one enemy attack

    @property
    def collider(self) -> Circle:
        return Circle((self.x, self.y), self.radius)

    def update(self, dt: float) -> None:
        self.trail.append((self.x, self.y))
        del self.trail[:-8]
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.angle += 1080.0 * dt

    def offscreen(self, w: int, h: int) -> bool:
        m = self.radius * 3
        return not (-m <= self.x <= w + m and -m <= self.y <= h + m)

    def draw(self, surf: pygame.Surface) -> None:
        for i, (tx, ty) in enumerate(self.trail):
            k = (i + 1) / len(self.trail)
            pygame.draw.circle(surf, (int(90 * k), int(95 * k), int(110 * k)), (int(tx), int(ty)),
                               max(1, int(self.radius * 0.5 * k)))
        outer, inner = self.radius * 1.35, self.radius * 0.42
        pts = []
        for i in range(8):  # four blades
            a = math.radians(self.angle + i * 45)
            r = outer if i % 2 == 0 else inner
            pts.append((self.x + math.cos(a) * r, self.y + math.sin(a) * r))
        pygame.draw.polygon(surf, (20, 20, 26), [(x + 2, y + 2) for x, y in pts])
        pygame.draw.polygon(surf, (200, 205, 220), pts)
        pygame.draw.polygon(surf, (90, 95, 110), pts, 2)
        pygame.draw.circle(surf, (25, 25, 30), (int(self.x), int(self.y)), max(2, int(self.radius * 0.25)))
