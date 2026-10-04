"""Hit sparks, smoke, floating text, screen shake and screen flash."""
from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pygame

from . import config as C
from .geometry import Vec


def render_outlined(font: pygame.font.Font, msg: str, color, outline: int = 0) -> pygame.Surface:
    """Text with a thick black outline and drop shadow: readable over any camera background."""
    outline = outline or max(2, font.get_height() // 12)
    fg = font.render(msg, True, color)
    black = font.render(msg, True, (0, 0, 0))
    pad = outline + 4
    surf = pygame.Surface((fg.get_width() + pad * 2, fg.get_height() + pad * 2), pygame.SRCALPHA)
    surf.blit(black, (pad + 4, pad + 4))  # drop shadow
    for i in range(16):
        a = math.tau * i / 16
        surf.blit(black, (pad + round(math.cos(a) * outline), pad + round(math.sin(a) * outline)))
    surf.blit(fg, (pad, pad))
    return surf


def make_vignette(size: Tuple[int, int], color) -> pygame.Surface:
    """Transparent in the middle, `color` at the edges and corners (the low-health pulse)."""
    w, h = size
    surf = pygame.Surface(size, pygame.SRCALPHA)
    surf.fill((*color, 0))
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)[:, None]
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)[None, :]
    d = np.sqrt(x * x + y * y) / math.sqrt(2.0)  # 0 in the centre, 1 in the corners
    alpha = pygame.surfarray.pixels_alpha(surf)
    alpha[:] = (np.clip((d - 0.3) / 0.7, 0.0, 1.0) ** 1.5 * 255).astype(np.uint8)
    del alpha  # unlock the surface
    return surf


class Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "color", "size", "gravity")

    def __init__(self, x, y, vx, vy, life, color, size, gravity):
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.life = self.max_life = life
        self.color, self.size, self.gravity = color, size, gravity


class FloatingText:
    __slots__ = ("surf", "x", "y", "life", "max_life")

    def __init__(self, surf, x, y, life):
        self.surf, self.x, self.y = surf, x, y
        self.life = self.max_life = life


class Effects:
    def __init__(self, screen_size: Tuple[int, int]):
        self.particles: List[Particle] = []
        self.texts: List[FloatingText] = []
        self.shake_mag = 0.0
        self.flash_color = (0, 0, 0)
        self.flash_alpha = 0.0
        self._overlay = pygame.Surface(screen_size)
        self._fonts: Dict[int, pygame.font.Font] = {}

    def font(self, size: int) -> pygame.font.Font:
        if size not in self._fonts:
            self._fonts[size] = pygame.font.Font(None, size)
        return self._fonts[size]

    def clear(self) -> None:
        self.particles.clear()
        self.texts.clear()
        self.shake_mag = 0.0
        self.flash_alpha = 0.0

    # --- spawning ------------------------------------------------------------
    def burst(self, pos: Vec, color, count: int = 18, speed: float = 420.0, size=(2.0, 5.0),
              life=(0.25, 0.55), gravity: float = 900.0) -> None:
        for _ in range(count):
            a = random.uniform(0, math.tau)
            v = random.uniform(0.3, 1.0) * speed
            self.particles.append(Particle(pos[0], pos[1], math.cos(a) * v, math.sin(a) * v,
                                           random.uniform(*life), color, random.uniform(*size), gravity))

    def smoke(self, points: Iterable[Vec], color=(14, 10, 22), per_point: int = 6) -> None:
        for x, y in points:
            for _ in range(per_point):
                self.particles.append(Particle(
                    x + random.uniform(-12, 12), y + random.uniform(-12, 12),
                    random.uniform(-60, 60), random.uniform(-160, -40),
                    random.uniform(0.6, 1.4), color, random.uniform(6, 14), -40.0))

    def text(self, msg: str, pos: Vec, color, size: int = 44, life: float = 0.9) -> None:
        """Floating text, scaled up and outlined so it reads from across the room (POPUP_TEXT_* in config)."""
        surf = render_outlined(self.font(int(size * C.POPUP_TEXT_SCALE)), msg, color)
        w, h = self._overlay.get_size()
        if surf.get_width() > w * 0.95:  # long messages shrink to fit the screen
            k = w * 0.95 / surf.get_width()
            surf = pygame.transform.smoothscale(surf, (int(surf.get_width() * k), int(surf.get_height() * k)))
        # keep it fully on screen (it also drifts upward while it fades)
        x = min(max(pos[0], surf.get_width() / 2 + 8), w - surf.get_width() / 2 - 8)
        y = min(max(pos[1], surf.get_height() / 2 + 8 + 60 * life), h - surf.get_height() / 2 - 8)
        self.texts.append(FloatingText(surf, x, y, life * C.POPUP_TEXT_LIFE_SCALE))

    def shake(self, magnitude: float) -> None:
        self.shake_mag = max(self.shake_mag, magnitude)

    def flash(self, color, alpha: float) -> None:
        self.flash_color = color
        self.flash_alpha = max(self.flash_alpha, alpha)

    # --- frame ---------------------------------------------------------------
    def update(self, dt: float) -> None:
        alive = []
        for p in self.particles:
            p.life -= dt
            if p.life > 0:
                p.vy += p.gravity * dt
                p.x += p.vx * dt
                p.y += p.vy * dt
                alive.append(p)
        self.particles = alive
        for t in self.texts:
            t.life -= dt
            t.y -= 60 * dt
        self.texts = [t for t in self.texts if t.life > 0]
        self.shake_mag *= math.exp(-10.0 * dt)
        self.flash_alpha = max(0.0, self.flash_alpha - 450.0 * dt)

    def shake_offset(self) -> Tuple[int, int]:
        if self.shake_mag < 0.5:
            return 0, 0
        m = self.shake_mag
        return int(random.uniform(-m, m)), int(random.uniform(-m, m))

    def draw_world(self, surf: pygame.Surface) -> None:
        for p in self.particles:
            k = p.life / p.max_life
            pygame.draw.circle(surf, p.color, (int(p.x), int(p.y)), max(1, int(p.size * (0.4 + 0.6 * k))))
        for t in self.texts:
            t.surf.set_alpha(int(255 * min(1.0, t.life / t.max_life * 2.0)))
            surf.blit(t.surf, t.surf.get_rect(center=(int(t.x), int(t.y))))

    def draw_overlay(self, surf: pygame.Surface) -> None:
        if self.flash_alpha <= 1:
            return
        self._overlay.fill(self.flash_color)
        self._overlay.set_alpha(int(min(255, self.flash_alpha)))
        surf.blit(self._overlay, (0, 0))
