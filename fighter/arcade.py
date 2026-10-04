"""Retro arcade look: palette, pixel fonts, chunky panels, segmented bars, scanlines and the coach.

Fonts are Press Start 2P (big 8-bit type) and VT323 (readable terminal text), both SIL Open Font
License, in fonts/. If they're missing the game falls back to pygame's default font.
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, Tuple

import pygame

# --- palette: a dark cabinet with neon -----------------------------------------
INK = (12, 9, 26)          # cabinet black (slightly purple)
PANEL = (24, 18, 48)
CYAN = (41, 227, 255)
MAGENTA = (255, 46, 136)
YELLOW = (255, 212, 0)
LIME = (125, 255, 74)
ORANGE = (255, 140, 40)
RED = (255, 64, 64)
WHITE = (245, 245, 255)
DIM = (150, 146, 196)
ROW_COLORS = (YELLOW, ORANGE, RED, MAGENTA, CYAN, LIME, WHITE, DIM, DIM, DIM)  # high-score table, by place

_FONT_DIR = Path(__file__).resolve().parent.parent / "fonts"
_FILES = {"pixel": "PressStart2P-Regular.ttf", "term": "VT323-Regular.ttf"}
_cache: Dict[Tuple[str, int], pygame.font.Font] = {}


def font(kind: str, size: int) -> pygame.font.Font:
    """'pixel' (Press Start 2P) or 'term' (VT323) at `size` px; pygame's default font if the file is missing."""
    key = (kind, size)
    if key not in _cache:
        path = _FONT_DIR / _FILES[kind]
        try:
            _cache[key] = pygame.font.Font(str(path), size)
        except (OSError, FileNotFoundError):
            _cache[key] = pygame.font.Font(None, int(size * (1.9 if kind == "pixel" else 1.1)))
    return _cache[key]


def pixel(size: int) -> pygame.font.Font:
    return font("pixel", size)


def term(size: int) -> pygame.font.Font:
    return font("term", size)


def text(surf: pygame.Surface, msg: str, pos, fnt: pygame.font.Font, color=WHITE, anchor: str = "topleft",
         shadow=(0, 0, 0), depth: int = 0) -> pygame.Rect:
    """Text with a hard, offset drop shadow (no blur): the arcade look."""
    depth = depth or max(2, fnt.get_height() // 10)
    fg = fnt.render(msg, False, color)
    rect = fg.get_rect(**{anchor: (int(pos[0]), int(pos[1]))})
    if shadow is not None:
        surf.blit(fnt.render(msg, False, shadow), rect.move(depth, depth))
    surf.blit(fg, rect)
    return rect


def panel(surf: pygame.Surface, rect: pygame.Rect, border=CYAN, fill=PANEL, alpha: int = 235) -> None:
    """Chunky cabinet panel: solid fill, a thick neon border and notched (cut) corners."""
    body = pygame.Surface(rect.size, pygame.SRCALPHA)
    body.fill((*fill, alpha))
    surf.blit(body, rect)
    pygame.draw.rect(surf, border, rect, 4)
    pygame.draw.rect(surf, INK, rect.inflate(-8, -8), 2)
    n = 10
    for cx, cy, sx, sy in ((rect.left, rect.top, 1, 1), (rect.right - 1, rect.top, -1, 1),
                           (rect.left, rect.bottom - 1, 1, -1), (rect.right - 1, rect.bottom - 1, -1, -1)):
        pygame.draw.polygon(surf, INK, [(cx, cy), (cx + sx * n, cy), (cx, cy + sy * n)])
        pygame.draw.line(surf, border, (cx + sx * n, cy), (cx, cy + sy * n), 4)


def button(surf: pygame.Surface, rect: pygame.Rect, label: str, selected: bool, t: float,
           fnt: pygame.font.Font) -> None:
    """Arcade menu button: the selected one is lit yellow with a blinking arrow beside it."""
    if selected:
        pygame.draw.rect(surf, INK, rect.move(6, 6))  # hard shadow
        pygame.draw.rect(surf, YELLOW, rect)
        pygame.draw.rect(surf, WHITE, rect, 3)
        text(surf, label, rect.center, fnt, INK, "center", shadow=None)
        if int(t * 4) % 2 == 0:
            cy = rect.centery
            ax = rect.left - 22
            pygame.draw.polygon(surf, YELLOW, [(ax - 12, cy - 12), (ax + 6, cy), (ax - 12, cy + 12)])
    else:
        pygame.draw.rect(surf, PANEL, rect)
        pygame.draw.rect(surf, DIM, rect, 3)
        text(surf, label, rect.center, fnt, WHITE, "center")


def seg_bar(surf: pygame.Surface, rect: pygame.Rect, frac: float, trail: float, color, rtl: bool = False,
            segments: int = 20) -> None:
    """Segmented health bar (lit blocks, a white 'just lost' trail, dark empty blocks)."""
    pygame.draw.rect(surf, INK, rect.inflate(8, 8))
    pygame.draw.rect(surf, WHITE, rect.inflate(8, 8), 2)
    gap = 3
    seg_w = (rect.width - gap * (segments - 1)) / segments
    for i in range(segments):
        k = (segments - 1 - i) if rtl else i
        x = rect.left + i * (seg_w + gap)
        lit = (k + 0.5) / segments <= frac
        trailing = (k + 0.5) / segments <= trail
        c = color if lit else WHITE if trailing else (40, 34, 66)
        pygame.draw.rect(surf, c, (int(x), rect.top, int(math.ceil(seg_w)), rect.height))
        if lit:  # a bright top edge, like a lit LED segment
            pygame.draw.rect(surf, tuple(min(255, v + 70) for v in c), (int(x), rect.top, int(math.ceil(seg_w)), 3))


_scan_cache: Dict[Tuple[int, int], pygame.Surface] = {}


def scanlines(surf: pygame.Surface, alpha: int = 46) -> None:
    """CRT scanlines over the whole screen."""
    key = surf.get_size()
    if key not in _scan_cache:
        s = pygame.Surface(key, pygame.SRCALPHA)
        for y in range(0, key[1], 3):
            pygame.draw.line(s, (0, 0, 0, alpha), (0, y), (key[0], y))
        _scan_cache[key] = s
    surf.blit(_scan_cache[key], (0, 0))


def sticker(surf: pygame.Surface, msg: str, center, fnt: pygame.font.Font, fill=MAGENTA, color=WHITE,
            angle: float = -8.0) -> None:
    """A slapped-on sticker: a tilted label with a thick white edge."""
    label = fnt.render(msg, False, color)
    pad = 12
    tag = pygame.Surface((label.get_width() + pad * 2 + 8, label.get_height() + pad * 2 + 8), pygame.SRCALPHA)
    inner = pygame.Rect(4, 4, label.get_width() + pad * 2, label.get_height() + pad * 2)
    pygame.draw.rect(tag, WHITE, inner.inflate(8, 8))
    pygame.draw.rect(tag, fill, inner)
    tag.blit(label, (inner.left + pad, inner.top + pad))
    tag = pygame.transform.rotate(tag, angle)
    surf.blit(tag, tag.get_rect(center=(int(center[0]), int(center[1]))))


def ordinal(n: int) -> str:
    return f"{n}{'TH' if 10 <= n % 100 <= 20 else {1: 'ST', 2: 'ND', 3: 'RD'}.get(n % 10, 'TH')}"


def rainbow(t: float):
    """Cycles yellow -> magenta -> cyan (for 'NEW HIGH SCORE!')."""
    return (YELLOW, MAGENTA, CYAN, LIME)[int(t * 8) % 4]


# --- the coach: what the corner shouts after a fight --------------------------------
QUIPS = {
    "flawless": ["FLAWLESS. THE CROWD IS STILL SCREAMING!", "THAT WAS ART. SWEATY, SWEATY ART."],
    "accuracy": ["SNIPER FISTS! EVERY SHOT COUNTED.", "CLEAN HANDS. YOU MADE THAT LOOK EASY."],
    "dodging": ["SLIPPERY! THEY COULDN'T LAY A FINGER ON YOU.", "FOOTWORK LIKE THAT? CAN'T TEACH IT."],
    "combos": ["COMBO MACHINE! DON'T YOU DARE SLOW DOWN.", "BAP BAP BAP! THAT'S THE RHYTHM."],
    "activity": ["NONSTOP ENGINE. YOUR LEGS WILL FEEL THIS TOMORROW.", "YOU NEVER STOPPED MOVING. RESPECT."],
    "scrappy": ["SCRAPPY WIN! A WIN IS A WIN.", "UGLY? MAYBE. WON? ABSOLUTELY."],
    "defeat": ["SHAKE IT OFF. HANDS UP. RUN IT BACK!", "CLOSE ONE. BREATHE... AND AGAIN!",
               "EVERY CHAMP LOSES ONE. GET BACK IN THERE."],
}


def coach_line(stats) -> str:
    """A line from the coach based on how the fight went (stable for a given fight)."""
    rng = random.Random(int(stats.started_at * 1000))
    if not stats.won:
        return rng.choice(QUIPS["defeat"])
    if stats.overall_rank() == "S":
        return rng.choice(QUIPS["flawless"])
    cats = {k: v for k, v in stats.category_scores().items() if v is not None}
    best = max(cats, key=cats.get) if cats else None
    if best is None or cats[best] < 60:
        return rng.choice(QUIPS["scrappy"])
    return rng.choice(QUIPS[best])
