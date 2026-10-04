"""Hit log: every resolved strike (who, what, when, where, how much, outcome), recorded once.

Each strike carries an id (the player's strike instance, the boss's attack number, a star or
spell id). Recording the same (attacker, id) twice is refused and counted as a duplicate, so
the log doubles as a check that hit registration is exactly-once. [D] shows the latest entries;
every session is also written to logs/hits_<time>.csv.
"""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple

import pygame


@dataclass
class HitEvent:
    t: float
    attacker: str
    target: str
    attack: str
    strike_id: str
    part: str
    damage: float
    result: str


class HitLog:
    def __init__(self, folder: Optional[Path] = None):
        self.events: List[HitEvent] = []
        self._seen: Set[Tuple[str, str]] = set()
        self.duplicates = 0
        self._t0 = time.perf_counter()
        self._folder = folder
        self._file = None
        self._writer = None

    def record(self, attacker: str, target: str, attack: str, strike_id, part: str, damage: float,
               result: str) -> bool:
        key = (attacker, str(strike_id))
        if key in self._seen:
            self.duplicates += 1
            return False
        self._seen.add(key)
        ev = HitEvent(round(time.perf_counter() - self._t0, 3), attacker, target, attack, str(strike_id), part,
                      round(float(damage), 1), result)
        self.events.append(ev)
        self._write(ev)
        return True

    def _write(self, ev: HitEvent) -> None:
        if self._folder is None:
            return
        try:
            if self._file is None:
                self._folder.mkdir(parents=True, exist_ok=True)
                path = self._folder / time.strftime("hits_%Y%m%d_%H%M%S.csv")
                self._file = open(path, "w", newline="", encoding="utf-8")
                self._writer = csv.writer(self._file)
                self._writer.writerow(["time_s", "attacker", "target", "attack", "strike_id", "part", "damage", "result"])
            self._writer.writerow([ev.t, ev.attacker, ev.target, ev.attack, ev.strike_id, ev.part, ev.damage, ev.result])
            self._file.flush()
        except OSError:
            self._folder = None

    def close(self) -> None:
        if self._file:
            self._file.close()

    def draw(self, surf: pygame.Surface, font: pygame.font.Font, x: int, y: int, n: int = 9) -> None:
        rows = self.events[-n:]
        w, h = 560, 20 * (len(rows) + 1) + 8
        panel = pygame.Surface((w, h), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 150))
        surf.blit(panel, (x, y))
        head = f"HIT LOG  {len(self.events)} events   duplicates rejected: {self.duplicates}"
        surf.blit(font.render(head, True, (255, 230, 150)), (x + 8, y + 4))
        for i, e in enumerate(rows):
            color = (255, 140, 140) if e.target == "PLAYER" and e.damage > 0 else (200, 230, 255)
            line = f"{e.t:7.2f}s  {e.attacker[:10]:10s} -> {e.target[:8]:8s} {e.attack[:13]:13s} #{e.strike_id:<4s} " \
                   f"{e.part[:5]:5s} {e.damage:5.1f}  {e.result}"
            surf.blit(font.render(line, True, color), (x + 8, y + 24 + i * 20))
