"""Fitness Score leaderboard: the best scores from won fights, saved as JSON next to the game."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from . import config as C


def clean_name(name: str) -> str:
    """Upper-case letters and digits only, at most LEADERBOARD_NAME_LEN characters."""
    return "".join(ch for ch in name.upper() if ch.isascii() and ch.isalnum())[:C.LEADERBOARD_NAME_LEN]


class Leaderboard:
    def __init__(self, path: Path, size: int = C.LEADERBOARD_SIZE):
        self.path, self.size = path, size
        self.entries: List[dict] = self._load()   # best first

    def _load(self) -> List[dict]:
        if not self.path.exists():
            raw = C.LEADERBOARD_SEED  # no saved board yet: start from the seeded one
        else:
            try:
                raw = json.loads(self.path.read_text())["entries"]
            except (OSError, ValueError, KeyError, TypeError):
                return []
        entries = []
        for e in raw if isinstance(raw, list) else []:
            try:
                entries.append({"name": clean_name(str(e["name"])) or "???", "score": int(e["score"]),
                                "boss": str(e.get("boss", "")), "rank": str(e.get("rank", "-"))})
            except (KeyError, TypeError, ValueError):
                continue  # skip a damaged entry, keep the rest
        entries.sort(key=lambda e: -e["score"])
        return entries[:self.size]

    def position(self, score: int) -> int:
        """1-based place a new score would take (behind existing equal scores)."""
        return 1 + sum(1 for e in self.entries if e["score"] >= score)

    def qualifies(self, score: int) -> bool:
        return score > 0 and self.position(score) <= self.size

    def add(self, name: str, score: int, boss: str, rank: str) -> int:
        """Insert and save. Returns the 1-based position, or 0 if it didn't make the board."""
        if not self.qualifies(score):
            return 0
        pos = self.position(score)
        self.entries.insert(pos - 1, {"name": clean_name(name) or "???", "score": int(score),
                                      "boss": boss, "rank": rank})
        del self.entries[self.size:]
        try:
            self.path.write_text(json.dumps({"entries": self.entries}, indent=2))
        except OSError as exc:
            print(f"[leaderboard] could not save: {exc}")
        return pos
