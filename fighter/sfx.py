"""Sound effects: fight bell, punch impacts, star throws, victory cheer and defeat groan.

Files are found by keyword in their names (see SFX_KEYWORDS in config), so they can
be renamed or replaced freely. A file holding several punches back to back is split
on its silent gaps into separate hits, and a random one plays each time (same for throws).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pygame

from . import config as C

AUDIO_EXTENSIONS = (".mp3", ".ogg", ".wav")
WINDOW_S = 0.01          # envelope resolution
SILENCE = 0.05           # fraction of peak loudness treated as silence
MERGE_GAP_S = 0.15       # quieter gaps shorter than this stay inside one hit
MIN_HIT_S = 0.03
TAIL_S = 0.08            # keep a little decay after each hit
FADE_S = 0.02            # fade the cut edges to avoid clicks


def _samples(sound: pygame.mixer.Sound) -> np.ndarray:
    a = pygame.sndarray.array(sound)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def _envelope(a: np.ndarray, rate: int):
    win = max(1, int(rate * WINDOW_S))
    mono = np.abs(a.astype(np.float32)).mean(axis=1)
    n = len(mono) // win
    return mono[: n * win].reshape(n, win).max(axis=1), win


def _make(a: np.ndarray, rate: int) -> pygame.mixer.Sound:
    a = a.astype(np.float32)
    fade = min(len(a) // 4, int(rate * FADE_S))
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)[:, None]
        a[:fade] *= ramp
        a[-fade:] *= ramp[::-1]
    return pygame.sndarray.make_sound(np.ascontiguousarray(a.astype(np.int16)))


def trim_leading_silence(sound: pygame.mixer.Sound, rate: int) -> pygame.mixer.Sound:
    a = _samples(sound)
    env, win = _envelope(a, rate)
    if not len(env) or env.max() <= 0:
        return sound
    start = int(np.argmax(env > env.max() * SILENCE)) * win
    return _make(a[start:], rate) if start > 0 else sound


def split_hits(sound: pygame.mixer.Sound, rate: int) -> List[pygame.mixer.Sound]:
    """Cut a recording of several impacts into one Sound per impact."""
    a = _samples(sound)
    env, win = _envelope(a, rate)
    if not len(env) or env.max() <= 0:
        return [sound]
    loud = env > env.max() * SILENCE
    runs, start = [], None
    for i, on in enumerate(loud):
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append([start, i])
            start = None
    if start is not None:
        runs.append([start, len(loud)])

    merged = []
    for run in runs:
        if merged and (run[0] - merged[-1][1]) * WINDOW_S < MERGE_GAP_S:
            merged[-1][1] = run[1]
        else:
            merged.append(run)

    hits = []
    tail = int(TAIL_S / WINDOW_S)
    for s, e in merged:
        if (e - s) * WINDOW_S < MIN_HIT_S:
            continue
        hits.append(_make(a[s * win: min(len(a), (e + tail) * win)], rate))
    return hits or [sound]


class SoundEffects:
    def __init__(self, folder: Path):
        self.available = False
        self.muted = False
        self.bell: Optional[pygame.mixer.Sound] = None
        self.cheer: Optional[pygame.mixer.Sound] = None
        self.groan: Optional[pygame.mixer.Sound] = None
        self.punches: List[pygame.mixer.Sound] = []
        self.throws: List[pygame.mixer.Sound] = []
        self._last: Dict[str, int] = {}

        files = sorted(p for p in folder.glob("*") if p.suffix.lower() in AUDIO_EXTENSIONS) if folder.is_dir() else []
        if not files:
            print(f"[sfx] no sound files in {folder}, playing without sound effects")
            return
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.set_num_channels(16)
            rate = pygame.mixer.get_init()[0]
            found: Dict[str, Path] = {}
            for role, keywords in C.SFX_KEYWORDS.items():
                found_path = next((p for p in files if p not in found.values()
                                   and any(k in p.name.lower() for k in keywords)), None)
                if found_path is None:
                    print(f"[sfx] no {role} sound found (looked for {', '.join(keywords)} in the file name)")
                else:
                    found[role] = found_path
            if "bell" in found:
                self.bell = trim_leading_silence(pygame.mixer.Sound(str(found["bell"])), rate)
            if "cheer" in found:
                self.cheer = trim_leading_silence(pygame.mixer.Sound(str(found["cheer"])), rate)
            if "groan" in found:
                self.groan = trim_leading_silence(pygame.mixer.Sound(str(found["groan"])), rate)
            if "punch" in found:
                self.punches = split_hits(pygame.mixer.Sound(str(found["punch"])), rate)
            if "throw" in found:
                self.throws = split_hits(pygame.mixer.Sound(str(found["throw"])), rate)
            self.available = True
        except pygame.error as exc:  # no audio device, unsupported file, ...
            print(f"[sfx] could not load sound effects: {exc}")

    def _play(self, sound: Optional[pygame.mixer.Sound], volume: float) -> None:
        if not self.available or self.muted or sound is None:
            return
        channel = sound.play()
        if channel is not None:
            channel.set_volume(max(0.0, min(1.0, volume * C.SFX_VOLUME)))

    def bell_ring(self) -> None:
        self._play(self.bell, 1.0)

    def _play_random(self, key: str, sounds: List[pygame.mixer.Sound], volume: float) -> None:
        if not sounds:
            return
        i = random.randrange(len(sounds))
        if i == self._last.get(key) and len(sounds) > 1:  # avoid the same sound twice in a row
            i = (i + 1) % len(sounds)
        self._last[key] = i
        self._play(sounds[i], volume)

    def punch(self, volume: float = 1.0) -> None:
        self._play_random("punch", self.punches, volume)

    def throw(self, volume: float = 1.0) -> None:
        self._play_random("throw", self.throws, volume)

    def cheer_crowd(self) -> None:
        self._play(self.cheer, 1.0)

    def groan_crowd(self) -> None:
        self._play(self.groan, 1.0)

    def stop(self) -> None:
        if self.available:
            pygame.mixer.stop()
