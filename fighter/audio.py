"""Fight music: plays the first audio file found in the music folder."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pygame

from . import config as C

AUDIO_EXTENSIONS = (".mp3", ".ogg", ".wav")


class Music:
    def __init__(self, folder: Path):
        self.track: Optional[Path] = None
        self.available = False
        self.muted = False
        self._paused = False
        self._playing = False

        tracks = sorted(p for p in folder.glob("*") if p.suffix.lower() in AUDIO_EXTENSIONS) if folder.is_dir() else []
        if not tracks:
            print(f"[music] no audio file in {folder}, playing without music")
            return
        self.track = tracks[0]
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load(str(self.track))
            pygame.mixer.music.set_volume(C.MUSIC_VOLUME)
            self.available = True
        except pygame.error as exc:  # no audio device, unsupported file, ...
            print(f"[music] could not load {self.track.name}: {exc}")

    def start(self) -> None:
        """Play from the beginning, looping until stopped."""
        if not self.available:
            return
        pygame.mixer.music.play(loops=-1, fade_ms=C.MUSIC_FADE_IN_MS)
        self._paused = False
        self._playing = True

    def fade_out(self) -> None:
        if self.available and self._playing:
            pygame.mixer.music.fadeout(C.MUSIC_FADE_OUT_MS)
            self._playing = False

    def stop(self) -> None:
        if self.available:
            pygame.mixer.music.stop()
            self._playing = False

    def set_paused(self, paused: bool) -> None:
        if not self.available or not self._playing or paused == self._paused:
            return
        if paused:
            pygame.mixer.music.pause()
        else:
            pygame.mixer.music.unpause()
        self._paused = paused

    def toggle_mute(self) -> None:
        if not self.available:
            return
        self.muted = not self.muted
        pygame.mixer.music.set_volume(0.0 if self.muted else C.MUSIC_VOLUME)
