"""Music: menu, fight and optional per-boss tracks, each the first audio file found in its folder."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pygame

from . import config as C

AUDIO_EXTENSIONS = (".mp3", ".ogg", ".wav")


class Music:
    def __init__(self, folders: Dict[str, Path], optional: Optional[Dict[str, Path]] = None):
        """folders: track name ('fight', 'menu', ...) -> folder holding that track. Subfolders are not searched.
        optional: more tracks (e.g. per-boss themes) that are skipped quietly when their folder is empty."""
        self.tracks: Dict[str, Path] = {}
        self.available = False
        self.muted = False
        self._paused = False
        self._playing = False
        self._current: Optional[str] = None  # name of the track loaded in the mixer

        optional = optional or {}
        for name, folder in {**folders, **optional}.items():
            files = sorted(p for p in folder.glob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS) \
                if folder.is_dir() else []
            if not files:
                if name not in optional:
                    print(f"[music] no audio file in {folder}, no {name} music")
                continue
            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                pygame.mixer.music.load(str(files[0]))  # check it loads; play() reloads it when needed
                self.tracks[name] = files[0]
                self.available = True
            except pygame.error as exc:  # no audio device, unsupported file, ...
                print(f"[music] could not load {files[0].name}: {exc}")
        self._current = None

    def play(self, name: str, fallback: Optional[str] = None) -> None:
        """Loop the named track (or `fallback` when that one doesn't exist) from the beginning.
        If it's already playing, leave it alone."""
        if name not in self.tracks and fallback is not None:
            name = fallback
        track = self.tracks.get(name)
        if track is None:
            self.stop()
            return
        if self._current == name and self._playing:
            self.set_paused(False)
            return
        try:
            if self._current != name:
                pygame.mixer.music.load(str(track))
                self._current = name
            pygame.mixer.music.set_volume(0.0 if self.muted else C.MUSIC_VOLUME)
            pygame.mixer.music.play(loops=-1, fade_ms=C.MUSIC_FADE_IN_MS)
        except pygame.error as exc:
            print(f"[music] could not play {track.name}: {exc}")
            return
        self._paused = False
        self._playing = True

    def start(self) -> None:
        """Fight music."""
        self.play("fight")

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
