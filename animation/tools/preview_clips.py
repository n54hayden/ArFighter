#!/usr/bin/env python3
"""Render contact sheets of the baked clips (side view, as in the game) for visual checks.

    python tools/preview_clips.py [clip ...] [--out DIR] [--boss N]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pygame  # noqa: E402

from fighter import config as C  # noqa: E402
from fighter.character import CharacterData  # noqa: E402
from fighter.render3d import CharacterRenderer, Look, placement, _rgb  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="*")
    ap.add_argument("--out", type=Path, default=ROOT / "tools" / "previews")
    ap.add_argument("--boss", type=int, default=0)
    ap.add_argument("--frames", type=int, default=6)
    args = ap.parse_args()
    pygame.init()
    pygame.display.set_mode((1, 1))
    data = CharacterData(ROOT / "assets" / "character" / "fighter.npz")
    cell_w, cell_h = 240, 300
    ren = CharacterRenderer(data, (cell_w, cell_h))
    boss = C.BOSSES[args.boss]
    look = Look(_rgb(boss["rim"]), _rgb(boss["rim"]), gear=tuple(k for k in ("staff", "sword", "headband",
                                                                             "headdress", "robe") if boss.get(k)))
    font = pygame.font.Font(None, 22)
    args.out.mkdir(parents=True, exist_ok=True)
    names = args.clips or list(data.clips)
    s = cell_h * 0.78 / data.height
    for name in names:
        clip = data.clips[name]
        n = args.frames
        sheet = pygame.Surface((cell_w * n, cell_h + 24))
        sheet.fill((70, 72, 80))
        meta = clip.meta
        for i in range(n):
            t = clip.duration * i / max(1, n - 1)
            rot, hips = clip.sample(t)
            M = placement(cell_w * 0.45, cell_h * 0.92, s, 1) @ data.above
            world = data.rig.world(rot, hips, root=M)
            img = ren.render(world, (0, 0, cell_w, cell_h), look,
                             grip_angle=0.0 if meta.get("grip", 1.0) > 0 else np.pi)
            cell = pygame.Surface((cell_w, cell_h))
            cell.fill((70, 72, 80))
            pygame.draw.line(cell, (110, 112, 120), (0, int(cell_h * 0.92)), (cell_w, int(cell_h * 0.92)), 1)
            cell.blit(img, (0, 0))
            active = "limb" in meta and meta.get("active_start", 9) <= t <= meta.get("active_end", -1)
            if active:
                pygame.draw.rect(cell, (255, 80, 60), cell.get_rect(), 3)
            cell.blit(font.render(f"{t:.2f}s", True, (230, 230, 230)), (6, 6))
            sheet.blit(cell, (i * cell_w, 24))
        label = f"{name}  ({clip.duration:.2f}s{', loop' if clip.loop else ''})"
        if "limb" in meta:
            label += f"  limb {['hand.L', 'hand.R', 'foot.L', 'foot.R'][int(meta['limb'])]}  contact {meta['contact']:.2f}s"
        sheet.blit(font.render(label, True, (255, 255, 255)), (6, 4))
        pygame.image.save(sheet, str(args.out / f"{name}.png"))
    print(f"wrote {len(names)} sheets to {args.out}")


if __name__ == "__main__":
    main()
