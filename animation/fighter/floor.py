"""Floor mapping: perspective floor, walls, AR floor grid and contact shadows.

The shadow fights on a single "lane": the floor row where you stood during calibration, at
your calibrated size. It only moves left and right along it (2D movement in a 2.5D scene).

Perspective. With a roughly level camera, a point on the floor at scale k (k = 1 where you
stood during calibration, smaller further away, k ~ 1 / distance) appears at

    y = horizon + (y0 - horizon) * k

where y0 is the calibrated floor row. The horizon starts at the image centre and is refined
from your feet while you move toward / away from the camera. It is used for your jump
detection, the Mage's floor zones and contact shadows; the shadow's lane is always at k = 1.

Walls. A SegFormer model (ADE20K classes) marks which pixels are floor. Along the lane, the
floor ends where a wall or a piece of furniture starts: those are the lane's left/right
limits. If the lane row itself is hidden (e.g. below the image), the floor's left and right
edges from the lowest visible rows are extended down to it (a side wall's base is a straight
line on the floor). The mask is averaged over a few frames so one bad frame can't move a wall.
"""
from __future__ import annotations

import math
import threading
import urllib.request
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
import pygame

from . import config as C

try:  # imported up front: importing inside the mapper thread would stall rendering
    import onnxruntime as ort
except ImportError:  # the game still runs, bounded by the screen edges
    ort = None

Vec = Tuple[float, float]


def _run_around(row: np.ndarray, c: int) -> Tuple[int, int]:
    """First / last column of the run of True cells containing column c (or the nearest run)."""
    if not row[c]:
        cols = np.flatnonzero(row)
        c = int(cols[np.argmin(np.abs(cols - c))])
    off = np.flatnonzero(~row[c::-1])
    left = c - (int(off[0]) - 1) if off.size else 0
    off = np.flatnonzero(~row[c:])
    right = c + (int(off[0]) - 1) if off.size else len(row) - 1
    return left, right


class FloorModel:
    MASK_SCALE = 4  # floor mask resolution = screen size / MASK_SCALE

    def __init__(self, size: Tuple[int, int]):
        self.sw, self.sh = size
        self.cx = self.sw * 0.5
        self.focal = (self.sw * 0.5) / math.tan(math.radians(C.CAMERA_HFOV_DEG * 0.5))
        self._shadow_cache: Dict[Tuple[int, int], pygame.Surface] = {}
        self.reset()

    def reset(self) -> None:
        self.horizon = self.sh * C.HORIZON_Y_FRAC
        self.y0 = float(self.sh - 10)        # floor row of the lane (k = 1)
        self.H0 = self.sh * 0.75             # your height in pixels at k = 1
        self.anchor_x = self.sw * 0.5        # where your feet were (the floor you stand on)
        self.feet_visible = False
        self.floor_mask: Optional[np.ndarray] = None  # bool (sh / 4, sw / 4), averaged over frames
        self._votes: Optional[np.ndarray] = None
        self._vote_n = 0
        self.lane = (self.sw * C.FLOOR_EDGE_MARGIN, self.sw * (1.0 - C.FLOOR_EDGE_MARGIN))
        self.walls: Tuple[Optional[float], Optional[float]] = (None, None)  # screen x of each wall found
        self.status = ""
        self._samples: deque = deque(maxlen=400)
        self._fit_t = 0.0
        self._grid: Optional[pygame.Surface] = None
        self._grid_t = 0.0
        self._dirty = True

    def set_reference(self, y0: float, H0: float, feet_visible: bool, anchor_x: float) -> None:
        """Called once calibration finishes: your floor row, height and position at k = 1."""
        self.y0, self.H0, self.feet_visible = float(y0), float(H0), feet_visible
        self.anchor_x = float(anchor_x)
        self.horizon = min(self.horizon, self.y0 - 0.2 * self.sh)
        self._samples.clear()
        self._dirty = True

    def add_floor_mask(self, mask: Optional[np.ndarray], note: str = "") -> str:
        """Fold one segmented frame into the floor map; returns a status line for the HUD."""
        if mask is not None:
            size = (self.sw // self.MASK_SCALE, self.sh // self.MASK_SCALE)
            m = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(np.float32)
            self._votes = m if self._votes is None else self._votes + m
            self._vote_n += 1
            self.floor_mask = self._votes / self._vote_n >= 0.5
        self._compute_lane()
        if self.floor_mask is None:
            self.status = note or "No floor in view: the shadow is kept inside the screen"
        elif not self.feet_visible:
            self.status = "Feet not visible: walls can't be placed, the shadow is kept inside the screen"
        else:
            left, right = self.walls
            found = [s for s, w in (("left", left), ("right", right)) if w is not None]
            self.status = (f"Floor mapped: wall{'s' if len(found) == 2 else ''} on the {' and '.join(found)}"
                           if found else "Floor mapped: open floor to both screen edges")
        self._dirty = True
        return self.status

    def _compute_lane(self) -> None:
        """Left/right limits of the lane: where the floor under the shadow meets a wall or furniture."""
        S = self.MASK_SCALE
        lo_def, hi_def = self.sw * C.FLOOR_EDGE_MARGIN, self.sw * (1.0 - C.FLOOR_EDGE_MARGIN)
        self.lane, self.walls = (lo_def, hi_def), (None, None)
        m = self.floor_mask
        if m is None or not self.feet_visible:
            return
        h, w = m.shape
        c = min(w - 1, max(0, int(self.anchor_x / S)))
        edges = [(r, *_run_around(m[r], c)) for r in range(h) if m[r].any()]
        if len(edges) < 4:
            return
        e = np.asarray(edges, dtype=np.float64)
        r0 = self.y0 / S
        near = e[np.abs(e[:, 0] - r0) <= 4]
        if len(near) >= 3:  # the lane row is visible: read the floor's extent straight off it
            left, right = float(np.median(near[:, 1])), float(np.median(near[:, 2]))
        else:  # extend the floor's edges from the lowest visible rows down to the lane row
            low = e[e[:, 0] >= e[:, 0].max() - 25]
            left = self._extend(low[:, 0], low[:, 1], r0, border=0)
            right = self._extend(low[:, 0], low[:, 2], r0, border=w - 1)
        wl = left * S if left is not None and left > 1 else None              # floor runs off the image: no wall
        wr = (right + 1) * S if right is not None and right < w - 2 else None
        lo = max(lo_def, wl) if wl is not None else lo_def
        hi = min(hi_def, wr) if wr is not None else hi_def
        if hi - lo < C.FLOOR_MIN_LANE * self.H0:  # implausibly narrow: a bad mask, not a corridor
            return
        self.lane, self.walls = (lo, hi), (wl, wr)

    @staticmethod
    def _extend(rows: np.ndarray, cols: np.ndarray, r0: float, border: int) -> Optional[float]:
        if len(rows) < 3 or np.any(np.abs(cols[-5:] - border) <= 1):
            return None  # the floor reaches the image edge near the bottom: no wall on that side
        slope, icpt = np.polyfit(rows, cols, 1)
        return float(slope * r0 + icpt)

    # --- projection ----------------------------------------------------------
    def y_from_k(self, k: float) -> float:
        return self.horizon + (self.y0 - self.horizon) * k

    def k_from_y(self, y: float) -> float:
        return (y - self.horizon) / max(1e-6, self.y0 - self.horizon)

    def depth_from_k(self, k: float) -> float:
        return self.focal / (self.H0 * max(k, 1e-3))

    def k_from_depth(self, z: float) -> float:
        return self.focal / (self.H0 * max(z, 1e-3))

    def world_x(self, x: float, k: float) -> float:
        return (x - self.cx) / (self.H0 * k)

    def screen_x(self, wx: float, k: float) -> float:
        return self.cx + wx * self.H0 * k

    @property
    def lane_z(self) -> float:
        return self.depth_from_k(1.0)

    # --- walkable lane -------------------------------------------------------
    def walkable(self, wx: float, wz: float, offsets: Iterable[float] = None) -> bool:
        """Is every point (wx + offset) on the lane between its walls?"""
        if offsets is None:
            offsets = (-C.FLOOR_HALF_FOOTPRINT, 0.0, C.FLOOR_HALF_FOOTPRINT)
        k = self.k_from_depth(wz)
        lo, hi = self.lane
        return all(lo <= self.screen_x(wx + off, k) <= hi for off in offsets)

    def nearest_walkable(self, wx: float, wz: float, offsets: Iterable[float] = None) -> Vec:
        offsets = tuple(offsets) if offsets is not None else None
        if self.walkable(wx, wz, offsets):
            return wx, wz
        for i in range(1, 121):
            d = i * 0.025
            for cand in (wx + d, wx - d):
                if self.walkable(cand, wz, offsets):
                    return cand, wz
        return wx, wz

    def clamp(self, prev: Vec, new: Vec, offsets: Iterable[float] = None) -> Vec:
        """Move from prev toward new without passing a wall: stops flush against it."""
        offsets = tuple(offsets) if offsets is not None else None
        if self.walkable(new[0], new[1], offsets):
            return new
        if not self.walkable(prev[0], prev[1], offsets):
            return self.nearest_walkable(new[0], new[1], offsets)
        lo, hi = prev[0], new[0]  # bisect to the last free spot, so he ends up touching the wall
        for _ in range(12):
            mid = (lo + hi) * 0.5
            if self.walkable(mid, new[1], offsets):
                lo = mid
            else:
                hi = mid
        return lo, new[1]

    # --- horizon refinement --------------------------------------------------
    def observe(self, player, now: float) -> None:
        """Collect (ankle row, depth ratio) while you stand on the floor and refit the horizon."""
        if not (player.calibrated and player.tracked and self.feet_visible) or player.is_jumping:
            return
        ankles = [player.pts[k][1] for k in ("l_ankle", "r_ankle") if k in player.pts]
        if not ankles:
            return
        self._samples.append((max(ankles), player.depth_ratio))
        if now - self._fit_t < 0.5 or len(self._samples) < 40:
            return
        self._fit_t = now
        data = np.asarray(self._samples, dtype=np.float64)
        y, r = data[:, 0], data[:, 1]
        if r.max() - r.min() < C.FLOOR_HORIZON_MIN_SPREAD:
            return
        slope, intercept = np.polyfit(r, y, 1)  # y = horizon + (y0 - horizon) * r
        if slope > 0:
            h = max(-0.5 * self.sh, min(self.y0 - 0.2 * self.sh, float(intercept)))
            self.horizon += (h - self.horizon) * 0.3

    # --- drawing -------------------------------------------------------------
    def _build_grid(self) -> None:
        """Perspective grid on the mapped floor, the lane line, and a glowing barrier at each wall."""
        surf = pygame.Surface((self.sw, self.sh), pygame.SRCALPHA)
        col = (*C.FLOOR_GRID_COLOR, 255)
        k_far, k_near = C.FLOOR_K_MIN, self.k_from_y(self.sh)
        if self.floor_mask is not None and k_near > k_far:
            step = C.FLOOR_GRID_STEP
            z = math.ceil(self.depth_from_k(k_near) / step) * step
            while z <= self.depth_from_k(k_far):
                y = int(self.y_from_k(self.k_from_depth(z)))
                pygame.draw.line(surf, col, (0, y), (self.sw, y), 1)
                z += step
            y_near, y_far = self.y_from_k(k_near), self.y_from_k(k_far)
            half = abs(self.world_x(0, k_far)) + step
            for i in range(-int(half / step), int(half / step) + 1):
                wx = i * step
                pygame.draw.aaline(surf, col, (self.screen_x(wx, k_near), y_near), (self.screen_x(wx, k_far), y_far))
            full = cv2.resize(self.floor_mask.astype(np.uint8), (self.sw, self.sh), interpolation=cv2.INTER_NEAREST)
            fade = np.clip((np.arange(self.sh) - self.horizon) / max(1.0, self.y0 - self.horizon), 0.25, 1.0)
            alpha = pygame.surfarray.pixels_alpha(surf)  # (w, h) view
            alpha[:] = (alpha * (full.T * fade[None, :]) * (C.FLOOR_GRID_ALPHA / 255.0)).astype(np.uint8)
            del alpha
        lo, hi = self.lane
        y0 = int(min(self.sh - 2, self.y0))
        pygame.draw.line(surf, (*C.FLOOR_GRID_COLOR, 120), (int(lo), y0), (int(hi), y0), 2)  # the lane
        top = self.y0 - 1.05 * self.H0
        for x in self.walls:
            if x is None:
                continue
            for i in range(24):  # barrier: bright at the floor, fading upward
                t = i / 24
                y1, y2 = top + (self.y0 - top) * t, top + (self.y0 - top) * (t + 1 / 24)
                a = int(150 * t ** 1.5)
                pygame.draw.line(surf, (*C.WALL_COLOR, a), (int(x), int(y1)), (int(x), int(y2)), 4)
            pygame.draw.circle(surf, (*C.WALL_COLOR, 200), (int(x), y0), 6)
        self._grid = surf
        self._grid_t = 0.0
        self._dirty = False

    def draw_grid(self, surf: pygame.Surface, dt: float, always: bool = False) -> None:
        """Shown for a few seconds after each mapping (and always in debug view), then fades away."""
        if self._dirty:
            self._build_grid()
        self._grid_t += dt
        t = self._grid_t
        a = 1.0 if always else min(1.0, t / 0.5) * max(0.0, min(1.0, (C.FLOOR_GRID_SHOW - t) / 1.0))
        if a <= 0.01:
            return
        self._grid.set_alpha(int(255 * a))
        surf.blit(self._grid, (0, 0))

    def draw_debug(self, surf: pygame.Surface) -> None:
        y = int(self.horizon)
        if 0 <= y < self.sh:
            pygame.draw.line(surf, (255, 255, 120), (0, y), (self.sw, y), 1)

    def draw_contact_shadow(self, surf: pygame.Surface, x: float, y: float, width: float,
                            strength: float = 1.0) -> None:
        """Soft ellipse on the floor; flattened by the floor's perspective at row y."""
        if strength <= 0.02:
            return
        w = max(8, int(width))
        aspect = max(0.12, min(0.4, (y - self.horizon) / self.focal))
        h = max(4, int(w * aspect))
        key = (w // 4 * 4, h // 2 * 2)
        img = self._shadow_cache.get(key)
        if img is None:
            if len(self._shadow_cache) > 300:
                self._shadow_cache.clear()
            img = pygame.Surface(key, pygame.SRCALPHA)
            for i in range(8):  # darker in the middle
                t = i / 8
                r = pygame.Rect(0, 0, int(key[0] * (1 - t * 0.85)), int(key[1] * (1 - t * 0.85)))
                r.center = (key[0] // 2, key[1] // 2)
                pygame.draw.ellipse(img, (0, 0, 0, int(C.CONTACT_SHADOW_ALPHA * 0.22)), r)
            self._shadow_cache[key] = img
        img.set_alpha(int(255 * max(0.0, min(1.0, strength))))
        surf.blit(img, img.get_rect(center=(int(x), int(y))))


class FloorMapper:
    """Runs the SegFormer floor model on one camera frame in a background thread."""

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self._session = None
        self._lock = threading.Lock()
        self._result = None
        self.busy = False

    def request(self, rgb: np.ndarray, person: Optional[np.ndarray], feet: Optional[Vec]) -> None:
        """rgb / person mask at the pose image size; feet in that image's pixels."""
        if self.busy:
            return
        self.busy = True
        threading.Thread(target=self._run, args=(rgb.copy(), None if person is None else person.copy(), feet),
                         name="FloorMapper", daemon=True).start()

    def poll(self):
        """(mask or None, note) once a request finishes, else None."""
        with self._lock:
            result, self._result = self._result, None
        return result

    def _load(self) -> None:
        if self._session is not None:
            return
        if ort is None:
            raise RuntimeError("onnxruntime is not installed (pip install onnxruntime)")
        if not self.model_path.exists():
            print(f"[floor] downloading floor model to {self.model_path} ...")
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.model_path.with_suffix(".part")
            urllib.request.urlretrieve(C.FLOOR_MODEL_URL, tmp)
            tmp.replace(self.model_path)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2  # leave cores for the pose model and the game
        self._session = ort.InferenceSession(str(self.model_path), opts, providers=["CPUExecutionProvider"])

    def _segment(self, rgb: np.ndarray) -> np.ndarray:
        x = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        x = ((x - self.MEAN) / self.STD).transpose(2, 0, 1)[None].astype(np.float32)
        logits = self._session.run(None, {"pixel_values": x})[0][0]
        labels = logits.argmax(axis=0).astype(np.uint8)
        return cv2.resize(labels, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def _postprocess(labels: np.ndarray, person: Optional[np.ndarray], feet: Optional[Vec]) -> np.ndarray:
        floor = np.isin(labels, C.FLOOR_CLASS_IDS)
        body = labels == 12
        if person is not None:
            body |= person > C.PERSON_MASK_THRESHOLD
        # The floor behind you is hidden by your body: fill rows where floor shows on both sides.
        h, w = floor.shape
        for row in np.nonzero(body.any(axis=1))[0]:
            cols = np.nonzero(body[row])[0]
            lo, hi = max(0, cols[0] - 6), min(w - 1, cols[-1] + 6)
            if floor[row, max(0, lo - 6):lo + 1].any() and floor[row, hi:hi + 7].any():
                floor[row, lo:hi + 1] = True
        m = floor.astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, comp = cv2.connectedComponents(m)
        if n <= 1:
            return m > 0
        keep = None
        if feet is not None:  # the patch of floor you're standing on
            fx, fy = int(feet[0]), int(feet[1])
            win = comp[max(0, fy - 10):min(h, fy + 25), max(0, fx - 40):min(w, fx + 40)]
            ids = win[win > 0]
            if ids.size:
                keep = int(np.bincount(ids).argmax())
        if keep is None:  # otherwise the biggest one
            keep = int(np.bincount(comp[comp > 0]).argmax())
        return comp == keep

    def _run(self, rgb: np.ndarray, person: Optional[np.ndarray], feet: Optional[Vec]) -> None:
        try:
            self._load()
            mask = self._postprocess(self._segment(rgb), person, feet)
            result = (mask, "") if mask.mean() >= C.FLOOR_MIN_FRACTION else \
                (None, "Not enough floor in view: the shadow is kept inside the screen")
        except Exception as exc:  # no model / no onnxruntime / bad download
            print(f"[floor] floor model unavailable: {type(exc).__name__}: {exc}")
            result = (None, "Floor model unavailable: the shadow is kept inside the screen")
        with self._lock:
            self._result = result
            self.busy = False
