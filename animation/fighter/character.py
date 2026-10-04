"""Skinned 3D fighter: rig, baked animation clips, blending and the animation state player.

Data comes from assets/character/fighter.npz, baked by tools/build_character.py from CC0
Quaternius assets and CMU motion capture (see ASSETS.md).

Conventions: quaternions are (x, y, z, w); model space is the rig's world space after its
root node: Y up, the character faces +Z, 1 unit = 1 metre.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# --- quaternion helpers (vectorised over leading axes) --------------------------
def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack((aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz), axis=-1)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = np.moveaxis(quat_normalize(q), -1, 0)
    m = np.empty(q.shape[:-1] + (3, 3))
    m[..., 0, 0] = 1 - 2 * (y * y + z * z); m[..., 0, 1] = 2 * (x * y - z * w); m[..., 0, 2] = 2 * (x * z + y * w)
    m[..., 1, 0] = 2 * (x * y + z * w); m[..., 1, 1] = 1 - 2 * (x * x + z * z); m[..., 1, 2] = 2 * (y * z - x * w)
    m[..., 2, 0] = 2 * (x * z - y * w); m[..., 2, 1] = 2 * (y * z + x * w); m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def mat_to_quat(m: np.ndarray) -> np.ndarray:
    """Rotation matrices (..., 3, 3) -> quaternions (..., 4)."""
    m = np.asarray(m, dtype=np.float64)
    flat = m.reshape(-1, 3, 3)
    out = np.empty((len(flat), 4))
    for i, r in enumerate(flat):
        tr = r[0, 0] + r[1, 1] + r[2, 2]
        if tr > 0:
            s = math.sqrt(tr + 1.0) * 2
            out[i] = ((r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s)
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
            out[i] = (0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s)
        elif r[1, 1] > r[2, 2]:
            s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
            out[i] = ((r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s)
        else:
            s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
            out[i] = ((r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s, (r[1, 0] - r[0, 1]) / s)
    return quat_normalize(out).reshape(m.shape[:-2] + (4,))


def quat_axis_angle(axis: Sequence[float], angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64)
    a = a / max(np.linalg.norm(a), 1e-12)
    s = math.sin(angle * 0.5)
    return np.array((a[0] * s, a[1] * s, a[2] * s, math.cos(angle * 0.5)))


def quat_blend(qs: Sequence[np.ndarray], ws: Sequence[float]) -> np.ndarray:
    """Weighted blend of (J, 4) quaternion sets (normalised lerp with hemisphere alignment)."""
    ref = qs[0]
    acc = np.zeros_like(ref)
    for q, w in zip(qs, ws):
        sign = np.where(np.sum(q * ref, axis=-1, keepdims=True) < 0, -1.0, 1.0)
        acc += q * sign * w
    return quat_normalize(acc)


def compose(t: np.ndarray, q: np.ndarray) -> np.ndarray:
    """(J, 3) translations + (J, 4) rotations -> (J, 4, 4) matrices."""
    m = np.zeros(q.shape[:-1] + (4, 4))
    m[..., :3, :3] = quat_to_mat(q)
    m[..., :3, 3] = t
    m[..., 3, 3] = 1.0
    return m


# --- rig and clips ---------------------------------------------------------------
class Rig:
    def __init__(self, names: Sequence[str], parents: Sequence[int], rest_t: np.ndarray, rest_r: np.ndarray,
                 inv_bind: np.ndarray, hips: int):
        self.names = list(names)
        self.parents = [int(p) for p in parents]
        self.rest_t = np.asarray(rest_t, dtype=np.float64)
        self.rest_r = np.asarray(rest_r, dtype=np.float64)
        self.inv_bind = np.asarray(inv_bind, dtype=np.float64)
        self.hips = int(hips)
        self.index = {n: i for i, n in enumerate(self.names)}
        self.order = self._topological()

    def _topological(self) -> List[int]:
        order, seen = [], set()

        def visit(i):
            if i in seen:
                return
            p = self.parents[i]
            if p >= 0:
                visit(p)
            seen.add(i)
            order.append(i)
        for i in range(len(self.names)):
            visit(i)
        return order

    def __len__(self) -> int:
        return len(self.names)

    def world(self, local_r: np.ndarray, hips_t: np.ndarray, root: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward kinematics: local rotations (J, 4) and the hips translation -> world matrices (J, 4, 4).

        `root` (4x4) places the whole character (yaw, position, scale) in the scene.
        """
        t = self.rest_t.copy()
        t[self.hips] = hips_t
        local = compose(t, local_r)
        world = np.empty_like(local)
        for i in self.order:
            p = self.parents[i]
            world[i] = (world[p] @ local[i]) if p >= 0 else (local[i] if root is None else root @ local[i])
        return world


@dataclass
class Clip:
    name: str
    fps: float
    rot: np.ndarray        # (F, J, 4) local rotations
    hips: np.ndarray       # (F, 3) hips translation (local to its parent)
    loop: bool
    meta: Dict[str, float] = field(default_factory=dict)  # strike limb, contact / active window, reach ...
    root: Optional[np.ndarray] = None  # (F,) forward travel in metres (root-motion clips), applied by the game

    @property
    def frames(self) -> int:
        return len(self.rot)

    def root_at(self, t: float) -> float:
        if self.root is None:
            return 0.0
        f = max(0.0, min(self.frames - 1.0, t * self.fps))
        i = int(f)
        j = min(i + 1, self.frames - 1)
        return float(self.root[i] + (self.root[j] - self.root[i]) * (f - i))

    @property
    def duration(self) -> float:
        return (self.frames - 1) / self.fps

    def sample(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        if self.loop:
            t = t % self.duration if self.duration > 0 else 0.0
        f = max(0.0, min(self.frames - 1.0, t * self.fps))
        i = int(f)
        j = min(i + 1, self.frames - 1)
        a = f - i
        if a < 1e-4 or i == j:
            return self.rot[i], self.hips[i]
        q0, q1 = self.rot[i], self.rot[j]
        sign = np.where(np.sum(q0 * q1, axis=-1, keepdims=True) < 0, -1.0, 1.0)
        return quat_normalize(q0 * (1 - a) + q1 * sign * a), self.hips[i] * (1 - a) + self.hips[j] * a


@dataclass
class Mesh:
    positions: np.ndarray  # (V, 3) float32
    normals: np.ndarray    # (V, 3)
    joints: np.ndarray     # (V, 4) int
    weights: np.ndarray    # (V, 4) float32
    indices: np.ndarray    # (I,) uint32
    part: np.ndarray       # (V,) 0 = body shell, 1 = joint caps


class CharacterData:
    """Everything baked into fighter.npz."""

    def __init__(self, path: Path):
        d = np.load(path, allow_pickle=False)
        self.rig = Rig([str(n) for n in d["joint_names"]], d["parents"], d["rest_t"], d["rest_r"],
                       d["inv_bind"], int(d["hips"]))
        self.above = d["above"].astype(np.float64)  # scene transform above the root joint
        self.mesh = Mesh(d["positions"], d["normals"], d["joints"].astype(np.int32), d["weights"],
                         d["indices"].astype(np.uint32), d["part"])
        self.height = float(d["height"])  # standing height in metres
        self.grip_axis = d["grip_axis"].astype(np.float64)  # weapon axis through the right fist (bind space)
        self.clips: Dict[str, Clip] = {}
        meta_keys = [str(k) for k in d["meta_keys"]]
        for i, name in enumerate(str(n) for n in d["clip_names"]):
            meta = {k: float(v) for k, v in zip(meta_keys, d["clip_meta"][i]) if not np.isnan(v)}
            root = d[f"root_{i}"].astype(np.float64)
            self.clips[name] = Clip(name, float(d["clip_fps"][i]), d[f"rot_{i}"].astype(np.float64),
                                    d[f"hips_{i}"].astype(np.float64), bool(d["clip_loop"][i]), meta,
                                    root if np.any(root) else None)


# --- animation player --------------------------------------------------------------
@dataclass
class Track:
    clip: Clip
    time: float = 0.0
    speed: float = 1.0
    weight: float = 0.0
    target: float = 1.0
    fade: float = 0.15
    loop: bool = False

    @property
    def finished(self) -> bool:
        return not self.loop and self.time >= self.clip.duration


class Animator:
    """Cross-fading clip player with an optional masked overlay layer and additive offsets.

    Base layer: the newest clip fades in while older ones fade out (weights always sum to 1).
    Overlay: e.g. a fighting guard on the upper body while the legs walk.
    """

    def __init__(self, data: CharacterData):
        self.data = data
        self.rig = data.rig
        self.tracks: List[Track] = []
        self.overlay: Optional[Track] = None
        self.overlay_mask = np.zeros(len(self.rig))
        self.overlay_weight = 0.0
        self.overlay_target = 0.0
        self.additive: Dict[int, np.ndarray] = {}  # joint -> extra local rotation (flinches, aim)

    # --- control -------------------------------------------------------------
    def play(self, name: str, fade: float = 0.15, speed: float = 1.0, loop: Optional[bool] = None,
             start: float = 0.0, restart: bool = False) -> Track:
        clip = self.data.clips[name]
        cur = self.current
        if cur is not None and cur.clip is clip and not restart and not cur.finished:
            cur.speed = speed
            return cur
        for tr in self.tracks:
            tr.target = 0.0
            tr.fade = max(fade, 1e-3)
        tr = Track(clip, start, speed, 0.0 if self.tracks else 1.0, 1.0, max(fade, 1e-3),
                   clip.loop if loop is None else loop)
        self.tracks.append(tr)
        return tr

    def set_overlay(self, name: Optional[str], mask: Optional[np.ndarray] = None, weight: float = 1.0,
                    speed: float = 1.0) -> None:
        if name is None:
            self.overlay_target = 0.0
            return
        if self.overlay is None or self.overlay.clip.name != name:
            self.overlay = Track(self.data.clips[name], 0.0, speed, 1.0, 1.0, 0.2, True)
        self.overlay.speed = speed
        if mask is not None:
            self.overlay_mask = mask
        self.overlay_target = weight

    @property
    def current(self) -> Optional[Track]:
        return self.tracks[-1] if self.tracks else None

    def normalized(self) -> float:
        cur = self.current
        return 0.0 if cur is None or cur.clip.duration <= 0 else min(1.0, cur.time / cur.clip.duration)

    def update(self, dt: float) -> float:
        """Advance; returns this step's root motion (metres forward) from root-motion clips."""
        travel = 0.0
        for tr in self.tracks:
            before = tr.clip.root_at(tr.time)
            tr.time += dt * tr.speed
            if tr.loop and tr.clip.duration > 0:
                tr.time %= tr.clip.duration
            if tr.clip.root is not None:
                travel += (tr.clip.root_at(min(tr.time, tr.clip.duration)) - before) * tr.weight
            step = dt / tr.fade
            tr.weight = min(tr.target, tr.weight + step) if tr.weight < tr.target else max(tr.target, tr.weight - step)
        self.tracks = [t for t in self.tracks if t.weight > 1e-3 or t is self.tracks[-1]]
        if self.overlay is not None:
            self.overlay.time += dt * self.overlay.speed
            k = min(1.0, dt / 0.2)
            self.overlay_weight += (self.overlay_target - self.overlay_weight) * k
        return travel

    # --- evaluation -----------------------------------------------------------
    def pose(self) -> Tuple[np.ndarray, np.ndarray]:
        samples = [(t.clip.sample(t.time), t.weight) for t in self.tracks if t.weight > 1e-3]
        if not samples:
            rot, hips = self.rig.rest_r.copy(), self.rig.rest_t[self.rig.hips].copy()
        else:
            total = sum(w for _, w in samples)
            ws = [w / total for _, w in samples]
            rot = quat_blend([s[0] for s, _ in samples], ws)
            hips = sum(s[1] * w for (s, _), w in zip(samples, ws))
        if self.overlay is not None and self.overlay_weight > 1e-3:
            orot, _ = self.overlay.clip.sample(self.overlay.time)
            w = (self.overlay_mask * self.overlay_weight)[:, None]
            sign = np.where(np.sum(rot * orot, axis=-1, keepdims=True) < 0, -1.0, 1.0)
            rot = quat_normalize(rot * (1 - w) + orot * sign * w)
        for j, q in self.additive.items():
            rot[j] = quat_mul(rot[j], q)
        return rot, hips
