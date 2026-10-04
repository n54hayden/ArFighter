#!/usr/bin/env python3
"""Bake the fighter character into assets/character/fighter.npz.

    python tools/build_character.py [--cache DIR]

Sources (all downloaded on first run into the cache directory; see ASSETS.md):
  * Quaternius "Universal Animation Library" (CC0): the rigged, skinned mannequin and its
    clips (idle, walk, jab, cross, hit reactions, death, roll, jumps, sword, spells).
  * Quaternius "Universal Animation Library 2" (CC0): knockback, get-up, hook, sword block /
    combos, overhand throw, ninja jump, slide. Different rig -> retargeted.
  * CMU Graphics Lab Motion Capture Database (free for any use): real recorded karate kicks
    (subject 135: front kick, roundhouse, side kick) and a boxing guard (14_01). Retargeted.

Retargeting is direction based: every mapped target bone is swung so it points the same way
as the corresponding source segment (twist inherited from its parent), the hips take their
full orientation from the source pelvis frame, and the hips position is scaled by leg length.
That works across rigs with different rest poses and bone lengths.

Each attack clip is analysed for its striking limb, contact time and active window (the limb
moving out toward its furthest reach), which drive hit detection in the game.
"""
from __future__ import annotations

import argparse
import io
import json
import struct
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from fighter.character import compose, mat_to_quat, quat_normalize, quat_to_mat  # noqa: E402

FPS = 30.0
UAL1 = "https://raw.githubusercontent.com/J-Ponzo/gltf-universal-animation-library/HEAD/glTF/"
UAL2_ZIP = "https://opengameart.org/sites/default/files/universal_animation_library_2standard.zip"
CMU = "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/"


# --- downloads ---------------------------------------------------------------------
def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {url}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    return dest


def get_sources(cache: Path) -> Dict[str, Path]:
    src = {
        "ual1": fetch(UAL1 + "AnimationLibrary_Godot_Standard.gltf", cache / "ual1/AnimationLibrary_Godot_Standard.gltf"),
    }
    fetch(UAL1 + "AnimationLibrary_Godot_Standard.bin", cache / "ual1/AnimationLibrary_Godot_Standard.bin")
    glb = cache / "ual2/UAL2_Standard.glb"
    if not glb.exists():
        z = zipfile.ZipFile(fetch(UAL2_ZIP, cache / "ual2/ual2.zip"))
        name = next(n for n in z.namelist() if n.endswith("Unreal-Godot/UAL2_Standard.glb"))
        glb.write_bytes(z.read(name))
    src["ual2"] = glb
    for trial in ("135/135_04", "135/135_07", "135/135_11", "014/14_01"):
        src[trial.split("/")[1]] = fetch(CMU + trial + ".bvh", cache / f"cmu/{trial.split('/')[1]}.bvh")
    return src


# --- glTF -------------------------------------------------------------------------------
CTYPE = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


class GLTF:
    def __init__(self, path: Path):
        data = path.read_bytes()
        if data[:4] == b"glTF":
            jlen = struct.unpack_from("<I", data, 12)[0]
            self.j = json.loads(data[20:20 + jlen])
            off = 20 + jlen
            blen = struct.unpack_from("<I", data, off)[0]
            self.bin = data[off + 8: off + 8 + blen]
        else:
            self.j = json.loads(data)
            self.bin = (path.parent / self.j["buffers"][0]["uri"]).read_bytes()
        self.nodes = self.j["nodes"]
        self.names = [n.get("name", f"node{i}") for i, n in enumerate(self.nodes)]
        self.parent = [-1] * len(self.nodes)
        for i, n in enumerate(self.nodes):
            for c in n.get("children", []):
                self.parent[c] = i
        self.rest_t = np.array([n.get("translation", [0, 0, 0]) for n in self.nodes], dtype=np.float64)
        self.rest_r = np.array([n.get("rotation", [0, 0, 0, 1]) for n in self.nodes], dtype=np.float64)
        self.anims = {a["name"]: a for a in self.j.get("animations", [])}
        self.order = []
        seen = set()

        def visit(i):
            if i in seen:
                return
            if self.parent[i] >= 0:
                visit(self.parent[i])
            seen.add(i)
            self.order.append(i)
        for i in range(len(self.nodes)):
            visit(i)

    def accessor(self, i: int) -> np.ndarray:
        a = self.j["accessors"][i]
        bv = self.j["bufferViews"][a["bufferView"]]
        n, dt = NCOMP[a["type"]], np.dtype(CTYPE[a["componentType"]])
        off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = bv.get("byteStride", n * dt.itemsize)
        raw = np.frombuffer(self.bin, np.uint8, stride * (a["count"] - 1) + n * dt.itemsize, off)
        out = np.lib.stride_tricks.as_strided(raw, (a["count"], n * dt.itemsize), (stride, 1)).copy()
        arr = out.view(dt).reshape(a["count"], n)
        if a.get("normalized"):
            arr = arr.astype(np.float32) / np.iinfo(dt).max
        return arr if n > 1 else arr[:, 0]

    def sample(self, anim: str, fps: float = FPS) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Local (F, N, 3) translations and (F, N, 4) rotations of every node on a uniform grid."""
        a = self.anims[anim]
        chans = []
        end = 0.0
        for c in a["channels"]:
            s = a["samplers"][c["sampler"]]
            t = self.accessor(s["input"]).astype(np.float64)
            v = self.accessor(s["output"]).astype(np.float64)
            if s.get("interpolation") == "CUBICSPLINE":
                v = v.reshape(len(t), 3, -1)[:, 1]
            chans.append((c["target"]["node"], c["target"]["path"], t, v, s.get("interpolation", "LINEAR")))
            end = max(end, float(t[-1]))
        times = np.arange(0.0, end + 1e-6, 1.0 / fps)
        if len(times) < 2:
            times = np.array([0.0, 1.0 / fps])
        T = np.repeat(self.rest_t[None], len(times), 0)
        R = np.repeat(self.rest_r[None], len(times), 0)
        for node, path, t, v, interp in chans:
            if path not in ("translation", "rotation"):
                continue
            idx = np.clip(np.searchsorted(t, times, side="right") - 1, 0, len(t) - 1)
            nxt = np.minimum(idx + 1, len(t) - 1)
            span = np.maximum(t[nxt] - t[idx], 1e-9)
            a_ = np.clip((times - t[idx]) / span, 0, 1)[:, None]
            if interp == "STEP":
                a_ = a_ * 0
            v0, v1 = v[idx], v[nxt]
            if path == "rotation":
                sign = np.where(np.sum(v0 * v1, axis=1, keepdims=True) < 0, -1.0, 1.0)
                R[:, node] = quat_normalize(v0 * (1 - a_) + v1 * sign * a_)
            else:
                T[:, node] = v0 * (1 - a_) + v1 * a_
        return times, T, R

    def world(self, T: np.ndarray, R: np.ndarray) -> np.ndarray:
        """(F, N, 3/4) local -> (F, N, 4, 4) world."""
        local = compose(T, R)
        W = np.empty_like(local)
        for i in self.order:
            p = self.parent[i]
            W[:, i] = local[:, i] if p < 0 else W[:, p] @ local[:, i]
        return W


# --- BVH ----------------------------------------------------------------------------------
def _axis_rot(axis: str, deg: np.ndarray) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    m = np.zeros(deg.shape + (3, 3))
    if axis == "X":
        m[..., 0, 0] = 1; m[..., 1, 1] = c; m[..., 1, 2] = -s; m[..., 2, 1] = s; m[..., 2, 2] = c
    elif axis == "Y":
        m[..., 1, 1] = 1; m[..., 0, 0] = c; m[..., 0, 2] = s; m[..., 2, 0] = -s; m[..., 2, 2] = c
    else:
        m[..., 2, 2] = 1; m[..., 0, 0] = c; m[..., 0, 1] = -s; m[..., 1, 0] = s; m[..., 1, 1] = c
    return m


def load_bvh(path: Path) -> Tuple[List[str], float, np.ndarray]:
    """Returns joint names, frame time and world positions (F, J, 3)."""
    toks = path.read_text().split()
    names, parents, offsets, chans = [], [], [], []
    stack, i = [], 0
    while toks[i] != "MOTION":
        tk = toks[i]
        if tk in ("ROOT", "JOINT"):
            names.append(toks[i + 1]); parents.append(stack[-1] if stack else -1)
            offsets.append([0, 0, 0]); chans.append([]); i += 2
        elif tk == "End":
            names.append(names[stack[-1]] + "_end"); parents.append(stack[-1])
            offsets.append([0, 0, 0]); chans.append([]); i += 2
        elif tk == "{":
            stack.append(len(names) - 1); i += 1
        elif tk == "}":
            stack.pop(); i += 1
        elif tk == "OFFSET":
            offsets[-1 if not stack else stack[-1]] = [float(x) for x in toks[i + 1:i + 4]]; i += 4
        elif tk == "CHANNELS":
            n = int(toks[i + 1]); chans[stack[-1]] = toks[i + 2:i + 2 + n]; i += 2 + n
        else:
            i += 1
    nframes = int(toks[i + 2]); ftime = float(toks[i + 5])
    vals = np.array(toks[i + 6:], dtype=np.float64).reshape(nframes, -1)
    F, J = nframes, len(names)
    W = np.zeros((F, J, 4, 4))
    col = 0
    for j in range(J):
        rot = np.repeat(np.eye(3)[None], F, 0)
        pos = np.repeat(np.array(offsets[j], dtype=np.float64)[None], F, 0)
        for ch in chans[j]:
            v = vals[:, col]; col += 1
            if ch.endswith("position"):
                pos[:, "XYZ".index(ch[0])] = v
            else:
                rot = rot @ _axis_rot(ch[0], v)
        local = np.zeros((F, 4, 4)); local[:, :3, :3] = rot; local[:, :3, 3] = pos; local[:, 3, 3] = 1
        W[:, j] = local if parents[j] < 0 else W[:, parents[j]] @ local
    return names, ftime, W[:, :, :3, 3]


# --- target rig ---------------------------------------------------------------------------
# Target bone -> the child that defines its direction.
SEGMENTS = {
    "DEF-spine.001": "DEF-spine.002", "DEF-spine.002": "DEF-spine.003", "DEF-spine.003": "DEF-neck",
    "DEF-neck": "DEF-head",
}
for s in "LR":
    SEGMENTS.update({f"DEF-shoulder.{s}": f"DEF-upper_arm.{s}", f"DEF-upper_arm.{s}": f"DEF-forearm.{s}",
                     f"DEF-forearm.{s}": f"DEF-hand.{s}", f"DEF-hand.{s}": f"DEF-f_middle.01.{s}",
                     f"DEF-thigh.{s}": f"DEF-shin.{s}", f"DEF-shin.{s}": f"DEF-foot.{s}",
                     f"DEF-foot.{s}": f"DEF-toe.{s}"})


def source_map(kind: str) -> Dict[str, object]:
    """Target bone -> (source a, source b) segment, plus the hips frame and leg joints."""
    if kind == "ual2":
        m = {"DEF-spine.001": ("spine_01", "spine_02"), "DEF-spine.002": ("spine_02", "spine_03"),
             "DEF-spine.003": ("spine_03", "neck_01"), "DEF-neck": ("neck_01", "Head")}
        for s, t in (("l", "L"), ("r", "R")):
            m.update({f"DEF-shoulder.{t}": (f"clavicle_{s}", f"upperarm_{s}"),
                      f"DEF-upper_arm.{t}": (f"upperarm_{s}", f"lowerarm_{s}"),
                      f"DEF-forearm.{t}": (f"lowerarm_{s}", f"hand_{s}"),
                      f"DEF-hand.{t}": (f"hand_{s}", f"middle_01_{s}"),
                      f"DEF-thigh.{t}": (f"thigh_{s}", f"calf_{s}"), f"DEF-shin.{t}": (f"calf_{s}", f"foot_{s}"),
                      f"DEF-foot.{t}": (f"foot_{s}", f"ball_{s}")})
        return {"seg": m, "hips": "pelvis", "chest": "spine_03", "thigh": ("thigh_l", "thigh_r"),
                "leg": ("thigh_l", "calf_l", "foot_l"), "feet": ("foot_l", "foot_r", "ball_l", "ball_r"),
                # hand orientation: (wrist, finger, index-side, pinky-side), so weapon grips keep their twist
                "hand_frame": {"L": ("hand_l", "middle_01_l", "index_01_l", "pinky_01_l"),
                               "R": ("hand_r", "middle_01_r", "index_01_r", "pinky_01_r")}}
    # CMU: Hips->LowerBack and Spine1->Neck have zero length, so the chain skips them.
    m = {"DEF-spine.001": ("LowerBack", "Spine"), "DEF-spine.002": ("Spine", "Spine1"),
         "DEF-spine.003": ("Spine1", "Neck1"), "DEF-neck": ("Neck1", "Head")}
    for s, t in (("Left", "L"), ("Right", "R")):
        m.update({f"DEF-shoulder.{t}": (f"{s}Shoulder", f"{s}Arm"), f"DEF-upper_arm.{t}": (f"{s}Arm", f"{s}ForeArm"),
                  f"DEF-forearm.{t}": (f"{s}ForeArm", f"{s}Hand"), f"DEF-hand.{t}": (f"{s}Hand", f"{s}HandIndex1"),
                  f"DEF-thigh.{t}": (f"{s}UpLeg", f"{s}Leg"), f"DEF-shin.{t}": (f"{s}Leg", f"{s}Foot"),
                  f"DEF-foot.{t}": (f"{s}Foot", f"{s}ToeBase")})
    return {"seg": m, "hips": "Hips", "chest": "Spine1", "thigh": ("LeftUpLeg", "RightUpLeg"),
            "leg": ("LeftUpLeg", "LeftLeg", "LeftFoot"), "feet": ("LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase"),
            # CMU's thumb joint sits on the wrist: its end site carries the direction
            "hand_frame": {"L": ("LeftHand", "LeftHandIndex1", "LThumb_end", "LeftHand"),
                           "R": ("RightHand", "RightHandIndex1", "RThumb_end", "RightHand")}}


def _frame(up: np.ndarray, side: np.ndarray) -> np.ndarray:
    """Orthonormal basis (..., 3, 3) with columns side', up, forward."""
    u = up / np.linalg.norm(up, axis=-1, keepdims=True)
    s = side - u * np.sum(side * u, axis=-1, keepdims=True)
    s = s / np.linalg.norm(s, axis=-1, keepdims=True)
    f = np.cross(s, u)
    return np.stack((s, u, f), axis=-1)


def _rot_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = float(np.dot(a, b))
    if c < -0.9999:
        axis = np.cross(a, [1, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0, 1, 0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


class Target:
    """The UAL1 mannequin rig, its mesh, and the reference data retargeting needs."""

    def __init__(self, g: GLTF):
        self.g = g
        skin = g.j["skins"][0]
        self.nodes = skin["joints"]                                   # node index per joint
        self.names = [g.names[n] for n in self.nodes]
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.parents = [self.nodes.index(g.parent[n]) if g.parent[n] in self.nodes else -1 for n in self.nodes]
        self.inv_bind = g.accessor(skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
        self.hips = self.idx["DEF-hips"]
        # Anything above the skeleton root (scene nodes) folds into the root joint's parent space.
        above = np.eye(4)
        p = g.parent[self.nodes[0]]
        chain = []
        while p >= 0:
            chain.append(p); p = g.parent[p]
        for n in reversed(chain):
            above = above @ compose(g.rest_t[n], g.rest_r[n])
        self.above = above
        self.rest_t = g.rest_t[self.nodes].copy()
        self.rest_r = g.rest_r[self.nodes].copy()
        W = self.world(self.rest_r[None], self.rest_t[self.hips][None])[0]
        self.ref_rot = W[:, :3, :3]
        self.ref_pos = W[:, :3, 3]
        self.leg = (np.linalg.norm(self.ref_pos[self.idx["DEF-shin.L"]] - self.ref_pos[self.idx["DEF-thigh.L"]])
                    + np.linalg.norm(self.ref_pos[self.idx["DEF-foot.L"]] - self.ref_pos[self.idx["DEF-shin.L"]]))
        self.hips_frame_ref = _frame(self.ref_pos[self.idx["DEF-spine.003"]] - self.ref_pos[self.hips],
                                     self.ref_pos[self.idx["DEF-thigh.L"]] - self.ref_pos[self.idx["DEF-thigh.R"]])

    def world(self, R: np.ndarray, hips_t: np.ndarray) -> np.ndarray:
        """(F, J, 4) local rotations + (F, 3) hips translations -> (F, J, 4, 4) model-space matrices."""
        T = np.repeat(self.rest_t[None], len(R), 0)
        T[:, self.hips] = hips_t
        local = compose(T, R)
        W = np.empty_like(local)
        for i in range(len(self.names)):  # skin joints are listed parents-first
            p = self.parents[i]
            W[:, i] = (self.above @ local[:, i]) if p < 0 else W[:, p] @ local[:, i]
        return W

    def clip_from_gltf(self, anim: str) -> Tuple[np.ndarray, np.ndarray]:
        _, T, R = self.g.sample(anim)
        return R[:, self.nodes], T[:, self.nodes[self.hips]]

    def retarget(self, pos: Dict[str, np.ndarray], kind: str, forward: Optional[np.ndarray] = None,
                 fist: Optional[np.ndarray] = None, keep_root_motion: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Source world positions (name -> (F, 3)) -> target local rotations and hips translations."""
        sm = source_map(kind)
        F = len(next(iter(pos.values())))
        # Scale by leg length; ground from the lowest foot; face +Z.
        a, b, c = (pos[n] for n in sm["leg"])
        s_leg = float(np.median(np.linalg.norm(b - a, axis=1) + np.linalg.norm(c - b, axis=1)))
        scale = self.leg / s_leg
        ground = float(np.percentile(np.min([pos[n][:, 1] for n in sm["feet"]], axis=0), 2))
        side0 = pos[sm["thigh"][0]][0] - pos[sm["thigh"][1]][0]
        if forward is None:  # facing from the pelvis: forward = side x up
            forward = np.cross(side0, [0, 1, 0])
        fwd = np.array([forward[0], 0.0, forward[2]]); fwd /= np.linalg.norm(fwd)
        yaw = np.arctan2(fwd[0], fwd[2])  # rotate so the source's forward becomes +Z
        cy, sy = np.cos(-yaw), np.sin(-yaw)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        origin = pos[sm["hips"]][0].copy()
        P = {k: ((v - [origin[0], ground, origin[2]]) @ Ry.T) * scale for k, v in pos.items()}

        J = len(self.names)
        Rw = np.zeros((F, J, 3, 3))
        up = P[sm["chest"]] - P[sm["hips"]]
        side = P[sm["thigh"][0]] - P[sm["thigh"][1]]
        Fs = _frame(up, side)
        hips_rot = Fs @ self.hips_frame_ref.T[None] @ self.ref_rot[self.hips][None]
        seg = sm["seg"]
        # Hands take their full orientation (direction + twist) from the source hand.
        hand_rot = {}
        for s in "LR":
            w, fing, a_, b_ = sm["hand_frame"][s]
            tw, tf, ta, tb = (self.idx[f"DEF-{n}.{s}"] for n in ("hand", "f_middle.01", "f_index.01", "f_pinky.01"))
            if kind == "cmu":  # CMU has no pinky: thumb side vs the wrist; same for the target
                tb = tw
                ta = self.idx[f"DEF-thumb.01.{s}"]
            ref = _frame(self.ref_pos[tf] - self.ref_pos[tw], self.ref_pos[ta] - self.ref_pos[tb])
            src = _frame(P[fing] - P[w], P[a_] - P[b_])
            rot = src @ ref.T[None] @ self.ref_rot[self.idx[f"DEF-hand.{s}"]][None]
            ok = np.isfinite(rot).all(axis=(1, 2))
            if ok.any():  # degenerate frames fall back to the plain swing below
                hand_rot[self.idx[f"DEF-hand.{s}"]] = (rot, ok)
        local_R = np.zeros((F, J, 4))
        for f in range(F):
            for j in range(J):
                p = self.parents[j]
                name = self.names[j]
                inherit = (Rw[f, p] @ self.ref_rot[p].T) if p >= 0 else np.eye(3)
                if j == self.hips:
                    Rw[f, j] = hips_rot[f]
                elif j in hand_rot and hand_rot[j][1][f]:
                    Rw[f, j] = hand_rot[j][0][f]
                elif name in seg and name in SEGMENTS and \
                        np.linalg.norm(P[seg[name][1]][f] - P[seg[name][0]][f]) > 1e-4:
                    ch = self.idx[SEGMENTS[name]]
                    d_ref = self.ref_pos[ch] - self.ref_pos[j]
                    cur = inherit @ d_ref
                    want = P[seg[name][1]][f] - P[seg[name][0]][f]
                    Rw[f, j] = _rot_between(cur, want) @ inherit @ self.ref_rot[j]
                elif fist is not None and ("f_" in name or "thumb" in name):
                    Rw[f, j] = Rw[f, p] @ quat_to_mat(fist[j])
                else:
                    Rw[f, j] = inherit @ self.ref_rot[j]
            parent_rot = np.stack([Rw[f, p] if p >= 0 else self.above[:3, :3] for p in self.parents])
            local_R[f] = mat_to_quat(np.transpose(parent_rot, (0, 2, 1)) @ Rw[f])
        hp = P[sm["hips"]].copy()
        root_fwd = np.zeros(F)
        if keep_root_motion:  # forward travel becomes root motion (the game moves the body)
            root_fwd = hp[:, 2] - hp[0, 2]
            hp[:, 2] = hp[0, 2]
            hp[:, 0] -= np.linspace(0, 1, F) * (hp[-1, 0] - hp[0, 0])
        else:  # remove net drift so the clip starts and ends on the same spot
            drift = hp[-1, [0, 2]] - hp[0, [0, 2]]
            hp[:, [0, 2]] -= np.linspace(0, 1, F)[:, None] * drift
        self.last_root_fwd = root_fwd
        hp_parent = self.parents[self.hips]  # hips translation is local to its parent (the root joint)
        parent_world = self.world(self.rest_r[None], self.rest_t[self.hips][None])[0][hp_parent] if hp_parent >= 0 \
            else self.above
        inv = np.linalg.inv(parent_world)
        hips_t = (hp @ inv[:3, :3].T) + inv[:3, 3]
        return local_R, hips_t


# --- clip tools -----------------------------------------------------------------------------
def resample_positions(pos: Dict[str, np.ndarray], src_dt: float, start: int, end: int) -> Dict[str, np.ndarray]:
    t_src = np.arange(start, end) * src_dt
    t = np.arange(t_src[0], t_src[-1] + 1e-9, 1.0 / FPS)
    return {k: np.stack([np.interp(t, t_src, v[start:end, i]) for i in range(3)], axis=1) for k, v in pos.items()}


def loop_seam(R: np.ndarray, H: np.ndarray, blend: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """Cross-fade the tail into the head so a loop has no pop."""
    R, H = R.copy(), H.copy()
    for i in range(blend):
        w = (i + 1) / (blend + 1)
        k = len(R) - blend + i
        sign = np.where(np.sum(R[k] * R[i], axis=-1, keepdims=True) < 0, -1.0, 1.0)
        R[k] = quat_normalize(R[k] * (1 - w) + R[i] * sign * w)
        H[k] = H[k] * (1 - w) + H[i] * w
    R[-1], H[-1] = R[0], H[0]
    return R, H


def ease_ends(R: np.ndarray, H: np.ndarray, guard_R: np.ndarray, guard_H: np.ndarray, n: int = 5):
    """Blend the first / last frames toward the guard pose so cuts from mocap start and end clean."""
    R, H = R.copy(), H.copy()
    for i in range(n):
        w = 1.0 - (i + 1) / (n + 1)
        for k in (i, len(R) - 1 - i):
            sign = np.where(np.sum(R[k] * guard_R, axis=-1, keepdims=True) < 0, -1.0, 1.0)
            R[k] = quat_normalize(R[k] * (1 - w) + guard_R * sign * w)
            H[k] = H[k] * (1 - w) + guard_H * w
    return R, H


EFFECTORS = {"hand.L": ("DEF-hand.L", 0.08), "hand.R": ("DEF-hand.R", 0.08),
             "foot.L": ("DEF-toe.L", 0.0), "foot.R": ("DEF-toe.R", 0.0)}


def analyse(tgt: Target, R: np.ndarray, H: np.ndarray, root_fwd: np.ndarray, weapon: bool = False,
            kick: bool = False) -> Dict[str, float]:
    """Strike limb, contact time, active window and reach of an attack clip (model space, metres).

    The limb is the effector that travels furthest forward; for kicks only frames with the foot
    off the floor count (so stepping through after the kick isn't mistaken for the kick).
    """
    W = tgt.world(R, H)
    hips0 = W[0, tgt.hips, :3, 3]
    best = None
    for li, (key, (joint, off)) in enumerate(EFFECTORS.items()):
        if weapon and not key.startswith("hand.R"):
            continue
        if kick and not key.startswith("foot"):
            continue
        m = W[:, tgt.idx[joint]]
        p = m[:, :3, 3] + m[:, :3, 1] * off  # a little past the joint, along the bone
        z = p[:, 2] + root_fwd - hips0[2]
        if kick:
            z = np.where(p[:, 1] > 0.3, z, z.min())
        gain = z.max() - z[0]
        if best is None or gain > best[0]:
            best = (gain, li, z, p)
    gain, li, z, p = best
    peak = int(np.argmax(z))
    thr = z[0] + 0.7 * (z[peak] - z[0])
    start = peak
    while start > 0 and z[start - 1] >= thr:
        start -= 1
    end = peak
    while end < len(z) - 1 and z[end + 1] >= thr and end - peak < 4:
        end += 1
    dur = (len(z) - 1) / FPS
    return {"limb": float(li), "contact": peak / FPS, "active_start": max(0.0, start / FPS - 1 / FPS),
            "active_end": min(dur, end / FPS + 1 / FPS), "reach": float(z[peak]),
            "strike_height": float(p[peak, 1])}


# --- the clip list -----------------------------------------------------------------------------
UAL1_CLIPS = {  # our name: (UAL1 clip, loop)
    "idle": ("Idle_Loop", True), "walk": ("Walk_Loop", True), "jog": ("Jog_Fwd_Loop", True),
    "guard_enter": ("Punch_Enter", False), "jab": ("Punch_Jab", False), "cross": ("Punch_Cross", False),
    "hit_chest": ("Hit_Chest", False), "hit_head": ("Hit_Head", False), "death": ("Death01", False),
    "roll": ("Roll", False), "jump_start": ("Jump_Start", False), "jump_loop": ("Jump_Loop", True),
    "jump_land": ("Jump_Land", False), "sword_attack": ("Sword_Attack", False), "sword_idle": ("Sword_Idle", True),
    "spell_enter": ("Spell_Simple_Enter", False), "spell_idle": ("Spell_Simple_Idle_Loop", True),
    "spell_shoot": ("Spell_Simple_Shoot", False), "spell_exit": ("Spell_Simple_Exit", False),
    "crouch_idle": ("Crouch_Idle_Loop", True),
}
UAL2_CLIPS = {
    "hit_knockback": ("Hit_Knockback", False), "get_up": ("LayToIdle", False), "hook": ("Melee_Hook", False),
    "hook_rec": ("Melee_Hook_Rec", False), "block": ("Sword_Block", False), "slash_a": ("Sword_Regular_A", False),
    "slash_b": ("Sword_Regular_B", False), "slash_c": ("Sword_Regular_C", False), "throw": ("OverhandThrow", False),
    "flip_start": ("NinjaJump_Start", False), "flip_loop": ("NinjaJump_Idle_Loop", True),
    "flip_land": ("NinjaJump_Land", False), "slide": ("Slide_Start", False), "slide_exit": ("Slide_Exit", False),
    "dash_thrust": ("Sword_Dash_RM", False), "shield_idle": ("Idle_Shield_Loop", True),
}
ATTACK_CLIPS = {"jab", "cross", "hook", "kick_front", "kick_round", "kick_side", "slide", "throw",
                "sword_attack", "slash_a", "slash_b", "slash_c", "dash_thrust"}
WEAPON_CLIPS = {"sword_attack", "slash_a", "slash_b", "slash_c", "dash_thrust"}
ROOT_MOTION_CLIPS = {"dash_thrust", "slide", "slide_exit", "hit_knockback"}  # travel is applied by the game
ROOT_SCALE = {"dash_thrust": 0.3}  # the source dash covers ~3 m: far too much for a living room


def cmu_kick(tgt: Target, path: Path, fist, guard) -> Tuple[np.ndarray, np.ndarray]:
    """Cut the highest kick out of a karate trial, facing the kick forward."""
    names, dt, W = load_bvh(path)
    pos = {n: W[:, i] for i, n in enumerate(names)}
    sm = source_map("cmu")
    ground = np.percentile(np.minimum(pos["LeftToeBase"][:, 1], pos["RightToeBase"][:, 1]), 2)
    leg = np.median(np.linalg.norm(pos["LeftLeg"] - pos["LeftUpLeg"], axis=1)
                    + np.linalg.norm(pos["LeftFoot"] - pos["LeftLeg"], axis=1))
    heights = np.stack([pos["LeftFoot"][:, 1], pos["RightFoot"][:, 1]], 1) - ground
    peak = int(np.argmax(heights.max(1)))
    foot = ("LeftFoot", "RightFoot")[int(np.argmax(heights[peak]))]
    lifted = heights.max(1) > 0.25 * leg
    a = peak
    while a > 0 and lifted[a - 1]:
        a -= 1
    b = peak
    while b < len(lifted) - 1 and lifted[b + 1]:
        b += 1
    pad = int(0.55 / dt)
    a, b = max(0, a - pad), min(len(lifted), b + int(0.5 / dt))
    fwd = pos[foot][peak] - pos[sm["hips"]][peak]
    seg = resample_positions(pos, dt, a, b)
    R, H = tgt.retarget(seg, "cmu", forward=fwd, fist=fist)
    return ease_ends(R, H, *guard)


def cmu_guard(tgt: Target, path: Path, fist) -> Tuple[np.ndarray, np.ndarray]:
    """A 2.4 s boxing-stance loop: the calmest stretch of hands-up footwork in the trial."""
    names, dt, W = load_bvh(path)
    pos = {n: W[:, i] for i, n in enumerate(names)}
    hands = (pos["LeftHand"] + pos["RightHand"]) * 0.5
    speed = np.linalg.norm(np.diff(pos["LeftHand"], axis=0), axis=1) + np.linalg.norm(np.diff(pos["RightHand"], axis=0), axis=1)
    up = hands[:-1, 1] - pos["Spine1"][:-1, 1]
    n = int(2.4 / dt)
    score = np.convolve(speed, np.ones(n), "valid") - 50 * np.convolve(np.minimum(up, 0), np.ones(n), "valid")
    a = int(np.argmin(score[int(1 / dt):])) + int(1 / dt)
    fwd = np.mean(hands[a:a + n] - pos["Hips"][a:a + n], axis=0)
    seg = resample_positions(pos, dt, a, a + n)
    R, H = tgt.retarget(seg, "cmu", forward=fwd, fist=fist)
    return loop_seam(R, H)


def build(cache: Path, out: Path) -> None:
    print("Sources:")
    src = get_sources(cache)
    g1 = GLTF(src["ual1"])
    tgt = Target(g1)
    clips: Dict[str, Tuple[np.ndarray, np.ndarray, bool, np.ndarray]] = {}
    print("UAL1 clips")
    for name, (anim, loop) in UAL1_CLIPS.items():
        R, H = tgt.clip_from_gltf(anim)
        if loop:
            R, H = loop_seam(R, H)
        clips[name] = (R, H, loop, np.zeros(len(R)))
    fist = clips["jab"][0][0]                 # finger curl of a clenched fist
    guard = (clips["jab"][0][0], clips["jab"][1][0])  # fighting guard: first frame of the jab
    print("UAL2 clips (retargeted)")
    g2 = GLTF(src["ual2"])
    _, T2, R2 = g2.sample("A_TPose")
    W2 = g2.world(T2, R2)
    side = W2[0, g2.names.index("thigh_l"), :3, 3] - W2[0, g2.names.index("thigh_r"), :3, 3]
    ual2_fwd = np.cross(side, [0, 1, 0])
    for name, (anim, loop) in UAL2_CLIPS.items():
        _, T, R = g2.sample(anim)
        W = g2.world(T, R)
        pos = {n: W[:, i, :3, 3] for i, n in enumerate(g2.names)}
        R_, H_ = tgt.retarget(pos, "ual2", forward=ual2_fwd, fist=fist, keep_root_motion=name in ROOT_MOTION_CLIPS)
        root = tgt.last_root_fwd * ROOT_SCALE.get(name, 1.0)
        if loop:
            R_, H_ = loop_seam(R_, H_)
        clips[name] = (R_, H_, loop, root)
    print("CMU mocap clips (retargeted)")
    for name, trial in (("kick_front", "135_04"), ("kick_round", "135_07"), ("kick_side", "135_11")):
        R, H = cmu_kick(tgt, src[trial], fist, guard)
        clips[name] = (R, H, False, np.zeros(len(R)))
    R, H = cmu_guard(tgt, src["14_01"], fist)
    clips["guard"] = (R, H, True, np.zeros(len(R)))

    W0 = tgt.world(tgt.rest_r[None], tgt.rest_t[tgt.hips][None])[0]
    # Weapon grip axis (in the right hand's bind space): the native sword idle holds the blade up,
    # so "up" in that pose, taken perpendicular to the fingers, is the axis through the fist.
    hand = tgt.idx["DEF-hand.R"]
    Wi = tgt.world(clips["sword_idle"][0][:1], clips["sword_idle"][1][:1])[0]
    Rb, Rc = W0[hand, :3, :3], Wi[hand, :3, :3]
    up = Rb @ Rc.T @ np.array([0.0, 1.0, 0.0])
    finger = Rb[:, 1]
    grip_axis = up - finger * np.dot(up, finger)
    grip_axis /= np.linalg.norm(grip_axis)
    print(f"  grip axis (bind space): {np.round(grip_axis, 3)}")

    def grip_sign(n: str, t: Optional[float]) -> float:
        """Which way a held weapon should point in this clip: +1 along the grip axis, -1 reversed.
        At contact it points toward the target (forward); otherwise mostly up and forward."""
        R, H = clips[n][0], clips[n][1]
        f = int(round(t * FPS)) if t is not None else len(R) // 2
        W = tgt.world(R[f:f + 1], H[f:f + 1])[0]
        d = W[hand, :3, :3] @ Rb.T @ grip_axis
        want = np.array([0.0, 0.2, 1.0]) if t is not None else np.array([0.0, 1.0, 0.4])
        return 1.0 if float(np.dot(d, want)) >= 0 else -1.0

    meta_keys = ["limb", "contact", "active_start", "active_end", "reach", "strike_height", "grip"]
    names = list(clips)
    meta = np.full((len(names), len(meta_keys)), np.nan)
    for i, n in enumerate(names):
        if n in ATTACK_CLIPS:
            m = analyse(tgt, clips[n][0], clips[n][1], clips[n][3], weapon=n in WEAPON_CLIPS,
                        kick=n.startswith("kick"))
            m["grip"] = grip_sign(n, m["contact"])
            meta[i] = [m[k] for k in meta_keys]
            print(f"  {n:13s} limb {list(EFFECTORS)[int(m['limb'])]:7s} contact {m['contact']:.2f}s "
                  f"active {m['active_start']:.2f}-{m['active_end']:.2f}s reach {m['reach']:.2f}m "
                  f"height {m['strike_height']:.2f}m grip {m['grip']:+.0f}  ({len(clips[n][0])} frames)")
        else:
            meta[i, meta_keys.index("grip")] = grip_sign(n, None)

    print("Mesh")
    mesh = g1.j["meshes"][0]
    P, N, Jn, Wt, I, part = [], [], [], [], [], []
    base = 0
    for pi, prim in enumerate(mesh["primitives"]):
        at = prim["attributes"]
        p = g1.accessor(at["POSITION"]).astype(np.float32)
        P.append(p); N.append(g1.accessor(at["NORMAL"]).astype(np.float32))
        Jn.append(g1.accessor(at["JOINTS_0"]).astype(np.uint16)); Wt.append(g1.accessor(at["WEIGHTS_0"]).astype(np.float32))
        I.append(g1.accessor(prim["indices"]).astype(np.uint32) + base)
        part.append(np.full(len(p), pi, np.uint8))
        base += len(p)
    positions = np.concatenate(P)
    height =float(W0[tgt.idx["DEF-head"], 1, 3] + 0.2)  # head joint + crown
    arrays = dict(
        joint_names=np.array(tgt.names), parents=np.array(tgt.parents, np.int32), rest_t=tgt.rest_t,
        rest_r=tgt.rest_r, inv_bind=tgt.inv_bind, hips=np.int32(tgt.hips), above=tgt.above,
        positions=positions, normals=np.concatenate(N), joints=np.concatenate(Jn), weights=np.concatenate(Wt),
        indices=np.concatenate(I), part=np.concatenate(part), height=np.float32(max(height, positions[:, 1].max())),
        clip_names=np.array(names), clip_fps=np.full(len(names), FPS), clip_loop=np.array([clips[n][2] for n in names]),
        meta_keys=np.array(meta_keys), clip_meta=meta, grip_axis=grip_axis)
    for i, n in enumerate(names):
        arrays[f"rot_{i}"] = clips[n][0].astype(np.float32)
        arrays[f"hips_{i}"] = clips[n][1].astype(np.float32)
        arrays[f"root_{i}"] = clips[n][3].astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB): {len(names)} clips, "
          f"{len(positions)} vertices, {len(arrays['indices']) // 3} triangles, {len(tgt.names)} joints")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", type=Path, default=ROOT / "tools" / "cache")
    ap.add_argument("--out", type=Path, default=ROOT / "assets" / "character" / "fighter.npz")
    args = ap.parse_args()
    build(args.cache, args.out)
