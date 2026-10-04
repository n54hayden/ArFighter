"""GPU renderer for the skinned fighter (moderngl, offscreen), composited into the pygame frame.

The character is skinned on the GPU, shaded as a shadow being (near-black body, rim light in
the boss colour, faintly glowing joint caps, emissive eyes) and rendered with 4x MSAA into an
offscreen buffer covering only its screen bounds; that region is read back and blitted.

Boss gear (staff, khopesh, headband, headdress, hood and robe) is real geometry bound to the
rig's joints, so it moves with the animation and gives exact collider endpoints.

Coordinates: the placement matrix maps model metres to screen pixels (x right, y down), so
joint world matrices from Rig.world(root=placement) are already in screen space.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame

from .character import CharacterData

try:
    import moderngl
except ImportError:  # the enemy falls back to a flat silhouette
    moderngl = None

VERT = """
#version 330
uniform mat4 bones[64];
uniform mat4 proj;
uniform mat3 nmat;
uniform mat4 gear_fix;   // held weapon: flipped end-for-end (with a twirl) when a clip holds it reversed
in vec3 in_pos; in vec3 in_norm; in vec4 in_joints; in vec4 in_weights; in float in_part;
out vec3 v_norm; out float v_part; out float v_height;
void main() {
    mat4 skin = bones[int(in_joints.x)] * in_weights.x + bones[int(in_joints.y)] * in_weights.y
              + bones[int(in_joints.z)] * in_weights.z + bones[int(in_joints.w)] * in_weights.w;
    vec4 lp = vec4(in_pos, 1.0);
    vec3 ln = in_norm;
    if (in_part > 3.5) { lp = gear_fix * lp; ln = mat3(gear_fix) * ln; }
    vec4 p = skin * lp;
    v_norm = normalize(nmat * mat3(skin) * ln);
    v_part = in_part;
    v_height = in_pos.y;
    gl_Position = proj * p;
}
"""

FRAG = """
#version 330
uniform vec3 body; uniform vec3 rim; uniform vec3 glow; uniform vec3 eyes; uniform vec3 gear;
uniform vec3 key_dir; uniform float flash; uniform float alpha; uniform float rim_side;
in vec3 v_norm; in float v_part; in float v_height;
out vec4 color;
void main() {
    vec3 n = normalize(v_norm);
    if (!gl_FrontFacing) n = -n;
    vec3 view = vec3(0.0, 0.0, 1.0);                        // screen space: the viewer looks down -z
    float fres = pow(1.0 - clamp(abs(dot(n, view)), 0.0, 1.0), 2.6);
    float key = max(dot(n, key_dir), 0.0);
    float side = rim_side == 0.0 ? 1.0 : 0.6 + 0.4 * clamp(n.x * rim_side, 0.0, 1.0);  // optional key side
    vec3 c;
    int part = int(v_part + 0.5);
    if (part == 1) {          // joint caps: dim inner glow
        c = body * 1.4 + glow * (0.35 + 0.35 * fres);
    } else if (part == 2) {   // eyes
        c = eyes;
    } else if (part >= 3) {   // gear (cloth / wood / metal): lit, readable against the body
        c = gear * (0.35 + 0.65 * key) + rim * fres * 0.6;
    } else {                  // body shell
        float ground = clamp(v_height / 1.8, 0.0, 1.0);    // a touch lighter toward the shoulders
        c = body * (0.85 + 0.3 * ground) + vec3(0.10, 0.10, 0.14) * key + rim * fres * 1.25 * side;
    }
    c = mix(c, vec3(1.0), flash);
    color = vec4(c, alpha);  // straight alpha (blitted with pygame's fast alpha path)
}
"""


@dataclass
class Look:
    """Per-boss appearance."""
    rim: Tuple[float, float, float]
    glow: Tuple[float, float, float]
    gear_color: Tuple[float, float, float] = (0.45, 0.35, 0.2)
    gear: Tuple[str, ...] = ()       # 'staff', 'sword', 'headband', 'headdress', 'robe'


def _rgb(c) -> Tuple[float, float, float]:
    return tuple(v / 255.0 for v in c)


# --- simple bound geometry -------------------------------------------------------------
def _cylinder(a: np.ndarray, b: np.ndarray, r0: float, r1: Optional[float] = None, seg: int = 10):
    r1 = r0 if r1 is None else r1
    axis = b - a
    length = np.linalg.norm(axis)
    z = axis / length
    x = np.cross(z, [0, 1, 0] if abs(z[1]) < 0.9 else [1, 0, 0]); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    pos, nor, idx = [], [], []
    for i in range(seg):
        ang = 2 * math.pi * i / seg
        d = math.cos(ang) * x + math.sin(ang) * y
        pos += [a + d * r0, b + d * r1]
        nor += [d, d]
    for i in range(seg):
        j = (i + 1) % seg
        idx += [2 * i, 2 * j, 2 * i + 1, 2 * j, 2 * j + 1, 2 * i + 1]
    # caps
    for end, r, sgn in ((a, r0, -1), (b, r1, 1)):
        c = len(pos)
        pos.append(end); nor.append(z * sgn)
        for i in range(seg):
            ang = 2 * math.pi * i / seg
            pos.append(end + (math.cos(ang) * x + math.sin(ang) * y) * r); nor.append(z * sgn)
        for i in range(seg):
            idx += [c, c + 1 + i, c + 1 + (i + 1) % seg]
    return np.array(pos), np.array(nor), np.array(idx)


def _sphere(c: np.ndarray, r: float, seg: int = 8):
    pos, nor, idx = [], [], []
    for i in range(seg + 1):
        th = math.pi * i / seg
        for j in range(seg * 2):
            ph = math.pi * j / seg
            d = np.array([math.sin(th) * math.cos(ph), math.cos(th), math.sin(th) * math.sin(ph)])
            pos.append(c + d * r); nor.append(d)
    w = seg * 2
    for i in range(seg):
        for j in range(w):
            a, b = i * w + j, i * w + (j + 1) % w
            idx += [a, b, a + w, b, b + w, a + w]
    return np.array(pos), np.array(nor), np.array(idx)


class _Builder:
    def __init__(self):
        self.parts: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = []

    def add(self, geo, joints, weights, part: float):
        pos, nor, idx = geo
        J = np.zeros((len(pos), 4)); W = np.zeros((len(pos), 4))
        J[:, :len(joints)] = joints
        W[:, :len(weights)] = weights
        self.parts.append((pos, nor, idx, J, W, part))

    def arrays(self):
        P, N, I, J, W, T = [], [], [], [], [], []
        base = 0
        for pos, nor, idx, j, w, part in self.parts:
            P.append(pos); N.append(nor); I.append(idx + base); J.append(j); W.append(w)
            T.append(np.full(len(pos), part)); base += len(pos)
        if not P:
            return None
        return (np.concatenate(P), np.concatenate(N), np.concatenate(I), np.concatenate(J), np.concatenate(W),
                np.concatenate(T))


class CharacterRenderer:
    SAMPLES = 4

    def __init__(self, data: CharacterData, max_size: Tuple[int, int]):
        if moderngl is None:
            raise RuntimeError("moderngl is not installed")
        self.data = data
        self.rig = data.rig
        self.ctx = moderngl.create_standalone_context()
        self.prog = self.ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        self.max_w, self.max_h = max_size
        size = (self.max_w, self.max_h)
        self._target_cache: Dict[Tuple[int, int], Tuple[object, object]] = {}
        self._buf = bytearray(self.max_w * self.max_h * 4)  # reused: reading into it is ~3x faster
        m = data.mesh
        self.base_vao = self._vao(m.positions, m.normals, m.joints, m.weights, m.indices, m.part.astype(np.float32))
        self.anchors = self._anchors()
        self._gear_vaos: Dict[Tuple[str, ...], Optional[object]] = {}

    def _targets(self, w: int, h: int):
        """(MSAA framebuffer, resolve framebuffer) of exactly w x h, created once per size."""
        key = (w, h)
        if key not in self._target_cache:
            if len(self._target_cache) > 48:  # bounded: release the oldest sizes
                for k in list(self._target_cache)[:16]:
                    for fb in self._target_cache.pop(k):
                        fb.release()
            msaa = self.ctx.framebuffer(
                color_attachments=[self.ctx.renderbuffer(key, 4, samples=self.SAMPLES)],
                depth_attachment=self.ctx.depth_renderbuffer(key, samples=self.SAMPLES))
            resolve = self.ctx.framebuffer(color_attachments=[self.ctx.renderbuffer(key, 4)])
            self._target_cache[key] = (msaa, resolve)
        return self._target_cache[key]

    def _vao(self, pos, nor, joints, weights, idx, part):
        data = np.hstack([pos.astype(np.float32), nor.astype(np.float32), joints.astype(np.float32),
                          weights.astype(np.float32), part.astype(np.float32).reshape(-1, 1)])
        vbo = self.ctx.buffer(np.ascontiguousarray(data).tobytes())
        ibo = self.ctx.buffer(np.ascontiguousarray(idx.astype(np.uint32)).tobytes())
        return self.ctx.vertex_array(self.prog, [(vbo, "3f 3f 4f 4f 1f", "in_pos", "in_norm", "in_joints",
                                                  "in_weights", "in_part")], ibo)

    # --- bind-pose landmarks used to attach gear ---------------------------------
    def _anchors(self) -> Dict[str, object]:
        m, rig = self.data.mesh, self.rig
        rest_world = rig.world(rig.rest_r, rig.rest_t[rig.hips], root=self.data.above)
        head = rig.index["DEF-head"]
        on_head = (m.joints[:, 0] == head) & (m.weights[:, 0] > 0.6)
        hp = m.positions[on_head]
        hand_r = rig.index["DEF-hand.R"]
        hw = rest_world[hand_r]
        grip = hw[:3, 3] + hw[:3, 1] * 0.075  # centre of the fist
        axis = self.data.grip_axis / np.linalg.norm(self.data.grip_axis)  # through the fist, thumb side
        bend = hw[:3, 1] - axis * np.dot(hw[:3, 1], axis)                   # toward the fingers
        return {"head_min": hp.min(0), "head_max": hp.max(0), "head": head, "hand_r": hand_r, "grip": grip,
                "axis": axis, "bend": bend / np.linalg.norm(bend),
                "hips": rig.hips, "thigh_l": rig.index["DEF-thigh.L"], "thigh_r": rig.index["DEF-thigh.R"],
                "spine": rig.index["DEF-spine.002"], "rest": rest_world}

    def gear_fix(self, angle: float) -> np.ndarray:
        """Bind-space transform for held weapons: rotate `angle` about the palm normal through the
        grip (pi = held reversed). Animating the angle makes the switch read as a twirl."""
        a = self.anchors
        n = np.cross(a["axis"], a["bend"])
        n /= np.linalg.norm(n)
        c, s = math.cos(angle), math.sin(angle)
        K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
        R = np.eye(3) + s * K + (1 - c) * K @ K
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = a["grip"] - R @ a["grip"]
        return M

    def gear_points(self, gear: Tuple[str, ...]) -> Dict[str, Tuple[int, np.ndarray]]:
        """Bind-space points on the gear, with the joint each follows (for colliders)."""
        a = self.anchors
        g, ax = a["grip"], a["axis"]
        pts = {}
        if "staff" in gear:
            pts["staff_back"] = (a["hand_r"], g - ax * 0.55)
            pts["staff_tip"] = (a["hand_r"], g + ax * 1.15)
        if "sword" in gear:
            for i, p in enumerate(self._blade_line()):
                pts[f"sword_{i}"] = (a["hand_r"], p)
        return pts

    def _blade_line(self) -> List[np.ndarray]:
        """Khopesh: straight from the fist, then a hook curving down and back."""
        a = self.anchors
        g, ax, bend = a["grip"], a["axis"], a["bend"]
        pts = [g, g + ax * 0.30]
        ang = 0.0
        p = pts[-1]
        for _ in range(4):
            ang += math.radians(22)
            p = p + (ax * math.cos(ang) + bend * math.sin(ang)) * 0.11
            pts.append(p)
        return pts

    def _gear_vao(self, gear: Tuple[str, ...]):
        if gear in self._gear_vaos:
            return self._gear_vaos[gear]
        a = self.anchors
        b = _Builder()
        hmin, hmax = a["head_min"], a["head_max"]
        hc = (hmin + hmax) * 0.5
        # Eyes: two small emissive spheres on the front of the head.
        eye_y = hmin[1] + (hmax[1] - hmin[1]) * 0.55
        for sx in (-1, 1):
            b.add(_sphere(np.array([hc[0] + sx * 0.032, eye_y, hmax[2] - 0.012]), 0.014, 6), [a["head"]], [1.0], 2)
        if "headband" in gear:
            y = hmin[1] + (hmax[1] - hmin[1]) * 0.7
            rx, rz = (hmax[0] - hmin[0]) * 0.53, (hmax[2] - hmin[2]) * 0.55
            ring_p, ring_n, ring_i = [], [], []
            n = 18
            for i in range(n):
                ang = 2 * math.pi * i / n
                d = np.array([math.cos(ang), 0, math.sin(ang)])
                c = np.array([hc[0], y, hc[2]]) + d * [rx, 0, rz]
                ring_p += [c + [0, 0.022, 0], c - [0, 0.022, 0]]; ring_n += [d, d]
            for i in range(n):
                j = (i + 1) % n
                ring_i += [2 * i, 2 * j, 2 * i + 1, 2 * j, 2 * j + 1, 2 * i + 1]
            b.add((np.array(ring_p), np.array(ring_n), np.array(ring_i)), [a["head"]], [1.0], 3)
            knot = np.array([hc[0], y, hmin[2] - 0.01])
            for k, dx in enumerate((-0.03, 0.03)):  # tails hanging behind
                b.add(_cylinder(knot, knot + [dx, -0.18 - 0.04 * k, -0.12], 0.012, 0.008, 6), [a["head"]], [1.0], 3)
        if "headdress" in gear:
            top = np.array([hc[0], hmax[1] + 0.005, hc[2]])
            b.add(_cylinder(top - [0, 0.07, 0], top, (hmax[0] - hmin[0]) * 0.56, (hmax[0] - hmin[0]) * 0.5, 14),
                  [a["head"]], [1.0], 3)
            for sx in (-1, 1):  # side lappets falling to the chest
                s = np.array([hc[0] + sx * (hmax[0] - hmin[0]) * 0.5, hmax[1] - 0.04, hc[2] - 0.01])
                b.add(_cylinder(s, s + [sx * 0.02, -0.26, 0.03], 0.035, 0.05, 8), [a["head"]], [1.0], 3)
            back = np.array([hc[0], hmax[1] - 0.05, hmin[2]])
            b.add(_cylinder(back, back + [0, -0.22, -0.05], 0.06, 0.1, 10), [a["head"]], [1.0], 3)
        if "robe" in gear:
            top = np.array([hc[0], hmax[1] + 0.04, hc[2] - 0.02])
            b.add(_cylinder(np.array([hc[0], hmin[1] - 0.02, hc[2] - 0.03]), top,
                            (hmax[0] - hmin[0]) * 0.72, 0.01, 12), [a["head"]], [1.0], 3)  # pointed hood
            hips = a["rest"][a["hips"]][:3, 3]
            tl, tr = a["rest"][a["thigh_l"]][:3, 3], a["rest"][a["thigh_r"]][:3, 3]
            # Skirt: a cone split in halves, each half following its thigh so the legs can move.
            for side, thigh, j in ((1, tl, a["thigh_l"]), (-1, tr, a["thigh_r"])):
                seg = 8
                pos, nor, idx = [], [], []
                for i in range(seg + 1):
                    ang = math.pi * i / seg - math.pi / 2
                    d = np.array([side * math.cos(ang), 0, math.sin(ang)])
                    pos += [hips + d * 0.16 + [0, 0.12, 0], np.array([thigh[0], hips[1] - 0.55, thigh[2]]) + d * 0.2]
                    nor += [d, d]
                for i in range(seg):
                    idx += [2 * i, 2 * i + 2, 2 * i + 1, 2 * i + 2, 2 * i + 3, 2 * i + 1]
                P = np.array(pos)
                J = np.zeros((len(P), 4)); W = np.zeros((len(P), 4))
                J[:, 0] = a["hips"]; J[:, 1] = j
                W[0::2, 0] = 1.0                      # waist follows the hips
                W[1::2, 0] = 0.25; W[1::2, 1] = 0.75  # hem follows the thigh
                b.parts.append((P, np.array(nor), np.array(idx), J, W, 3))
        if "staff" in gear:
            pts = self.gear_points(("staff",))
            b.add(_cylinder(pts["staff_back"][1], pts["staff_tip"][1], 0.018, 0.018, 8), [a["hand_r"]], [1.0], 4)
            for k in ("staff_back", "staff_tip"):
                b.add(_sphere(pts[k][1], 0.026, 5), [a["hand_r"]], [1.0], 4)
        if "sword" in gear:
            line = self._blade_line()
            g = line[0]
            b.add(_cylinder(g - a["axis"] * 0.12, g + a["axis"] * 0.02, 0.016, 0.016, 8), [a["hand_r"]], [1.0], 4)
            for p, q in zip(line[1:], line[2:]):
                b.add(_cylinder(p, q, 0.022, 0.02, 4), [a["hand_r"]], [1.0], 4)
            b.add(_cylinder(line[0], line[1], 0.012, 0.012, 4), [a["hand_r"]], [1.0], 4)
        arr = b.arrays()
        vao = None if arr is None else self._vao(arr[0], arr[1], arr[3], arr[4], arr[2], arr[5].astype(np.float32))
        self._gear_vaos[gear] = vao
        return vao

    # --- rendering ----------------------------------------------------------------
    def render(self, world: np.ndarray, rect: Tuple[int, int, int, int], look: Look, flash: float = 0.0,
               alpha: float = 1.0, rim_side: float = 0.0, gear_color=None, eyes=None,
               grip_angle: float = 0.0, latency: bool = False):
        """world: (J, 4, 4) joint matrices in screen space. rect: screen region to draw (x, y, w, h).

        Returns an RGBA surface for `rect` (or (surface, rect) with latency=True, kept for callers
        that want the rect back).
        """
        x0, y0, w, h = rect
        if w < 4 or h < 4:
            return (None, rect) if latency else None
        # Render into an exactly-sized target (sizes rounded up to 64 px): reading a whole buffer
        # back is several times faster than reading a sub-rectangle of a big one on many drivers.
        w = min(self.max_w, -(-w // 64) * 64)
        h = min(self.max_h, -(-h // 64) * 64)
        msaa, resolve = self._targets(w, h)
        skin = (world @ self.rig.inv_bind).astype(np.float32)
        bones = np.zeros((64, 4, 4), np.float32)
        bones[:len(skin)] = skin
        # screen px -> NDC (rows come back top-down, so y maps straight) ; depth from z
        proj = np.array([[2.0 / w, 0, 0, -1 - 2.0 * x0 / w],
                         [0, 2.0 / h, 0, -1 - 2.0 * y0 / h],
                         [0, 0, -1.0 / 2000.0, 0],
                         [0, 0, 0, 1]], np.float32)
        p = self.prog
        p["bones"].write(np.ascontiguousarray(bones.transpose(0, 2, 1)).tobytes())
        p["proj"].write(np.ascontiguousarray(proj.T).tobytes())
        p["gear_fix"].write(np.ascontiguousarray(self.gear_fix(grip_angle).astype(np.float32).T).tobytes())
        rot = world[self.rig.hips][:3, :3]
        nm = np.eye(3, dtype=np.float32)
        nm[1, 1] = -1  # screen y points down; light and view are set up in y-up terms
        p["nmat"].write(np.ascontiguousarray((nm / max(np.linalg.norm(rot[:, 0]), 1e-6)).T).tobytes())
        p["body"].value = (0.03, 0.026, 0.045)
        p["rim"].value = look.rim
        p["glow"].value = look.glow
        p["eyes"].value = eyes or look.glow
        p["gear"].value = gear_color or look.gear_color
        p["key_dir"].value = (0.35, 0.75, 0.55)
        p["flash"].value = float(flash)
        p["alpha"].value = float(alpha)
        p["rim_side"].value = float(rim_side)
        msaa.use()
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.clear(0, 0, 0, 0, depth=1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        self.base_vao.render()
        gear = self._gear_vao(look.gear)
        if gear is not None:
            gear.render()
        self.ctx.copy_framebuffer(resolve, msaa)
        resolve.read_into(self._buf, components=4)
        img = pygame.image.frombuffer(memoryview(self._buf)[:w * h * 4], (w, h), "RGBA").convert_alpha()
        return (img, (x0, y0, w, h)) if latency else img


def placement(x: float, ground_y: float, scale: float, facing: int, lift: float = 0.0,
              yaw_extra: float = 0.0, roll: float = 0.0, pivot: float = 0.0) -> np.ndarray:
    """Model metres -> screen pixels: feet at (x, ground_y), facing +1 = screen right.

    roll rotates the body about the screen's depth axis around a point `pivot` metres above the
    feet (e.g. the waist, for flips).
    """
    yaw = math.radians(90.0 * facing) + yaw_extra
    cy, sy = math.cos(yaw), math.sin(yaw)
    Ry = np.array([[cy, 0, sy, 0], [0, 1, 0, 0], [-sy, 0, cy, 0], [0, 0, 0, 1]])
    S = np.diag([scale, -scale, scale, 1.0])
    T = np.eye(4); T[0, 3] = x; T[1, 3] = ground_y - lift
    M = T @ S
    if roll:
        cr, sr = math.cos(roll), math.sin(roll)
        Rz = np.array([[cr, -sr, 0, 0], [sr, cr, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        up, down = np.eye(4), np.eye(4)
        up[1, 3], down[1, 3] = pivot, -pivot
        M = M @ up @ Rz @ down
    return M @ Ry
