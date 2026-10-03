"""2D collision primitives (pure math) plus debug drawing for them."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Union

import pygame

Vec = Tuple[float, float]


@dataclass(frozen=True)
class Circle:
    center: Vec
    radius: float


@dataclass(frozen=True)
class Capsule:
    """A line segment a-b swept by a radius (a limb)."""
    a: Vec
    b: Vec
    radius: float


Shape = Union[Circle, Capsule]


def closest_point_on_segment(p: Vec, a: Vec, b: Vec) -> Vec:
    abx, aby = b[0] - a[0], b[1] - a[1]
    denom = abx * abx + aby * aby
    if denom <= 1e-9:
        return a
    t = ((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / denom
    t = max(0.0, min(1.0, t))
    return (a[0] + abx * t, a[1] + aby * t)


def dist_point_segment(p: Vec, a: Vec, b: Vec) -> float:
    c = closest_point_on_segment(p, a, b)
    return math.hypot(p[0] - c[0], p[1] - c[1])


def _cross(o: Vec, a: Vec, b: Vec) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def segments_intersect(p1: Vec, p2: Vec, q1: Vec, q2: Vec) -> bool:
    d1, d2 = _cross(q1, q2, p1), _cross(q1, q2, p2)
    d3, d4 = _cross(p1, p2, q1), _cross(p1, p2, q2)
    return d1 * d2 < 0 and d3 * d4 < 0


def dist_segment_segment(p1: Vec, p2: Vec, q1: Vec, q2: Vec) -> float:
    if segments_intersect(p1, p2, q1, q2):
        return 0.0
    return min(
        dist_point_segment(p1, q1, q2),
        dist_point_segment(p2, q1, q2),
        dist_point_segment(q1, p1, p2),
        dist_point_segment(q2, p1, p2),
    )


def intersects(s1: Shape, s2: Shape) -> bool:
    if isinstance(s1, Circle) and isinstance(s2, Circle):
        return math.hypot(s1.center[0] - s2.center[0], s1.center[1] - s2.center[1]) <= s1.radius + s2.radius
    if isinstance(s1, Circle):
        return dist_point_segment(s1.center, s2.a, s2.b) <= s1.radius + s2.radius
    if isinstance(s2, Circle):
        return intersects(s2, s1)
    return dist_segment_segment(s1.a, s1.b, s2.a, s2.b) <= s1.radius + s2.radius


def _ipt(p: Vec) -> Tuple[int, int]:
    return int(p[0]), int(p[1])


def draw_shape(surf: pygame.Surface, shape: Shape, color, width: int = 2) -> None:
    """Outline a collider (used by the 'D' debug overlay)."""
    if isinstance(shape, Circle):
        pygame.draw.circle(surf, color, _ipt(shape.center), max(1, int(shape.radius)), width)
        return
    (ax, ay), (bx, by), r = shape.a, shape.b, shape.radius
    length = math.hypot(bx - ax, by - ay)
    pygame.draw.circle(surf, color, _ipt(shape.a), max(1, int(r)), width)
    pygame.draw.circle(surf, color, _ipt(shape.b), max(1, int(r)), width)
    if length > 1e-6:
        nx, ny = -(by - ay) / length * r, (bx - ax) / length * r
        pygame.draw.line(surf, color, (ax + nx, ay + ny), (bx + nx, by + ny), width)
        pygame.draw.line(surf, color, (ax - nx, ay - ny), (bx - nx, by - ny), width)
