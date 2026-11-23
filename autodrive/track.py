"""Closed-loop 2D track geometry.

The track is stored as a dense centre-line polyline plus two boundary
polylines (left / right).  Everything the environment and the renderer need
is precomputed once and then queried analytically, so a step costs only a
handful of numpy calls.

Coordinate frame: metres, x to the right, y up, heading measured from +x.
"""
from __future__ import annotations

import numpy as np

# Default circuit: a stadium-like loop with a gentle S-bend on the top straight
# and a tighter final corner.  Control points are (x, y) in metres.
DEFAULT_CONTROL_POINTS = [
    (-78.0, -26.0),
    (-40.0, -26.0),
    (-14.0, -20.0),
    (14.0, -20.0),
    (40.0, -26.0),
    (78.0, -26.0),
    (96.0, -16.0),
    (98.0, 2.0),
    (88.0, 18.0),
    (66.0, 26.0),
    (34.0, 24.0),
    (8.0, 30.0),
    (-18.0, 24.0),
    (-48.0, 26.0),
    (-78.0, 26.0),
    (-96.0, 16.0),
    (-98.0, -2.0),
    (-92.0, -18.0),
]


def catmull_rom_closed(points, samples_per_segment: int = 24, alpha: float = 0.5) -> np.ndarray:
    """Centripetal Catmull-Rom spline through a closed set of control points.

    ``alpha=0.5`` (centripetal) avoids the cusps and self-intersections that a
    uniform (``alpha=0``) parameterisation produces when the control points are
    unevenly spaced.
    """
    P = np.asarray(points, dtype=float)
    n = len(P)
    chunks = []
    for i in range(n):
        p0, p1, p2, p3 = P[(i - 1) % n], P[i], P[(i + 1) % n], P[(i + 2) % n]
        t0 = 0.0
        t1 = float(np.linalg.norm(p1 - p0) ** alpha)
        t2 = t1 + float(np.linalg.norm(p2 - p1) ** alpha)
        t3 = t2 + float(np.linalg.norm(p3 - p2) ** alpha)
        if not (t0 < t1 < t2 < t3):
            continue
        t = np.linspace(t1, t2, samples_per_segment, endpoint=False)[:, None]
        a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
        b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
        chunks.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    return np.vstack(chunks)


def wrap_angle(a):
    """Wrap an angle (or array of angles) to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


class Track:
    """Dense polyline representation of a closed circuit."""

    def __init__(self, half_width: float = 4.0, control_points=None, samples_per_segment: int = 24):
        self.half_width = float(half_width)
        pts = catmull_rom_closed(
            DEFAULT_CONTROL_POINTS if control_points is None else control_points,
            samples_per_segment=samples_per_segment,
        )
        self.points = pts
        self.n_points = len(pts)

        # Tangents / normals by central difference on the closed polyline.
        nxt = np.roll(pts, -1, axis=0)
        prv = np.roll(pts, 1, axis=0)
        tang = nxt - prv
        self.segment_len = np.linalg.norm(nxt - pts, axis=1)
        self.tangents = tang / np.linalg.norm(tang, axis=1, keepdims=True)
        # Left-hand normal (rotate tangent by +90 deg).
        self.normals = np.stack([-self.tangents[:, 1], self.tangents[:, 0]], axis=1)

        self.s = np.concatenate([[0.0], np.cumsum(self.segment_len)])[: self.n_points]
        self.length = float(self.s[-1] + self.segment_len[-1])

        self.left = pts + self.normals * self.half_width
        self.right = pts - self.normals * self.half_width

        # Boundary segments (origin A + edge E) for ray casting.
        self._seg_a = np.vstack([self.left, self.right])
        seg_b = np.vstack([np.roll(self.left, -1, axis=0), np.roll(self.right, -1, axis=0)])
        self._seg_e = seg_b - self._seg_a
        self.half_count = self.n_points

        self.bbox = np.array(
            [
                [pts[:, 0].min() - self.half_width - 2.0, pts[:, 1].min() - self.half_width - 2.0],
                [pts[:, 0].max() + self.half_width + 2.0, pts[:, 1].max() + self.half_width + 2.0],
            ]
        )

    # ------------------------------------------------------------------ queries
    def curvature(self) -> np.ndarray:
        """Discrete curvature kappa_i = dtheta/ds (1/m)."""
        dtheta = wrap_angle(np.roll(self._heading(), -1) - self._heading())
        ds = self.segment_len
        return dtheta / np.maximum(ds, 1e-9)

    def _heading(self) -> np.ndarray:
        return np.arctan2(self.tangents[:, 1], self.tangents[:, 0])

    def min_turn_radius(self) -> float:
        k = np.abs(self.curvature())
        k = k[k > 1e-6]
        return float(1.0 / k.max()) if k.size else float("inf")

    def nearest_index(self, xy, hint: int | None = None, window: int = 90) -> int:
        """Index of the closest centre-line sample.

        With a ``hint`` only a local window is searched (the car moves at most
        ~1.5 m per step, so this is exact in practice and much cheaper).
        """
        p = np.asarray(xy, dtype=float).reshape(2)
        if hint is None:
            d = self.points - p
        else:
            idx = (hint + np.arange(-window, window + 1)) % self.n_points
            d = self.points[idx] - p
            return int(idx[np.argmin(np.einsum("ij,ij->i", d, d))])
        return int(np.argmin(np.einsum("ij,ij->i", d, d)))

    def project(self, xy, hint: int | None = None, window: int = 40):
        """Return ``(index, arc length s, signed lateral error)``.

        The lateral error is ``+`` when the point is left of the centre line.
        """
        i = self.nearest_index(xy, hint=hint, window=window)
        p = np.asarray(xy, dtype=float).reshape(2)
        rel = p - self.points[i]
        lateral = float(rel @ self.normals[i])
        s = float(self.s[i] + rel @ self.tangents[i])
        return i, s, lateral

    def raycast(self, origin, headings, max_range: float = 20.0, hint: int | None = None, window: int = 160):
        """Distance from ``origin`` along each heading to the track boundary.

        Vectorised ray/segment intersection against a window of boundary
        segments.  Returns an array of distances clipped to ``max_range``.
        """
        o = np.asarray(origin, dtype=float).reshape(2)
        h = np.asarray(headings, dtype=float).reshape(-1)
        d = np.stack([np.cos(h), np.sin(h)], axis=1)  # (R, 2)

        if hint is None:
            idx = np.arange(self.half_count)
        else:
            idx = (hint + np.arange(-window, window + 1)) % self.half_count
        a = np.vstack([self._seg_a[idx], self._seg_a[idx + self.half_count]])
        e = np.vstack([self._seg_e[idx], self._seg_e[idx + self.half_count]])

        # Solve  o + t*d = a + u*e  for every (ray, segment) pair.
        denom = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]  # cross(d, e)
        safe = np.abs(denom) > 1e-12
        denom = np.where(safe, denom, 1.0)
        ap = a[None, :, :] - o[None, None, :]
        t = (ap[..., 0] * e[None, :, 1] - ap[..., 1] * e[None, :, 0]) / denom
        u = (ap[..., 0] * d[:, None, 1] - ap[..., 1] * d[:, None, 0]) / (-denom)
        hit = safe & (t > 1e-6) & (t <= max_range) & (u >= 0.0) & (u <= 1.0)
        t = np.where(hit, t, np.inf)
        return np.minimum(t.min(axis=1), max_range)
