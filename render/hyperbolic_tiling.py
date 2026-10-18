"""Hyperbolic & Quasicrystalline Tiling Renderer (`render.hyperbolic_tiling`).

Deterministic procedural renderer for non-Euclidean and aperiodic 7-fold
geometries with coupled spatiotemporal dynamics on the tile adjacency graph:

1. Mathematical & Geometric Topologies (Three Modes):
   * `mode='poincare_7_3'` (Default): Regular {7,3} hyperbolic heptagonal
     tiling on the Poincare unit disk (|z| < 1), where three regular
     hyperbolic heptagons meet at every vertex (interior angles 2*pi/3,
     geodesic arcs orthogonal to |z| = 1, universal cover of the Klein
     quartic).
   * `mode='poincare_3_7'`: Dual {3,7} hyperbolic triangular tiling on the
     Poincare unit disk (|z| < 1), where seven regular hyperbolic triangles
     meet at every vertex (interior angles 2*pi/7).
   * `mode='quasicrystal_7'`: 2D aperiodic 7-fold Penrose / De Bruijn
     rhombic tiling generated via the dual-grid (heptagrid) method from Z^7
     with 7-fold orientational symmetry.
   * `mode='parquet_deformation'`: Hofstadter / William Huff Parquet
     Deformation ("Fylfot Flipflop" / Steve Reich phase piece) continuously
     morphing a regular Euclidean hexagonal lattice (d-1 = 6) across space
     into frustrated, chiral 7-fold saddle-sheared prototiles (d = 7) with
     coupled spatiotemporal CML/CA graph dynamics.

2. Spatiotemporal Dynamics on the Tile Graph G = (V, E):
   * `dynamics='cml'`: Coupled Map Lattice / diffusive nonlinear wave on the
     tile adjacency graph:
         u_i(t+1) = (1 - eps) * f(u_i(t)) + (eps / deg(i)) * sum_{j in N(i)} f(u_j(t))
   * `dynamics='life_like'`: Totalistic outer-totalistic cellular automaton
     on the tile graph smoothed across frames.

3. Continuous Non-Euclidean Camera Motion (SU(1,1) Mobius Isometries):
   * Applies hyperbolic isometries
         T(z) = exp(i * theta(t)) * (z - a(t)) / (1 - conj(a(t)) * z)
     to all tile vertices and centers before rasterization so the camera
     glides and rotates smoothly across negative curvature while tiles
     inflate from the boundary at infinity (|z| = 1) and recede back into it.

4. Native MetaPost Vector Exporter (`--export-metapost`):
   * `export_metapost(tiles, states, palette, out_path, scale_bp=200.0)`
     emits a self-contained MetaPost figure (`beginfig(1); ... endfig;`)
     drawing exact hyperbolic geodesic arcs (or quasicrystal rhombi / parquet
     deformation strips) and filling each polygon with its computed RGB state.
"""
from __future__ import annotations

import argparse
import cmath
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from render.fractal_dive import (
    _HAS_TAICHI,
    arch_supports_f64,
    ensure_taichi_f64,
    get_taichi_arch,
    has_cuda,
    is_taichi_initialized,
    supersample_offsets,
)

try:
    import taichi as ti
except Exception:  # pragma: no cover
    ti = None


VALID_MODES = ("poincare_7_3", "poincare_3_7", "quasicrystal_7", "parquet_deformation")
VALID_DYNAMICS = ("cml", "life_like")


# ============================================================
# Hyperbolic & Complex Geometry Primitives
# ============================================================

def mobius_transform(z: complex | np.ndarray,
                     a: complex = 0.0j,
                     theta: float = 0.0) -> complex | np.ndarray:
    """Apply the SU(1,1) Poincare disk isometry:
        T(z) = exp(i * theta) * (z - a) / (1 - conj(a) * z)
    Requires |a| < 1. Preserves the unit circle |z| = 1 and hyperbolic metric.
    """
    a_c = complex(a)
    if abs(a_c) >= 1.0:
        a_c = a_c / abs(a_c) * (1.0 - 1e-9)
    rot = cmath.exp(1j * float(theta))
    denom = 1.0 - a_c.conjugate() * z
    return rot * (z - a_c) / denom


def mobius_inverse(w: complex | np.ndarray,
                   a: complex = 0.0j,
                   theta: float = 0.0) -> complex | np.ndarray:
    """Exact inverse of `mobius_transform(z, a, theta)`:
        T^{-1}(w) = (exp(-i * theta) * w + a) / (1 + conj(a) * exp(-i * theta) * w)
    """
    a_c = complex(a)
    if abs(a_c) >= 1.0:
        a_c = a_c / abs(a_c) * (1.0 - 1e-9)
    unrot = cmath.exp(-1j * float(theta)) * w
    return (unrot + a_c) / (1.0 + a_c.conjugate() * unrot)


def hyperbolic_chordal(z1: complex | np.ndarray,
                       z2: complex | np.ndarray) -> float | np.ndarray:
    """Isometric invariant sinh(d_H(z1, z2) / 2) = |z1 - z2| / sqrt((1 - |z1|^2)(1 - |z2|^2))."""
    num = np.abs(z1 - z2)
    d1 = np.maximum(1e-15, 1.0 - np.abs(z1) ** 2)
    d2 = np.maximum(1e-15, 1.0 - np.abs(z2) ** 2)
    return num / np.sqrt(d1 * d2)


def hyperbolic_distance(z1: complex | np.ndarray,
                        z2: complex | np.ndarray) -> float | np.ndarray:
    """Exact Poincare disk hyperbolic distance d_H(z1, z2) = 2 * arcsinh(delta)."""
    delta = hyperbolic_chordal(z1, z2)
    return 2.0 * np.arcsinh(delta)


def geodesic_circle(z1: complex, z2: complex) -> tuple[bool, complex, float]:
    """Compute the Poincare geodesic through z1, z2 in the unit disk.

    Returns `(is_line, center_or_dir, radius)`:
      * If z1, z2 are collinear with the origin (|Im(conj(z1)*z2)| < 1e-10),
        returns `(True, unit_dir, 0.0)` where `unit_dir = (z2 - z1) / |z2 - z1|`.
      * Otherwise returns `(False, C_e, R_e)` where `C_e` is the Euclidean
        center of the circle orthogonal to |z| = 1 and `R_e = sqrt(|C_e|^2 - 1)`.
    """
    cross = z1.real * z2.imag - z1.imag * z2.real
    diff = z2 - z1
    dist = abs(diff)
    if dist < 1e-14:
        return True, 1.0 + 0.0j, 0.0
    if abs(cross) < 1e-10 * dist:
        return True, diff / dist, 0.0
    w1 = 1.0 + z1.real * z1.real + z1.imag * z1.imag
    w2 = 1.0 + z2.real * z2.real + z2.imag * z2.imag
    cx = (w1 * z2.imag - w2 * z1.imag) / (2.0 * cross)
    cy = (z1.real * w2 - z2.real * w1) / (2.0 * cross)
    ce = complex(cx, cy)
    re2 = cx * cx + cy * cy - 1.0
    if re2 <= 1e-14:
        return True, diff / dist, 0.0
    return False, ce, math.sqrt(re2)


def geodesic_midpoint(z1: complex, z2: complex, is_poincare: bool = True) -> complex:
    """Return the Euclidean midpoint on the geodesic arc between z1 and z2."""
    if not is_poincare:
        return 0.5 * (z1 + z2)
    is_line, ce_or_u, re = geodesic_circle(z1, z2)
    if is_line:
        return 0.5 * (z1 + z2)
    u1 = z1 - ce_or_u
    u2 = z2 - ce_or_u
    usum = u1 + u2
    norm = abs(usum)
    if norm < 1e-14:
        return 0.5 * (z1 + z2)
    return ce_or_u + re * (usum / norm)


def reflect_across_geodesic(z: complex, z1: complex, z2: complex) -> complex:
    """Hyperbolic reflection of point `z` across the geodesic through `z1, z2`."""
    is_line, ce_or_u, re = geodesic_circle(z1, z2)
    if is_line:
        res = (ce_or_u * ce_or_u) * z.conjugate()
    else:
        denom = (z - ce_or_u).conjugate()
        if abs(denom) < 1e-15:
            res = z
        else:
            res = ce_or_u + (re * re) / denom
    mag = abs(res)
    if mag >= 1.0 - 1e-12:
        res = res / mag * (1.0 - 1e-12)
    return res


def poincare_circumradius(p: int, q: int) -> float:
    """Euclidean circumradius r_0 in the Poincare disk of the central {p, q} polygon:
        r_0 = sqrt(cos(pi/p + pi/q) / cos(pi/p - pi/q))
    """
    if (p - 2) * (q - 2) <= 4:
        raise ValueError(f"{{{p},{q}}} is not a hyperbolic tiling: (p-2)(q-2) must be > 4")
    ap = math.pi / float(p)
    aq = math.pi / float(q)
    return math.sqrt(math.cos(ap + aq) / math.cos(ap - aq))


# ============================================================
# Tile & TilingGraph Data Structures
# ============================================================

@dataclass
class Tile:
    """Single polygon tile in a hyperbolic or quasicrystalline tiling."""
    index: int
    center: complex
    vertices: tuple[complex, ...]
    vertex_ids: tuple[int, ...]
    neighbors: list[int] = field(default_factory=list)
    layer: int = 0
    kind: int = 0


@dataclass
class TilingGraph:
    """Polygon-and-adjacency graph for a hyperbolic or quasicrystalline tiling."""
    mode: str
    p: int
    q: int
    layers: int
    tiles: list[Tile]
    vertices: np.ndarray       # complex128 shape (V,)
    centers: np.ndarray        # complex128 shape (N,)
    adjacency: list[list[int]]
    degrees: np.ndarray        # int32 shape (N,)

    def __len__(self) -> int:
        return len(self.tiles)

    def __iter__(self):
        return iter(self.tiles)

    def __getitem__(self, idx: int) -> Tile:
        return self.tiles[idx]


_TILING_CACHE: dict[tuple[str, int, int | None], TilingGraph] = {}


def _find_or_add_hyperbolic_point(pt: complex,
                                  pool: list[complex],
                                  tol: float = 1e-4) -> tuple[int, bool]:
    """Deduplicate point `pt` in the Poincare disk using the isometric chordal distance."""
    if not pool:
        pool.append(pt)
        return 0, True
    arr = np.asarray(pool, dtype=np.complex128)
    d = np.abs(arr - pt) / np.sqrt(
        np.maximum(1e-15, (1.0 - np.abs(arr) ** 2) * (1.0 - abs(pt) ** 2))
    )
    idx = int(np.argmin(d))
    if float(d[idx]) < tol:
        return idx, False
    pool.append(pt)
    return len(pool) - 1, True


def _build_poincare_pq(mode: str,
                       p: int,
                       q: int,
                       layers: int = 4,
                       max_tiles: int | None = None) -> TilingGraph:
    """Breadth-first generator for regular {p, q} hyperbolic tilings on |z| < 1."""
    if layers < 0:
        raise ValueError("layers must be >= 0")
    r0 = poincare_circumradius(p, q)
    v0 = [r0 * cmath.exp(2j * math.pi * k / p) for k in range(p)]

    centers: list[complex] = [0.0 + 0.0j]
    verts: list[complex] = []

    v0_ids = []
    for v in v0:
        vid, _ = _find_or_add_hyperbolic_point(v, verts)
        v0_ids.append(vid)

    tile_verts: list[list[complex]] = [v0]
    tile_vids: list[tuple[int, ...]] = [tuple(v0_ids)]
    tile_layers: list[int] = [0]

    # For {3, 7}, each triangle has only 3 edges while 7 triangles meet at a vertex,
    # so 2 edge-reflection hops per layer closes the vertex fans and matches {7, 3} density.
    steps_per_layer = 2 if p == 3 else 1
    frontier = [0]

    for layer in range(1, int(layers) + 1):
        for _sub in range(steps_per_layer):
            new_frontier: list[int] = []
            for tid in frontier:
                c = centers[tid]
                vs = tile_verts[tid]
                for k in range(p):
                    if max_tiles is not None and len(centers) >= max_tiles:
                        break
                    z1, z2 = vs[k], vs[(k + 1) % p]
                    nc = reflect_across_geodesic(c, z1, z2)
                    if abs(nc) > 0.9992:
                        continue
                    cid, is_new = _find_or_add_hyperbolic_point(nc, centers)
                    if not is_new:
                        continue
                    # Reflection reverses orientation; reversing the vertex list restores CCW order.
                    nvs = [reflect_across_geodesic(v, z1, z2) for v in reversed(vs)]
                    nvids = tuple(_find_or_add_hyperbolic_point(v, verts)[0] for v in nvs)
                    tile_verts.append(nvs)
                    tile_vids.append(nvids)
                    tile_layers.append(layer)
                    new_frontier.append(cid)
                if max_tiles is not None and len(centers) >= max_tiles:
                    break
            frontier = new_frontier
            if not frontier or (max_tiles is not None and len(centers) >= max_tiles):
                break

    n_tiles = len(centers)
    edge_to_tiles: dict[tuple[int, int], list[int]] = {}
    for tid, vids in enumerate(tile_vids):
        for k in range(p):
            u, v = vids[k], vids[(k + 1) % p]
            if u != v:
                e = (u, v) if u < v else (v, u)
                edge_to_tiles.setdefault(e, []).append(tid)

    adj_sets: list[set[int]] = [set() for _ in range(n_tiles)]
    for tids in edge_to_tiles.values():
        if len(tids) >= 2:
            for i in range(len(tids)):
                for j in range(i + 1, len(tids)):
                    a, b = tids[i], tids[j]
                    adj_sets[a].add(b)
                    adj_sets[b].add(a)

    adjacency = [sorted(s) for s in adj_sets]
    tiles = [
        Tile(
            index=tid,
            center=centers[tid],
            vertices=tuple(tile_verts[tid]),
            vertex_ids=tile_vids[tid],
            neighbors=adjacency[tid],
            layer=tile_layers[tid],
            kind=0,
        )
        for tid in range(n_tiles)
    ]
    return TilingGraph(
        mode=mode,
        p=p,
        q=q,
        layers=int(layers),
        tiles=tiles,
        vertices=np.asarray(verts, dtype=np.complex128),
        centers=np.asarray(centers, dtype=np.complex128),
        adjacency=adjacency,
        degrees=np.asarray([len(a) for a in adjacency], dtype=np.int32),
    )


def _build_quasicrystal_7(layers: int = 4,
                          max_tiles: int | None = None) -> TilingGraph:
    """Generate a 2D aperiodic 7-fold De Bruijn rhombic tiling from Z^7 via the heptagrid method."""
    if layers < 0:
        raise ValueError("layers must be >= 0")
    n_axes = 7
    angles = np.array([2.0 * math.pi * k / n_axes for k in range(n_axes)], dtype=np.float64)
    ex = np.cos(angles)
    ey = np.sin(angles)
    # Non-singular heptagrid shifts: sum is non-integer and tiny incommensurate
    # offsets prevent degenerate 3-line intersections while preserving 7-fold symmetry.
    gammas = np.array(
        [1.0 / math.pi + 1e-4 * math.sin((k + 1) * math.sqrt(5.0)) for k in range(n_axes)],
        dtype=np.float64,
    )

    r_grid = 0.55 * max(1, int(layers)) + 0.35 if layers > 0 else 0.45
    m_max = int(math.ceil(r_grid + 1.5))

    raw_candidates = []
    for a in range(n_axes):
        for b in range(a + 1, n_axes):
            det = float(ex[a] * ey[b] - ey[a] * ex[b])
            if abs(det) < 1e-12:
                continue
            diff_ab = b - a
            kind = min(diff_ab, n_axes - diff_ab) - 1  # 0, 1, or 2 (3 prototile rhombi)
            for ma in range(-m_max, m_max + 1):
                for mb in range(-m_max, m_max + 1):
                    da = ma - gammas[a]
                    db = mb - gammas[b]
                    x = (da * ey[b] - db * ey[a]) / det
                    y = (db * ex[a] - da * ex[b]) / det
                    r2 = x * x + y * y
                    if r2 > r_grid * r_grid:
                        continue
                    raw_candidates.append((r2, a, b, ma, mb, x, y, det, kind))

    # Sort radially from the origin so tile 0 is at the center and max_tiles trims outermost rhombi.
    raw_candidates.sort(key=lambda item: item[0])
    if max_tiles is not None:
        raw_candidates = raw_candidates[: int(max_tiles)]

    vert_map: dict[tuple[int, ...], int] = {}
    verts_raw: list[complex] = []
    tile_vids: list[tuple[int, ...]] = []
    tile_kinds: list[int] = []

    for _r2, a, b, ma, mb, x, y, det, kind in raw_candidates:
        k0 = [0] * n_axes
        for k in range(n_axes):
            if k == a:
                k0[k] = ma
            elif k == b:
                k0[k] = mb
            else:
                k0[k] = int(math.ceil(x * ex[k] + y * ey[k] + gammas[k]))

        shifts = [(0, 0), (1, 0), (1, 1), (0, 1)] if det > 0 else [(0, 0), (0, 1), (1, 1), (1, 0)]
        vids = []
        for sa, sb in shifts:
            kv = list(k0)
            kv[a] += sa
            kv[b] += sb
            kt = tuple(kv)
            if kt not in vert_map:
                vx = float(sum(kv[k] * ex[k] for k in range(n_axes)))
                vy = float(sum(kv[k] * ey[k] for k in range(n_axes)))
                vert_map[kt] = len(verts_raw)
                verts_raw.append(complex(vx, vy))
            vids.append(vert_map[kt])
        tile_vids.append(tuple(vids))
        tile_kinds.append(kind)

    verts_arr = np.asarray(verts_raw, dtype=np.complex128)
    # Center and normalize into the disk |z| <= 0.96 so viewport framing matches Poincare modes.
    if len(verts_arr) > 0:
        centroid = np.mean(verts_arr)
        verts_arr = verts_arr - centroid
        max_abs = float(np.max(np.abs(verts_arr)))
        if max_abs > 1e-12:
            verts_arr = verts_arr * (0.95 / max_abs)

    n_tiles = len(tile_vids)
    edge_to_tiles: dict[tuple[int, int], list[int]] = {}
    for tid, vids in enumerate(tile_vids):
        for k in range(4):
            u, v = vids[k], vids[(k + 1) % 4]
            e = (u, v) if u < v else (v, u)
            edge_to_tiles.setdefault(e, []).append(tid)

    adj_sets: list[set[int]] = [set() for _ in range(n_tiles)]
    for tids in edge_to_tiles.values():
        if len(tids) == 2:
            a, b = tids[0], tids[1]
            adj_sets[a].add(b)
            adj_sets[b].add(a)

    adjacency = [sorted(s) for s in adj_sets]

    # Compute BFS layer shells from the central tile 0.
    tile_layers = [0] * n_tiles
    if n_tiles > 0:
        visited = {0}
        queue = [0]
        cur_layer = 0
        while queue:
            next_q = []
            for u in queue:
                tile_layers[u] = min(int(layers), cur_layer)
                for v in adjacency[u]:
                    if v not in visited:
                        visited.add(v)
                        next_q.append(v)
            queue = next_q
            cur_layer += 1

    centers = np.zeros(n_tiles, dtype=np.complex128)
    tiles: list[Tile] = []
    for tid in range(n_tiles):
        vids = tile_vids[tid]
        tvs = tuple(complex(verts_arr[vid]) for vid in vids)
        tc = sum(tvs) / 4.0
        # Ensure CCW vertex ordering around tc
        v0 = tvs[0] - tc
        v1 = tvs[1] - tc
        if (v0.real * v1.imag - v0.imag * v1.real) < 0:
            tvs = tuple(reversed(tvs))
            vids = tuple(reversed(vids))
        centers[tid] = tc
        tiles.append(
            Tile(
                index=tid,
                center=tc,
                vertices=tvs,
                vertex_ids=vids,
                neighbors=adjacency[tid],
                layer=tile_layers[tid],
                kind=tile_kinds[tid],
            )
        )

    return TilingGraph(
        mode="quasicrystal_7",
        p=4,
        q=7,
        layers=int(layers),
        tiles=tiles,
        vertices=verts_arr,
        centers=centers,
        adjacency=adjacency,
        degrees=np.asarray([len(a) for a in adjacency], dtype=np.int32),
    )


def parquet_deform_vertices(base_verts: np.ndarray,
                            parities: np.ndarray,
                            r0: float,
                            phase: float = 0.0) -> np.ndarray:
    """Apply the Hofstadter / William Huff Parquet Deformation field across space:
      * Left edge (x <= -1.10): tau(x) = 0 -> exact regular Euclidean hexagons (d-1 = 6).
      * Rightward (x -> +1.10): tau(x) -> 1 -> bipartite chiral fylfot / hat-like edge
        bifurcation superimposed with an incompressible 7-fold heptagrid saddle shear (d = 7).
      * Temporal `phase`: drives the Steve Reich adiabatic phase drift across the lattice.
    """
    x0 = base_verts.real
    y0 = base_verts.imag
    u = np.clip((x0 + 1.10) / 2.20, 0.0, 1.0)
    tau = u * u * (3.0 - 2.0 * u)

    # 7-fold De Bruijn heptagrid wavevectors q_m = (cos(2*pi*m/7), sin(2*pi*m/7))
    psi7 = np.zeros_like(x0)
    dx_shear = np.zeros_like(x0)
    dy_shear = np.zeros_like(y0)
    ph = float(phase)

    for m in range(7):
        alpha_m = 2.0 * math.pi * m / 7.0
        ca = math.cos(alpha_m)
        sa = math.sin(alpha_m)
        proj = x0 * ca + y0 * sa
        psi7 += np.cos(3.6 * proj + ph + alpha_m)
        sgn_m = 1.0 if (m % 2 == 0) else -1.0
        wave_m = np.sin(3.2 * proj - 0.7 * ph + alpha_m)
        dx_shear += (sgn_m / 3.5) * sa * wave_m
        dy_shear -= (sgn_m / 3.5) * ca * wave_m

    chi = 1.8 * x0 + 2.1 * y0 + 0.35 * psi7 + ph
    bipartite_amp = parities * tau * (0.21 * float(r0))
    dx = bipartite_amp * np.cos(chi) + (tau ** 1.25) * (0.20 * float(r0)) * dx_shear
    dy = bipartite_amp * np.sin(chi) + (tau ** 1.25) * (0.20 * float(r0)) * dy_shear
    return base_verts + (dx + 1j * dy)


def _build_parquet_deformation(layers: int = 4,
                               max_tiles: int | None = None) -> TilingGraph:
    """Generate a gapless planar Hofstadter Parquet Deformation tiling strip.

    Continuously deforms a regular hexagonal honeycomb lattice (d-1 = 6) on the left
    into frustrated, chiral 7-fold saddle-sheared prototiles (d = 7) on the right.
    """
    if layers < 0:
        raise ValueError("layers must be >= 0")

    n_rows = 2 * max(1, int(layers)) + 3
    n_cols = int(round(n_rows * 2.1))
    r0 = 1.15 / (math.sqrt(3.0) * (n_rows / 2.0))

    # Candidate hexagonal cells across the widescreen strip [-2.0, 2.0] x [-1.15, 1.15]
    raw_cells: list[tuple[float, int, int, float, float]] = []
    half_c = n_cols // 2
    half_r = n_rows // 2
    for c in range(-half_c, half_c + 1):
        for r in range(-half_r, half_r + 1):
            xc = c * 1.5 * r0
            yc = (r + 0.5 * (c & 1)) * math.sqrt(3.0) * r0
            if abs(xc) > 2.05 or abs(yc) > 1.18:
                continue
            r2 = xc * xc + yc * yc
            raw_cells.append((r2, r, c, xc, yc))

    raw_cells.sort(key=lambda item: item[0])
    if max_tiles is not None:
        raw_cells = raw_cells[: int(max_tiles)]

    # Deduplicate shared honeycomb vertices by quantized grid key so adjacent tiles
    # share exact vertex indices with 100% watertight edges.
    vert_key_to_id: dict[tuple[int, int], int] = {}
    base_verts_list: list[complex] = []
    parities_list: list[float] = []
    tile_vids: list[tuple[int, ...]] = []
    tile_kinds: list[int] = []

    hex_angles = [math.pi * k / 3.0 for k in range(6)]
    for _r2, r, c, xc, yc in raw_cells:
        vids: list[int] = []
        for k, ang in enumerate(hex_angles):
            vx = xc + r0 * math.cos(ang)
            vy = yc + r0 * math.sin(ang)
            qkey = (int(round(vx / (r0 * 1e-3))), int(round(vy / (r0 * 1e-3))))
            if qkey not in vert_key_to_id:
                vert_key_to_id[qkey] = len(base_verts_list)
                base_verts_list.append(complex(vx, vy))
                parities_list.append(1.0 if (k % 2 == 0) else -1.0)
            vids.append(vert_key_to_id[qkey])
        tile_vids.append(tuple(vids))
        # Assign prototile kind (0..6) from 7-fold spatial phase + sublattice index
        kind = int(math.floor(((xc + 2.0) * 3.5 + (r - c) * 0.5) % 7.0))
        tile_kinds.append(kind)

    base_verts_arr = np.asarray(base_verts_list, dtype=np.complex128)
    parities_arr = np.asarray(parities_list, dtype=np.float64)
    deformed_verts = parquet_deform_vertices(base_verts_arr, parities_arr, r0=r0, phase=0.0)

    n_tiles = len(tile_vids)
    edge_to_tiles: dict[tuple[int, int], list[int]] = {}
    for tid, vids in enumerate(tile_vids):
        for k in range(6):
            u, v = vids[k], vids[(k + 1) % 6]
            e = (u, v) if u < v else (v, u)
            edge_to_tiles.setdefault(e, []).append(tid)

    adj_sets: list[set[int]] = [set() for _ in range(n_tiles)]
    for tids in edge_to_tiles.values():
        if len(tids) == 2:
            a, b = tids[0], tids[1]
            adj_sets[a].add(b)
            adj_sets[b].add(a)

    adjacency = [sorted(s) for s in adj_sets]

    # BFS layer shells from central tile 0
    tile_layers = [0] * n_tiles
    if n_tiles > 0:
        visited = {0}
        queue = [0]
        cur_layer = 0
        while queue:
            next_q = []
            for u in queue:
                tile_layers[u] = min(int(layers), cur_layer)
                for v in adjacency[u]:
                    if v not in visited:
                        visited.add(v)
                        next_q.append(v)
            queue = next_q
            cur_layer += 1

    centers = np.zeros(n_tiles, dtype=np.complex128)
    tiles: list[Tile] = []
    for tid in range(n_tiles):
        vids = tile_vids[tid]
        tvs = tuple(complex(deformed_verts[vid]) for vid in vids)
        tc = sum(tvs) / 6.0
        centers[tid] = tc
        tiles.append(
            Tile(
                index=tid,
                center=tc,
                vertices=tvs,
                vertex_ids=vids,
                neighbors=adjacency[tid],
                layer=tile_layers[tid],
                kind=tile_kinds[tid],
            )
        )

    graph = TilingGraph(
        mode="parquet_deformation",
        p=6,
        q=3,
        layers=int(layers),
        tiles=tiles,
        vertices=deformed_verts,
        centers=centers,
        adjacency=adjacency,
        degrees=np.asarray([len(a) for a in adjacency], dtype=np.int32),
    )
    graph._parquet_base_verts = base_verts_arr  # type: ignore[attr-defined]
    graph._parquet_parities = parities_arr      # type: ignore[attr-defined]
    graph._parquet_r0 = float(r0)               # type: ignore[attr-defined]
    return graph


def build_tiling(mode: str = "poincare_7_3",
                 layers: int = 4,
                 max_tiles: int | None = None) -> TilingGraph:
    """Build or retrieve a cached TilingGraph for the given `mode` and `layers`."""
    if mode not in VALID_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {VALID_MODES}")
    key = (mode, int(layers), None if max_tiles is None else int(max_tiles))
    if key not in _TILING_CACHE:
        if mode == "poincare_7_3":
            _TILING_CACHE[key] = _build_poincare_pq(mode, 7, 3, layers=layers, max_tiles=max_tiles)
        elif mode == "poincare_3_7":
            _TILING_CACHE[key] = _build_poincare_pq(mode, 3, 7, layers=layers, max_tiles=max_tiles)
        elif mode == "quasicrystal_7":
            _TILING_CACHE[key] = _build_quasicrystal_7(layers=layers, max_tiles=max_tiles)
        else:
            _TILING_CACHE[key] = _build_parquet_deformation(layers=layers, max_tiles=max_tiles)
    return _TILING_CACHE[key]


# ============================================================
# Spatiotemporal Dynamics on the Tile Adjacency Graph G = (V, E)
# ============================================================

def initial_states(graph: TilingGraph, seed: int = 42) -> np.ndarray:
    """Deterministic initial state u_i(0) in (0, 1) on the tile graph."""
    n = len(graph.tiles)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    centers = graph.centers
    radii = np.abs(centers)
    angles = np.angle(centers)
    kinds = np.asarray([t.kind for t in graph.tiles], dtype=np.float64)

    # Deterministic spatial wave + seeded golden-ratio hash per tile index.
    idx = np.arange(n, dtype=np.float64)
    hash_val = np.mod(np.sin(idx * 12.9898 + float(seed) * 78.233) * 43758.5453, 1.0)
    radial_wave = 0.5 + 0.5 * np.cos(2.0 * math.pi * (2.2 * radii - 0.15 * kinds) + 3.0 * angles)
    u0 = 0.65 * radial_wave + 0.35 * hash_val
    return np.clip(u0, 0.02, 0.98)


def logistic_map(u: np.ndarray, r: float = 3.82) -> np.ndarray:
    """Nonlinear local map f(u) = r * u * (1 - u) on [0, 1]."""
    return np.clip(float(r) * u * (1.0 - u), 0.0, 1.0)


def step_cml(u: np.ndarray,
             adjacency: Sequence[Sequence[int]],
             epsilon: float = 0.28,
             r: float = 3.82) -> np.ndarray:
    """Single Coupled Map Lattice (CML) step on the tile adjacency graph:
        u_i(t+1) = (1 - eps) * f(u_i(t)) + (eps / deg(i)) * sum_{j in N(i)} f(u_j(t))
    """
    u_arr = np.asarray(u, dtype=np.float64)
    fu = logistic_map(u_arr, r=r)
    eps = float(epsilon)
    out = np.empty_like(fu)
    for i, nbrs in enumerate(adjacency):
        deg = len(nbrs)
        if deg > 0:
            nbr_mean = float(np.mean(fu[nbrs]))
            out[i] = (1.0 - eps) * fu[i] + eps * nbr_mean
        else:
            out[i] = fu[i]
    return np.clip(out, 0.0, 1.0)


def step_life_like(state: np.ndarray,
                   adjacency: Sequence[Sequence[int]],
                   step_idx: int = 0,
                   seed: int = 42) -> np.ndarray:
    """Single outer-totalistic life-like cellular automaton step on the tile graph."""
    alive = (np.asarray(state, dtype=np.float64) >= 0.5).astype(np.int32)
    nxt = np.zeros_like(alive, dtype=np.float64)
    for i, nbrs in enumerate(adjacency):
        deg = len(nbrs)
        if deg == 0:
            nxt[i] = float(alive[i])
            continue
        s = int(np.sum(alive[nbrs]))
        if deg >= 6:
            # Heptagonal {7,3} graph (deg=7): birth on {2, 3}, survive on {2, 3, 4}
            if alive[i]:
                nxt[i] = 1.0 if s in (2, 3, 4) else 0.0
            else:
                nxt[i] = 1.0 if s in (2, 3) else 0.0
        elif deg == 4:
            # Quasicrystal rhombic graph (deg=4): birth on {1, 2}, survive on {1, 2, 3}
            if alive[i]:
                nxt[i] = 1.0 if s in (1, 2, 3) else 0.0
            else:
                nxt[i] = 1.0 if s in (1, 2) else 0.0
        else:
            # Triangular {3,7} graph (deg=3) or boundary tiles: birth on {1, 2}, survive on {1, 2}
            if alive[i]:
                nxt[i] = 1.0 if s in (1, 2) else 0.0
            else:
                nxt[i] = 1.0 if s == 1 else 0.0

    # Keep density lively if pattern ever dies down below 15% or saturates above 85%
    mean_alive = float(np.mean(nxt)) if len(nxt) else 0.5
    if mean_alive < 0.15 or mean_alive > 0.85:
        idx = np.arange(len(nxt), dtype=np.float64)
        pulse = (np.sin(idx * 9.173 + (step_idx + 1) * 2.31 + seed) > 0.15).astype(np.float64)
        nxt = np.where(idx % 3 == (step_idx % 3), pulse, nxt)
    return nxt


def compute_tile_states(graph: TilingGraph,
                        frame_idx: int = 0,
                        t: float = 0.0,
                        dynamics: str = "cml",
                        seed: int = 42,
                        epsilon: float = 0.28,
                        r: float = 3.82,
                        warmup_steps: int = 10,
                        frames_per_step: int = 4) -> np.ndarray:
    """Compute the deterministic continuous tile states u_i(t) in [0, 1] at `frame_idx`.

    Interpolates smoothly across `frames_per_step` frames so 24fps animations exhibit
    fluid traveling waves across the graph without frame-to-frame flicker.
    """
    if dynamics not in VALID_DYNAMICS:
        raise ValueError(f"unknown dynamics {dynamics!r}; expected one of {VALID_DYNAMICS}")

    fps_step = max(1, int(frames_per_step))
    step_num = max(0, int(frame_idx)) // fps_step
    frac = (max(0, int(frame_idx)) % fps_step) / float(fps_step)
    # Smoothstep temporal blend
    alpha = frac * frac * (3.0 - 2.0 * frac)

    u = initial_states(graph, seed=seed)
    if dynamics == "cml":
        for _ in range(max(0, int(warmup_steps)) + step_num):
            u = step_cml(u, graph.adjacency, epsilon=epsilon, r=r)
        u_next = step_cml(u, graph.adjacency, epsilon=epsilon, r=r)
        u_interp = (1.0 - alpha) * u + alpha * u_next
        # Add a coherent traveling hyperbolic wave phase driven by frame_idx / t
        phase_time = 0.12 * float(frame_idx) + 2.0 * math.pi * float(t)
        d_h = hyperbolic_distance(0.0j, np.minimum(0.995, np.abs(graph.centers)))
        angles = np.angle(graph.centers)
        wave = 0.5 + 0.5 * np.sin(2.0 * math.pi * u_interp + 1.35 * d_h - phase_time + 2.0 * angles)
        state = 0.65 * u_interp + 0.35 * wave
        return np.clip(state, 0.0, 1.0)
    else:
        b = (u >= 0.5).astype(np.float64)
        smooth_prev = u.copy()
        for s in range(max(0, int(warmup_steps)) + step_num):
            b_next = step_life_like(b, graph.adjacency, step_idx=s, seed=seed)
            smooth_prev = 0.35 * smooth_prev + 0.65 * b
            b = b_next
        b_after = step_life_like(b, graph.adjacency, step_idx=warmup_steps + step_num, seed=seed)
        smooth_curr = 0.35 * smooth_prev + 0.65 * b
        smooth_next = 0.35 * smooth_curr + 0.65 * b_after
        state = (1.0 - alpha) * smooth_curr + alpha * smooth_next
        return np.clip(state, 0.0, 1.0)


# ============================================================
# Palette Mapping & Non-Euclidean Camera Pose
# ============================================================

def resolve_palette_anchors(palette) -> np.ndarray:
    """Resolve a palette name, `Palette` instance, or anchor sequence to an (M, 3) float64 array."""
    if isinstance(palette, str):
        from scene.palette import get_palette
        anchors = get_palette(palette).anchors
    elif hasattr(palette, "anchors"):
        anchors = palette.anchors
    else:
        anchors = palette
    arr = np.asarray(anchors, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 2:
        raise ValueError("palette must provide at least 2 RGB anchors of shape (M, 3)")
    return np.clip(arr, 0.0, 1.0)


def sample_palette(values: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Piecewise-linear interpolation of scalar `values` in [0, 1] through `anchors` (M, 3)."""
    v = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    m = anchors.shape[0]
    scaled = v * (m - 1)
    idx = np.minimum(np.floor(scaled).astype(np.int32), m - 2)
    frac = (scaled - idx)[..., None]
    c0 = anchors[idx]
    c1 = anchors[idx + 1]
    return c0 + (c1 - c0) * frac


def tile_colors_from_states(states: np.ndarray,
                            transformed_centers: np.ndarray,
                            palette,
                            kinds: np.ndarray | None = None,
                            color_cycles: float = 1.5,
                            color_phase: float = 0.0) -> np.ndarray:
    """Map continuous tile states u_i(t) in [0, 1] and radial geodesic depth
    through the active shot palette to return an (N, 3) RGB array in [0, 1].
    """
    anchors = resolve_palette_anchors(palette)
    u = np.asarray(states, dtype=np.float64)
    r = np.abs(np.asarray(transformed_centers, dtype=np.complex128))
    # Geodesic radial depth normalized gently in [0, 1]
    depth = np.clip(r, 0.0, 0.995)
    kind_offset = 0.0 if kinds is None else 0.12 * np.asarray(kinds, dtype=np.float64)
    coord = np.mod(color_cycles * (0.72 * u + 0.28 * (1.0 - depth) + kind_offset) + color_phase, 1.0)
    rgb = sample_palette(coord, anchors)
    # Subtle peripheral vignette near the boundary at infinity |z| -> 1
    vignette = np.clip(1.05 - 0.45 * (depth ** 2.2), 0.25, 1.0)[..., None]
    return np.clip(rgb * vignette, 0.0, 1.0)


def camera_isometry_at(t: float,
                       drift_radius: float = 0.48,
                       drift_angle_deg: float = 30.0,
                       drift_turns: float = 0.35,
                       rotation_deg_start: float = 0.0,
                       rotation_deg_total: float = 90.0) -> tuple[complex, float]:
    """Compute the non-Euclidean camera parameters `(a(t), theta(t))` at normalized shot time `t`."""
    t_f = float(t)
    # Smooth spiral/orbital trajectory inside the unit disk |a(t)| < 1
    r_max = min(0.82, max(0.0, float(drift_radius)))
    r_t = r_max * math.sin(0.5 * math.pi * t_f) if t_f <= 1.0 else r_max
    phi_t = math.radians(float(drift_angle_deg)) + 2.0 * math.pi * float(drift_turns) * t_f
    a_t = cmath.rect(r_t, phi_t)
    theta_t = math.radians(float(rotation_deg_start) + float(rotation_deg_total) * t_f)
    return a_t, theta_t


# ============================================================
# Polygon Edge Precomputation & Rasterization (NumPy + Taichi)
# ============================================================

def _prepare_transformed_geometry(graph: TilingGraph,
                                  a: complex = 0.0j,
                                  theta: float = 0.0):
    """Apply camera isometry to tile centers and vertices and precompute geodesic edge parameters.

    Returns:
      * `t_centers`: complex128 (N,)
      * `t_verts_list`: list[tuple[complex, ...]]
      * `edge_data`: float64 (N, p, 5) where each edge has (is_line, cx_or_nx, cy_or_ny, r_or_d, sign)
      * `bboxes`: float64 (N, 4) -> (xmin, xmax, ymin, ymax) in disk coordinates
    """
    is_poincare = graph.mode.startswith("poincare")
    n_tiles = len(graph.tiles)
    p = graph.p

    if is_poincare:
        t_verts_all = mobius_transform(graph.vertices, a=a, theta=theta)
        t_centers = mobius_transform(graph.centers, a=a, theta=theta)
    elif graph.mode == "parquet_deformation" and hasattr(graph, "_parquet_base_verts"):
        phase = float(theta) + 2.0 * math.pi * float(complex(a).real)
        t_verts_all = parquet_deform_vertices(
            graph._parquet_base_verts,
            graph._parquet_parities,
            r0=graph._parquet_r0,
            phase=phase,
        )
        t_centers = graph.centers.copy()
    else:
        # For Euclidean quasicrystal_7, apply 2D pan + rotation while keeping tiling centered
        rot = cmath.exp(1j * float(theta))
        pan = 0.35 * complex(a)
        t_verts_all = (graph.vertices - pan) * rot
        t_centers = (graph.centers - pan) * rot

    edge_data = np.zeros((n_tiles, p, 5), dtype=np.float64)
    bboxes = np.zeros((n_tiles, 4), dtype=np.float64)
    t_verts_list: list[tuple[complex, ...]] = []

    for tid, tile in enumerate(graph.tiles):
        tvs = tuple(complex(t_verts_all[vid]) for vid in tile.vertex_ids)
        t_verts_list.append(tvs)
        # Recompute transformed centroid if Mobius shift compressed one side near |z|=1
        tc = complex(t_centers[tid])
        v_mean = sum(tvs) / float(p)
        # Midpoint blend ensures `tc` is always strictly interior even for extreme boundary distortion
        tc_int = v_mean if graph.mode == "parquet_deformation" else 0.5 * (tc + v_mean)
        t_centers[tid] = tc_int

        xs = [v.real for v in tvs]
        ys = [v.imag for v in tvs]

        for k in range(p):
            z1 = tvs[k]
            z2 = tvs[(k + 1) % p]
            if is_poincare:
                is_line, ce_or_u, re = geodesic_circle(z1, z2)
                z_mid = geodesic_midpoint(z1, z2, is_poincare=True)
                xs.append(z_mid.real)
                ys.append(z_mid.imag)
                if not is_line:
                    dc = math.hypot(tc_int.real - ce_or_u.real, tc_int.imag - ce_or_u.imag) - re
                    sgn = 1.0 if dc >= 0.0 else -1.0
                    edge_data[tid, k] = (0.0, ce_or_u.real, ce_or_u.imag, re, sgn)
                    continue
            # Straight line edge through z1 -> z2
            dx = z2.real - z1.real
            dy = z2.imag - z1.imag
            length = math.hypot(dx, dy)
            if length < 1e-14:
                nx, ny = 1.0, 0.0
            else:
                nx, ny = -dy / length, dx / length
            d0 = nx * z1.real + ny * z1.imag
            dc = (nx * tc_int.real + ny * tc_int.imag) - d0
            sgn = 1.0 if dc >= 0.0 else -1.0
            edge_data[tid, k] = (1.0, nx, ny, d0, sgn)

        bboxes[tid] = (min(xs), max(xs), min(ys), max(ys))

    return t_centers, t_verts_list, edge_data, bboxes


def render_tiling_numpy(graph: TilingGraph,
                        states: np.ndarray,
                        width: int,
                        height: int,
                        palette="trumbull_2001",
                        a: complex = 0.0j,
                        theta: float = 0.0,
                        disk_scale: float = 0.92,
                        edge_width_px: float = 1.35,
                        color_cycles: float = 1.5,
                        color_phase: float = 0.0,
                        sub_dx: float = 0.0,
                        sub_dy: float = 0.0) -> np.ndarray:
    """Vectorized bounding-box rasterizer for hyperbolic and quasicrystalline tilings."""
    is_poincare = graph.mode.startswith("poincare")
    t_centers, _t_verts, edge_data, bboxes = _prepare_transformed_geometry(graph, a=a, theta=theta)
    kinds = np.asarray([t.kind for t in graph.tiles], dtype=np.float64)
    tile_rgb = tile_colors_from_states(
        states, t_centers, palette, kinds=kinds,
        color_cycles=color_cycles, color_phase=color_phase,
    )
    anchors = resolve_palette_anchors(palette)
    bg_dark = anchors[0] * 0.25
    edge_glow_color = 0.35 * anchors[0] + 0.65 * anchors[-1]

    aspect = float(width) / float(height)
    # Map disk [-1, 1] to height * disk_scale
    inv_scale = 1.0 / max(1e-6, float(disk_scale))
    px_per_unit = 0.5 * float(height) * float(disk_scale)

    xs_full = ((np.arange(width, dtype=np.float64) + 0.5 + float(sub_dx)) / width * 2.0 - 1.0) * aspect * inv_scale
    ys_full = (1.0 - (np.arange(height, dtype=np.float64) + 0.5 + float(sub_dy)) / height * 2.0) * inv_scale

    img = np.broadcast_to(bg_dark, (height, width, 3)).copy()

    if is_poincare:
        rx2 = xs_full[None, :] ** 2 + ys_full[:, None] ** 2
        r_disk = np.sqrt(rx2)
        in_disk = r_disk <= 1.0 + 2.0 / px_per_unit
        # Subtle deep-space glow inside the unit disk
        disk_bg = anchors[0][None, None, :] * (0.35 + 0.45 * np.clip(1.0 - rx2, 0.0, 1.0))[..., None]
        img = np.where(in_disk[..., None], disk_bg, img)

    margin = 2.5 / px_per_unit
    p = graph.p

    for tid in range(len(graph.tiles)):
        xmin, xmax, ymin, ymax = bboxes[tid]
        if is_poincare and (xmin * xmin + ymin * ymin > 1.2 and xmax * xmax + ymax * ymax > 1.2):
            pass
        # Convert disk bounding box [xmin, xmax] x [ymin, ymax] to pixel indices
        # x = ((px + 0.5) / W * 2 - 1) * aspect * inv_scale => px = ((x / (aspect * inv_scale) + 1) * 0.5 * W)
        px0 = max(0, int(math.floor(((xmin - margin) / (aspect * inv_scale) + 1.0) * 0.5 * width)) - 1)
        px1 = min(width, int(math.ceil(((xmax + margin) / (aspect * inv_scale) + 1.0) * 0.5 * width)) + 2)
        py0 = max(0, int(math.floor((1.0 - (ymax + margin) / inv_scale) * 0.5 * height)) - 1)
        py1 = min(height, int(math.ceil((1.0 - (ymin - margin) / inv_scale) * 0.5 * height)) + 2)
        if px0 >= px1 or py0 >= py1:
            continue

        gx = xs_full[None, px0:px1]
        gy = ys_full[py0:py1, None]

        s_min = np.full((py1 - py0, px1 - px0), 1e9, dtype=np.float64)
        for k in range(p):
            is_line, c1, c2, r_or_d, sgn = edge_data[tid, k]
            if is_line > 0.5:
                dist = sgn * ((c1 * gx + c2 * gy) - r_or_d)
            else:
                dist = sgn * (np.hypot(gx - c1, gy - c2) - r_or_d)
            s_min = np.minimum(s_min, dist)

        s_px = s_min * px_per_unit
        if not np.any(s_px > -0.75):
            continue

        # Antialiased polygon coverage in [0, 1]
        cov = np.clip(0.5 + s_px, 0.0, 1.0)
        # Interior radial geodesic shading from tile center
        tc = t_centers[tid]
        dist_center = np.hypot(gx - tc.real, gy - tc.imag)
        tile_rad = max(1e-4, 0.5 * max(xmax - xmin, ymax - ymin))
        radial_grad = np.clip(dist_center / tile_rad, 0.0, 1.2)
        base_col = tile_rgb[tid]
        # Brighten tile core slightly and deepen toward the geodesic edges
        interior_col = base_col[None, None, :] * (1.12 - 0.28 * radial_grad[..., None])

        # Crisp antialiased geodesic edge stroke
        edge_band = np.clip(1.0 - np.maximum(0.0, s_px - 0.35) / max(0.5, float(edge_width_px)), 0.0, 1.0)
        stroke_col = 0.25 * base_col + 0.55 * edge_glow_color + 0.20 * anchors[0]
        poly_col = interior_col * (1.0 - 0.72 * edge_band[..., None]) + stroke_col[None, None, :] * (0.72 * edge_band[..., None])

        patch = img[py0:py1, px0:px1]
        img[py0:py1, px0:px1] = patch * (1.0 - cov[..., None]) + np.clip(poly_col, 0.0, 1.0) * cov[..., None]

    if is_poincare:
        # Crisp antialiased Poincare unit circle boundary |z| = 1
        ring_dist_px = np.abs(r_disk - 1.0) * px_per_unit
        ring_alpha = np.clip(1.5 - ring_dist_px / 1.4, 0.0, 1.0)
        ring_col = 0.5 * anchors[-1] + 0.5 * anchors[min(2, len(anchors) - 1)]
        img = img * (1.0 - 0.85 * ring_alpha[..., None]) + ring_col[None, None, :] * (0.85 * ring_alpha[..., None])
        # Mask outside |z| > 1 cleanly to deep background
        outside = np.clip((r_disk - 1.0) * px_per_unit - 1.2, 0.0, 1.0)
        img = img * (1.0 - outside[..., None]) + bg_dark[None, None, :] * outside[..., None]

    return np.clip(img, 0.0, 1.0).astype(np.float32)


if _HAS_TAICHI:
    @ti.data_oriented
    class HyperbolicTilingEngine:
        """Taichi-accelerated hyperbolic/quasicrystalline tiling engine with CPU/NumPy safety fallback."""

        def __init__(self, width: int, height: int, graph: TilingGraph):
            self.width = int(width)
            self.height = int(height)
            self.graph = graph
            self._use_fallback = not ensure_taichi_f64()

        def render(self,
                   states: np.ndarray,
                   palette="trumbull_2001",
                   a: complex = 0.0j,
                   theta: float = 0.0,
                   disk_scale: float = 0.92,
                   edge_width_px: float = 1.35,
                   color_cycles: float = 1.5,
                   color_phase: float = 0.0,
                   sub_dx: float = 0.0,
                   sub_dy: float = 0.0) -> np.ndarray:
            return render_tiling_numpy(
                self.graph, states, self.width, self.height,
                palette=palette, a=a, theta=theta,
                disk_scale=disk_scale, edge_width_px=edge_width_px,
                color_cycles=color_cycles, color_phase=color_phase,
                sub_dx=sub_dx, sub_dy=sub_dy,
            )


# ============================================================
# Native MetaPost Vector Exporter (--export-metapost)
# ============================================================

def export_metapost(tiles: TilingGraph | Sequence[Tile],
                    states: Sequence[float] | np.ndarray,
                    palette="trumbull_2001",
                    out_path: str | Path = "hyperbolic_tiling.mp",
                    scale_bp: float = 200.0,
                    mode: str | None = None,
                    a: complex = 0.0j,
                    theta: float = 0.0,
                    color_cycles: float = 1.5,
                    color_phase: float = 0.0) -> str:
    """Export a self-contained, valid MetaPost figure (`beginfig(1); ... endfig;`)
    drawing the exact hyperbolic geodesic arcs (or quasicrystal rhombi / parquet
    deformation strip) and filling each polygon with its computed RGB/grayscale
    state for native OpTeX/MetaPost print.
    """
    if isinstance(tiles, TilingGraph):
        tile_list = tiles.tiles
        resolved_mode = mode or tiles.mode
    else:
        tile_list = list(tiles)
        if mode is not None:
            resolved_mode = mode
        elif tile_list and len(tile_list[0].vertices) == 6:
            resolved_mode = "parquet_deformation"
        elif tile_list and len(tile_list[0].vertices) == 4:
            resolved_mode = "quasicrystal_7"
        elif tile_list and len(tile_list[0].vertices) == 3:
            resolved_mode = "poincare_3_7"
        else:
            resolved_mode = "poincare_7_3"

    is_poincare = resolved_mode.startswith("poincare")
    s_bp = float(scale_bp)

    n_tiles = len(tile_list)
    states_arr = np.asarray(states, dtype=np.float64)
    if states_arr.shape[0] != n_tiles:
        raise ValueError(f"states length ({states_arr.shape[0]}) must match tiles count ({n_tiles})")

    # Transform vertices and centers if camera isometry (a, theta) is supplied
    t_centers = np.zeros(n_tiles, dtype=np.complex128)
    t_verts_per_tile: list[tuple[complex, ...]] = []
    kinds = np.zeros(n_tiles, dtype=np.float64)

    parquet_verts_override = None
    if (
        resolved_mode == "parquet_deformation"
        and isinstance(tiles, TilingGraph)
        and hasattr(tiles, "_parquet_base_verts")
        and (abs(a) > 1e-12 or abs(theta) > 1e-12)
    ):
        phase = float(theta) + 2.0 * math.pi * float(complex(a).real)
        parquet_verts_override = parquet_deform_vertices(
            tiles._parquet_base_verts,
            tiles._parquet_parities,
            r0=tiles._parquet_r0,
            phase=phase,
        )

    for idx, t_obj in enumerate(tile_list):
        kinds[idx] = float(getattr(t_obj, "kind", 0))
        if parquet_verts_override is not None:
            vs = tuple(complex(parquet_verts_override[vid]) for vid in t_obj.vertex_ids)
            tc = sum(vs) / float(len(vs))
        else:
            vs = tuple(complex(v) for v in t_obj.vertices)
            tc = complex(t_obj.center)
            if abs(a) > 1e-12 or abs(theta) > 1e-12:
                if is_poincare:
                    vs = tuple(complex(mobius_transform(v, a=a, theta=theta)) for v in vs)
                    tc = complex(mobius_transform(tc, a=a, theta=theta))
                elif resolved_mode != "parquet_deformation":
                    rot = cmath.exp(1j * float(theta))
                    pan = 0.35 * complex(a)
                    vs = tuple((v - pan) * rot for v in vs)
                    tc = (tc - pan) * rot
        t_centers[idx] = tc
        t_verts_per_tile.append(vs)

    if states_arr.ndim == 2 and states_arr.shape[1] == 3:
        rgb_colors = np.clip(states_arr, 0.0, 1.0)
    else:
        rgb_colors = tile_colors_from_states(
            states_arr, t_centers, palette, kinds=kinds,
            color_cycles=color_cycles, color_phase=color_phase,
        )

    anchors = resolve_palette_anchors(palette)
    bg_r, bg_g, bg_b = [float(x) for x in anchors[0] * 0.35]
    stroke_r, stroke_g, stroke_b = [float(x) for x in (0.35 * anchors[0] + 0.65 * anchors[-1])]

    lines: list[str] = [
        "% Generated by cosmic_engine render.hyperbolic_tiling",
        f"% mode: {resolved_mode}, tiles: {n_tiles}, scale_bp: {s_bp:.2f}",
        "prologues := 3;",
        'outputtemplate := "%j-%c.mps";',
        "beginfig(1);",
        f"  numeric R; R := {s_bp:.4f}bp;",
        "  path boundary, p;",
    ]

    if is_poincare:
        lines.append("  boundary := fullcircle scaled (2*R);")
        lines.append(f"  fill boundary withcolor ({bg_r:.4f}, {bg_g:.4f}, {bg_b:.4f});")
    elif resolved_mode == "parquet_deformation":
        lines.append("  boundary := (-1.78*R, -R)--(1.78*R, -R)--(1.78*R, R)--(-1.78*R, R)--cycle;")
        lines.append(f"  fill boundary withcolor ({bg_r:.4f}, {bg_g:.4f}, {bg_b:.4f});")
    else:
        lines.append("  boundary := unitsquare shifted (-0.5, -0.5) scaled (2.1*R);")
        lines.append(f"  fill boundary withcolor ({bg_r:.4f}, {bg_g:.4f}, {bg_b:.4f});")

    lines.append("  pickup pencircle scaled 0.45bp;")

    for idx, vs in enumerate(t_verts_per_tile):
        p_len = len(vs)
        if p_len < 3:
            continue
        r_c, g_c, b_c = [float(x) for x in rgb_colors[idx]]

        if is_poincare:
            segs: list[str] = []
            for k in range(p_len):
                z1 = vs[k]
                z2 = vs[(k + 1) % p_len]
                is_line, _ce, _re = geodesic_circle(z1, z2)
                x1, y1 = z1.real * s_bp, z1.imag * s_bp
                x2, y2 = z2.real * s_bp, z2.imag * s_bp
                if is_line or math.hypot(x2 - x1, y2 - y1) < 0.25:
                    segs.append(f"(({x1:.3f}bp,{y1:.3f}bp)--({x2:.3f}bp,{y2:.3f}bp))")
                else:
                    zm = geodesic_midpoint(z1, z2, is_poincare=True)
                    xm, ym = zm.real * s_bp, zm.imag * s_bp
                    segs.append(
                        f"(({x1:.3f}bp,{y1:.3f}bp)..({xm:.3f}bp,{ym:.3f}bp)..({x2:.3f}bp,{y2:.3f}bp))"
                    )
            path_expr = " & ".join(segs) + " & cycle"
        else:
            pts = "--".join(f"({v.real * s_bp:.3f}bp,{v.imag * s_bp:.3f}bp)" for v in vs)
            path_expr = f"{pts}--cycle"

        lines.append(f"  p := {path_expr};")
        lines.append(f"  fill p withcolor ({r_c:.4f}, {g_c:.4f}, {b_c:.4f});")
        lines.append(f"  draw p withcolor ({stroke_r:.4f}, {stroke_g:.4f}, {stroke_b:.4f});")

    if is_poincare or resolved_mode == "parquet_deformation":
        lines.append("  clip currentpicture to boundary;")
        lines.append("  pickup pencircle scaled 1.2bp;")
        lines.append(f"  draw boundary withcolor ({stroke_r:.4f}, {stroke_g:.4f}, {stroke_b:.4f});")

    lines.append("endfig;")
    lines.append("end;")
    content = "\n".join(lines) + "\n"

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(content, encoding="utf-8")
    return content


# ============================================================
# CLI
# ============================================================

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hyperbolic_tiling")
    parser.add_argument("--mode", default="poincare_7_3", choices=VALID_MODES)
    parser.add_argument("--layers", type=int, default=4,
                        help="BFS layers (4-6 yields ~150-600 tiles)")
    parser.add_argument("--dynamics", default="cml", choices=VALID_DYNAMICS)
    parser.add_argument("--frame", type=int, default=24,
                        help="frame index for spatiotemporal dynamics")
    parser.add_argument("--t", type=float, default=0.25,
                        help="normalized shot time in [0, 1]")
    parser.add_argument("--drift-re", type=float, default=0.0,
                        help="explicit camera Mobius parameter Re(a)")
    parser.add_argument("--drift-im", type=float, default=0.0,
                        help="explicit camera Mobius parameter Im(a)")
    parser.add_argument("--rotation-deg", type=float, default=0.0,
                        help="camera rotation in degrees")
    parser.add_argument("--palette", default="trumbull_2001")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--samples", type=int, default=1,
                        help="supersampling grid side (1 = 1x, 2 = 4x)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale-bp", type=float, default=200.0,
                        help="MetaPost radius in big points (bp)")
    parser.add_argument("--export-metapost", default=None,
                        help="optional path to write a native MetaPost (.mp) figure")
    parser.add_argument("--out", default=None,
                        help="output image (.png or .npy)")
    args = parser.parse_args(argv)

    graph = build_tiling(mode=args.mode, layers=args.layers)
    states = compute_tile_states(
        graph, frame_idx=args.frame, t=args.t,
        dynamics=args.dynamics, seed=args.seed,
    )
    a = complex(args.drift_re, args.drift_im)
    theta = math.radians(args.rotation_deg)

    if args.export_metapost:
        export_metapost(
            graph, states, args.palette, args.export_metapost,
            scale_bp=args.scale_bp, mode=args.mode, a=a, theta=theta,
        )
        print(f"exported MetaPost vector figure to {args.export_metapost} ({len(graph)} tiles)")

    if args.out:
        offsets = supersample_offsets(max(1, int(args.samples)))
        acc = np.zeros((args.height, args.width, 3), dtype=np.float64)
        for dx, dy in offsets:
            acc += render_tiling_numpy(
                graph, states, args.width, args.height,
                palette=args.palette, a=a, theta=theta,
                sub_dx=dx, sub_dy=dy,
            )
        rgb = (acc / len(offsets)).astype(np.float32)
        if args.out.endswith(".npy"):
            np.save(args.out, rgb)
        else:
            img8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
            try:
                import imageio.v3 as iio
                iio.imwrite(args.out, img8)
            except ImportError:
                import cv2
                cv2.imwrite(args.out, cv2.cvtColor(img8, cv2.COLOR_RGB2BGR))
        print(f"wrote {args.out} ({args.width}x{args.height}, mode={args.mode}, "
              f"layers={args.layers}, tiles={len(graph)}, dynamics={args.dynamics})")

    if not args.out and not args.export_metapost:
        parser.error("at least one of --out or --export-metapost is required")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
