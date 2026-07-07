#!/usr/bin/env python3
# =====================================================================
# Chronos v21 brain — Perception (BACKLOG Epic B, phase B1)  [IMPLEMENTED]
#
# Object-centric scene parsing for ARC-AGI-3 frames. Recent work shows a
# "perception bottleneck" on abstract-reasoning benchmarks: agents fail not
# because they cannot plan but because they never form the right OBJECTS to
# plan over (arXiv:2512.21329). This module turns a raw frame into a compact,
# JSON-serializable scene graph — connected-component objects with colour,
# size, bounding box and centroid — plus a frame-diff (what changed between
# two frames) and ACTION6 click-target extraction.
#
# Why connected components (and not the existing per-colour median): v19's
# `_dyn_clicks`/`_frame_objs` take ONE median centroid per colour, so several
# spatially-separate blobs of the same colour collapse to a single point that
# can land between them (on background). vc33-style click walls need a click
# ON each distinct component (its verified solutions hit separate coords like
# (12,56),(24,56),(34,56),(46,56) — four blobs of one row). True connected
# components give one target per blob (BACKLOG #6).
#
# Convention (matches the v19 engine): a frame is a 2-D grid of ints 0..15
# indexed grid[row][col]; ACTION6 click data is {'x': col, 'y': row}. Pure
# Python, no numpy/engine import, so `import brain.perception` works in the
# offline sandbox exactly like blitz.py.
# =====================================================================

_NCOLORS = 16


def to_grid(frame):
    """Normalise a frame to a list-of-lists of ints. Accepts a numpy array
    (any object exposing .tolist()), a list of rows, or a single flat row."""
    if frame is None:
        return []
    tl = getattr(frame, "tolist", None)
    if callable(tl):
        frame = tl()
    if not frame:
        return []
    first = frame[0]
    if isinstance(first, (list, tuple)):
        return [[int(v) for v in row] for row in frame]
    # a single flat row -> treat as one-row grid
    return [[int(v) for v in frame]]


def dims(grid):
    """(height, width). Width is taken from the first row (frames are rectangular)."""
    if not grid:
        return (0, 0)
    return (len(grid), len(grid[0]) if grid[0] else 0)


def histogram(grid):
    """Colour histogram: list of length 16, counts of each colour id."""
    h = [0] * _NCOLORS
    for row in grid:
        for v in row:
            if 0 <= v < _NCOLORS:
                h[v] += 1
    return h


def background_color(grid):
    """The most common colour (the presumed background)."""
    h = histogram(grid)
    best, bestc = 0, -1
    for c, n in enumerate(h):
        if n > bestc:
            best, bestc = c, n
    return best


def _median_int(vals):
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n else 0


def connected_components(grid, background=None, diagonal=False,
                         min_size=1, max_frac=0.5):
    """Flood-fill same-colour, non-background objects.

    Args:
      grid:        2-D list of ints (see to_grid).
      background:  colour to ignore; defaults to the modal colour.
      diagonal:    8-connectivity if True, else 4-connectivity.
      min_size:    drop objects smaller than this many cells.
      max_frac:    drop objects covering more than this fraction of the grid
                   (usually large fill/background-like regions).

    Returns a deterministic list (sorted by top,left,colour) of object dicts:
      {'color', 'size', 'bbox': (top,left,bottom,right),
       'centroid': (row,col), 'cells': [(row,col), ...]}
    Pure: no engine/network/global state.
    """
    H, W = dims(grid)
    if H == 0 or W == 0:
        return []
    if background is None:
        background = background_color(grid)
    total = H * W
    seen = [[False] * W for _ in range(H)]
    if diagonal:
        nbrs = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                (0, 1), (1, -1), (1, 0), (1, 1))
    else:
        nbrs = ((-1, 0), (1, 0), (0, -1), (0, 1))

    objects = []
    for r in range(H):
        for c in range(W):
            if seen[r][c]:
                continue
            color = grid[r][c]
            if color == background:
                seen[r][c] = True
                continue
            # iterative flood fill
            stack = [(r, c)]
            seen[r][c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in nbrs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr][nc] \
                            and grid[nr][nc] == color:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            size = len(cells)
            if size < min_size or size > total * max_frac:
                continue
            rows = [p[0] for p in cells]
            cols = [p[1] for p in cells]
            objects.append({
                "color": int(color),
                "size": size,
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
                "centroid": (_median_int(rows), _median_int(cols)),
                "cells": sorted(cells),
            })
    objects.sort(key=lambda o: (o["bbox"][0], o["bbox"][1], o["color"]))
    return objects


def scene(grid, diagonal=False, min_size=1, max_frac=0.5):
    """Full scene graph for a frame."""
    grid = to_grid(grid)
    H, W = dims(grid)
    bg = background_color(grid)
    objs = connected_components(grid, background=bg, diagonal=diagonal,
                                min_size=min_size, max_frac=max_frac)
    return {
        "dims": (H, W),
        "background": bg,
        "n_objects": len(objs),
        "objects": objs,
    }


def click_targets(grid, diagonal=False, min_size=1, max_frac=0.5, limit=None):
    """ACTION6 click targets: one per connected component, at its centroid.

    Returns a list of {'x': col, 'y': row} dicts (the engine's click-data
    convention), smallest object first (small distinct blobs are the usual
    selection targets, matching v19's size-sorted ordering), de-duplicated by
    (x, y). `limit` caps the count. Pure — the caller still verifies any plan
    a click seeds.
    """
    grid = to_grid(grid)
    objs = connected_components(grid, diagonal=diagonal,
                                min_size=min_size, max_frac=max_frac)
    objs = sorted(objs, key=lambda o: (o["size"], o["bbox"][0], o["bbox"][1]))
    out, seen = [], set()
    for o in objs:
        row, col = o["centroid"]
        key = (col, row)
        if key in seen:
            continue
        seen.add(key)
        out.append({"x": col, "y": row})
        if limit is not None and len(out) >= limit:
            break
    return out


def diff(grid0, grid1):
    """Per-cell change-set between two frames of the SAME dimensions.

    Returns {'changed': [(row,col), ...], 'n_changed', 'bbox': (t,l,b,r)|None,
    'appeared': [...], 'disappeared': [...], 'recolored': [...]} where the
    three lists are (row,col) partitioned by whether the cell went
    background->colour, colour->background, or colour->other-colour. If the
    grids differ in size, only the overlapping region is compared. Pure.
    """
    g0, g1 = to_grid(grid0), to_grid(grid1)
    H = min(dims(g0)[0], dims(g1)[0])
    W = min(dims(g0)[1], dims(g1)[1])
    bg0, bg1 = background_color(g0), background_color(g1)
    changed, appeared, disappeared, recolored = [], [], [], []
    for r in range(H):
        for c in range(W):
            a, b = g0[r][c], g1[r][c]
            if a == b:
                continue
            changed.append((r, c))
            if a == bg0 and b != bg1:
                appeared.append((r, c))
            elif a != bg0 and b == bg1:
                disappeared.append((r, c))
            else:
                recolored.append((r, c))
    bbox = None
    if changed:
        rs = [p[0] for p in changed]
        cs = [p[1] for p in changed]
        bbox = (min(rs), min(cs), max(rs), max(cs))
    return {
        "changed": changed,
        "n_changed": len(changed),
        "bbox": bbox,
        "appeared": appeared,
        "disappeared": disappeared,
        "recolored": recolored,
    }
