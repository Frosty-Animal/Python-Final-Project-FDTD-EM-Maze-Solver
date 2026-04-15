"""
maze_em_solver.py
=================
Use the FDTD-derived first-arrival-time field to solve the maze.

Concept (the physical analog of the Fast Marching Method):
  When the EM pulse propagates through the maze, it explores every
  accessible corridor in parallel.  At each grid cell we record the
  first time the wave magnitude exceeds a small threshold; the
  resulting T_arrival(x, y) field is essentially the "geodesic
  distance" (in time) from the source to every reachable point.

  Because the wave equation enforces |grad T| = 1/c (the eikonal
  equation), a path from goal to source can be recovered by
  following T_arrival back down to zero.

This module:
  - trace_em_path(...)            Path goal -> source via the wave field
  - smooth_arrival_field(...)     Local averaging to remove sinusoidal noise
  - pixel_path_length(...)        Euclidean length of a pixel path
  - pixel_path_to_maze_cells()    Project pixel path onto maze cells

The path-tracer uses a two-stage strategy:
  Stage 1 - Greedy eikonal-aware descent.  At each step pick the
            unvisited neighbor that maximizes (T_curr - T_next)/step_len
            (the local approximation to the wave velocity 1/c).  This
            faithfully follows the wavefront backward.
  Stage 2 - Dijkstra fallback.  If greedy descent gets stuck in a local
            minimum (due to wave reflections creating noise), run
            Dijkstra on the reached-cells graph from goal to source with
            edge weight = step length.  This always succeeds if the wave
            actually reached the goal.
"""

import heapq
import numpy as np


# ============================================================
# Public API
# ============================================================
def trace_em_path(T_arrival, src_pix, goal_pix, pec_mask,
                  connectivity=8, smooth=True, verbose=False):
    """
    Trace a path from goal back to source through the FDTD arrival-time field.

    Parameters
    ----------
    T_arrival : (Nx, Ny) float ndarray
        First-arrival time at each pixel, with np.inf for unreached cells.
    src_pix, goal_pix : (i, j) tuples of ints
        Pixel locations of source and goal.
    pec_mask : (Nx, Ny) bool ndarray
        True where the cell is wall (excluded from the path).
    connectivity : 4 or 8
        4 = N/S/E/W moves only;  8 = also allow diagonals.
    smooth : bool
        If True, apply local averaging to T_arrival before tracing
        to remove pixel-scale noise from sinusoidal source oscillations.
    verbose : bool
        Print which stage produced the final path.

    Returns
    -------
    path : list of (i, j) pixel tuples, source -> goal
           or None if the wave never reached the goal.
    """
    src = tuple(int(x) for x in src_pix)
    goal = tuple(int(x) for x in goal_pix)

    if not np.isfinite(T_arrival[goal]):
        return None  # wave never made it

    T = smooth_arrival_field(T_arrival, window=5) if smooth else T_arrival
    offsets = _offsets(connectivity)

    # ----- Stage 1: eikonal-aware greedy descent ---------
    path = _greedy_descent(T, src, goal, pec_mask, offsets)
    if path is not None:
        if verbose:
            print(f"  [trace_em_path] greedy descent succeeded "
                  f"({len(path)} pixels)")
        return path

    # ----- Stage 2: Dijkstra fallback --------------------
    path = _dijkstra_on_reached(T, src, goal, pec_mask, offsets)
    if verbose:
        if path is None:
            print("  [trace_em_path] Dijkstra fallback also failed.")
        else:
            print(f"  [trace_em_path] greedy got stuck; Dijkstra fallback "
                  f"succeeded ({len(path)} pixels)")
    return path


# ============================================================
# Internal: descent strategies
# ============================================================
def _offsets(connectivity):
    """Return list of (di, dj, step_length) tuples for the given connectivity."""
    if connectivity == 8:
        return [(di, dj, float(np.hypot(di, dj)))
                for di in (-1, 0, 1) for dj in (-1, 0, 1)
                if (di, dj) != (0, 0)]
    elif connectivity == 4:
        return [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]
    else:
        raise ValueError("connectivity must be 4 or 8")


def _greedy_descent(T, src, goal, pec_mask, offsets):
    """
    Eikonal-aware greedy descent.  At each step pick the unvisited
    non-wall neighbor maximizing (T_curr - T_next)/step_len, requiring
    the ratio to be positive.  Returns None if it gets stuck before
    reaching the source.
    """
    Nx, Ny = T.shape
    path = [goal]
    visited = {goal}
    safety_limit = 8 * (Nx + Ny)

    while path[-1] != src:
        i, j = path[-1]
        cur_t = T[i, j]
        best_score = 0.0  # require strictly positive descent rate
        best_nb = None

        for di, dj, step in offsets:
            ni, nj = i + di, j + dj
            if not (0 <= ni < Nx and 0 <= nj < Ny):
                continue
            if pec_mask[ni, nj]:
                continue
            if (ni, nj) in visited:
                continue
            t_drop = cur_t - T[ni, nj]
            if t_drop <= 0:
                continue
            score = t_drop / step    # eikonal velocity proxy
            if score > best_score:
                best_score = score
                best_nb = (ni, nj)

        if best_nb is None:
            return None  # stuck; let caller fall back

        path.append(best_nb)
        visited.add(best_nb)

        if len(path) > safety_limit:
            return None

    return path[::-1]  # reverse: source -> goal


def _dijkstra_on_reached(T, src, goal, pec_mask, offsets):
    """
    Dijkstra from goal to source over the subgraph of cells the wave
    reached (T < inf).  Edge weight = geometric step length.
    Always finds the shortest path through reached cells if one exists.
    """
    Nx, Ny = T.shape
    dist = {goal: 0.0}
    parent = {goal: None}
    heap = [(0.0, goal)]

    while heap:
        d, node = heapq.heappop(heap)
        if node == src:
            break
        if d > dist.get(node, float('inf')):
            continue  # stale heap entry
        i, j = node
        for di, dj, step in offsets:
            ni, nj = i + di, j + dj
            if not (0 <= ni < Nx and 0 <= nj < Ny):
                continue
            if pec_mask[ni, nj]:
                continue
            if not np.isfinite(T[ni, nj]):
                continue  # restrict to reached cells only
            nb = (ni, nj)
            new_d = d + step
            if new_d < dist.get(nb, float('inf')):
                dist[nb] = new_d
                parent[nb] = node
                heapq.heappush(heap, (new_d, nb))

    if src not in parent:
        return None

    # Reconstruct: parents point from src back toward goal,
    # so walking from src gives src -> ... -> goal in correct order.
    path = []
    node = src
    while node is not None:
        path.append(node)
        node = parent[node]
    return path


# ============================================================
# Field smoothing
# ============================================================
def smooth_arrival_field(T, window=5):
    """
    Average T over a window x window neighborhood, ignoring inf values.
    Removes pixel-scale noise from sinusoidal source oscillations
    without changing the global gradient direction.

    window=3   : light smoothing (1-pixel halo)
    window=5   : default; effective for typical FDTD grids (2-pixel halo)
    window=7+  : heavier smoothing for very noisy fields
    """
    pad = window // 2
    Tp = np.pad(T, pad, mode='edge')
    out = np.zeros_like(T)
    count = np.zeros_like(T)
    for di in range(window):
        for dj in range(window):
            block = Tp[di:di + T.shape[0], dj:dj + T.shape[1]]
            valid = np.isfinite(block)
            out[valid] += block[valid]
            count[valid] += 1
    avg = np.full_like(T, np.inf)
    nz = count > 0
    avg[nz] = out[nz] / count[nz]
    return avg


# ============================================================
# Helpers for analysis / comparison with BFS
# ============================================================
def pixel_path_length(path):
    """Total Euclidean length of a path given as list of (i,j) pixel coords."""
    if path is None or len(path) < 2:
        return 0.0
    p = np.asarray(path, dtype=float)
    diffs = np.diff(p, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def pixel_path_to_maze_cells(path, meta):
    """
    Convert a pixel-coord path back to the unique sequence of maze
    (row, col) cells it passes through.  Useful for length comparison
    with the BFS solution.
    """
    if path is None:
        return None
    pix = meta['pix_per_cell']
    pad = meta['pad']
    out = []
    last = None
    for i, j in path:
        r = (i - pad) // pix
        c = (j - pad) // pix
        cell = (int(r), int(c))
        if cell != last:
            out.append(cell)
            last = cell
    return out
