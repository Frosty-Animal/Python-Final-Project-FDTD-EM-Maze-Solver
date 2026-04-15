"""
multipath_maze.py
=================
Generates a maze with MULTIPLE distinct paths from start to goal, for
testing whether a path-finding algorithm returns the actual *shortest*
path (not just any valid path).

The standard recursive-backtracker maze in maze.py is "perfect": between
any two cells there is exactly one path.  That makes it impossible to
test whether a solver finds the shortest route, because any valid route
IS the shortest one.

This module subclasses Maze and "braids" it by knocking down a chosen
fraction of interior walls.  Each removed wall closes a loop in the
maze graph, creating an alternative route.  A braiding fraction of:
    0.0  -> identical to the perfect maze (one path)
    0.2  -> mild braiding, a few alternative shortcuts
    0.5  -> heavy braiding, many loops
    1.0  -> "fully braided": no dead ends remain anywhere

Use it as a drop-in replacement for Maze:

    from multipath_maze import MultiPathMaze
    maze = MultiPathMaze(size=8, braid=0.3, seed=42)

Then pass it to fdtd_main's pipeline exactly like a normal Maze.

Standalone usage:
    python multipath_maze.py
prints a side-by-side ASCII view of the perfect and braided versions
and reports how many alternative paths exist.
"""

import numpy as np
from collections import deque
from maze import Maze, _FLOOR, _WALL


class MultiPathMaze(Maze):
    """A Maze with extra openings carved to create multiple solution paths."""

    def __init__(self, size=8, braid=0.3, seed=None,
                 force_path_diversity=True):
        """
        Parameters
        ----------
        size : int
            Maze dimension (number of cells per side).
        braid : float in [0, 1]
            Fraction of removable interior walls to knock down.
            0   = perfect maze (single path), identical to Maze.
            0.3 = a comfortable amount of branching for solver testing.
            1.0 = remove every wall whose removal would create a loop.
        seed : int or None
            RNG seed.  If None, uses a fresh random state.
        force_path_diversity : bool
            If True and the resulting maze still has only one path from
            start to goal, force at least one extra opening on the BFS
            shortest path so an alternative exists.  This guarantees a
            useful test maze even if random braiding happened to miss
            the start->goal corridor.
        """
        # Honor the seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Build the standard perfect maze first
        super().__init__(size)

        if not 0.0 <= braid <= 1.0:
            raise ValueError("braid must be in [0, 1]")
        self.braid = braid

        if braid > 0.0:
            self._braid(braid)

        if force_path_diversity and braid > 0.0:
            self._ensure_alternative_path()

    # --------------------------------------------------------
    def _braid(self, fraction):
        """
        Knock down a fraction of removable interior walls.

        A wall cell at position (i, j) in the display grid is "removable"
        if it currently blocks two open floor cells (i.e. its two
        opposite neighbors are both floors).  Removing it merges two
        previously distinct corridors and creates a loop.
        """
        candidates = []
        H, W = self.shape
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                if self.maze[i, j]:
                    continue  # already a floor
                # Horizontal-wall candidate: floors above and below
                if (self.maze[i - 1, j] and self.maze[i + 1, j]
                        and not self.maze[i, j - 1] and not self.maze[i, j + 1]):
                    candidates.append((i, j))
                # Vertical-wall candidate: floors left and right
                elif (self.maze[i, j - 1] and self.maze[i, j + 1]
                        and not self.maze[i - 1, j] and not self.maze[i + 1, j]):
                    candidates.append((i, j))

        if not candidates:
            return

        n_remove = int(round(fraction * len(candidates)))
        if n_remove == 0:
            return

        # np.random.choice doesn't take 2-tuples; sample indices instead
        idx = np.random.permutation(len(candidates))[:n_remove]
        for k in idx:
            i, j = candidates[k]
            self.maze[i, j] = _FLOOR

    # --------------------------------------------------------
    def _ensure_alternative_path(self):
        """
        If the maze has only one shortest path, knock down a wall adjacent
        to the BFS path to create at least one alternative.
        """
        if self.count_shortest_path_alternatives() >= 2:
            return  # already have alternative shortest paths

        # Otherwise look for any wall whose removal would create a loop
        if self.count_distinct_paths(max_paths=2) >= 2:
            return  # at least one loop exists somewhere

        # Find the BFS path, then look for any wall adjacent to it whose
        # removal would open a new connection
        start = tuple(2 * x + 1 for x in self.start_position)
        goal = tuple(2 * x + 1 for x in self.goal_position)
        bfs_path = self._bfs(start, goal)
        if bfs_path is None:
            return

        H, W = self.shape
        for (i, j) in bfs_path:
            for di, dj in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                wi, wj = i + di // 2, j + dj // 2  # the wall between
                ni, nj = i + di, j + dj            # the cell beyond
                if not (0 < wi < H - 1 and 0 < wj < W - 1):
                    continue
                if not (0 <= ni < H and 0 <= nj < W):
                    continue
                if not self.maze[wi, wj] and self.maze[ni, nj]:
                    # This wall blocks an existing floor; opening it
                    # creates a loop
                    self.maze[wi, wj] = _FLOOR
                    return

    # --------------------------------------------------------
    def _bfs(self, start, goal):
        """Return one shortest path (as list of cells) start -> goal, or None."""
        if not (self.maze[start] and self.maze[goal]):
            return None
        H, W = self.shape
        prev = {start: None}
        q = deque([start])
        while q:
            cur = q.popleft()
            if cur == goal:
                # Reconstruct
                path = []
                while cur is not None:
                    path.append(cur)
                    cur = prev[cur]
                return path[::-1]
            ci, cj = cur
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = ci + di, cj + dj
                if 0 <= ni < H and 0 <= nj < W and self.maze[ni, nj]:
                    if (ni, nj) not in prev:
                        prev[(ni, nj)] = cur
                        q.append((ni, nj))
        return None

    # --------------------------------------------------------
    def count_distinct_paths(self, max_paths=10):
        """
        Estimate the number of distinct routes from start to goal.

        Uses the cyclomatic complexity of the maze graph:
            n_loops = E - V + 1
        where V is the number of floor cells and E is the number of
        floor-floor adjacencies.  A perfect maze has n_loops = 0
        (one unique path between any two cells).  Each independent loop
        adds one alternative route somewhere in the maze, and at least
        a fraction of those typically affect start->goal connectivity.

        Returns
        -------
        int
            n_loops + 1, the number of independent paths in the graph.
            Capped at max_paths for display purposes.
        """
        H, W = self.shape
        floors = self.maze
        V = int(floors.sum())
        if V == 0:
            return 0

        # Count horizontal + vertical floor-floor adjacencies (each once)
        E = int(np.sum(floors[:-1, :] & floors[1:, :])
              + np.sum(floors[:, :-1] & floors[:, 1:]))

        # Cyclomatic number for a connected planar graph
        n_loops = max(0, E - V + 1)
        return min(max_paths, n_loops + 1)

    def count_shortest_path_alternatives(self, tolerance=0):
        """
        Count how many distinct shortest paths exist from start to goal.

        Uses BFS layer counting: at each cell, store the number of
        BFS-shortest paths reaching it; at the goal this is the total
        number of distinct shortest routes.

        Parameters
        ----------
        tolerance : int
            If 0, count only true-shortest paths.  If >0, also count
            paths within `tolerance` extra cells of the optimum.
            (tolerance > 0 not yet implemented; reserved for future.)
        """
        start = tuple(2 * x + 1 for x in self.start_position)
        goal = tuple(2 * x + 1 for x in self.goal_position)
        if not (self.maze[start] and self.maze[goal]):
            return 0

        H, W = self.shape
        dist = np.full(self.shape, -1, dtype=int)
        n_paths = np.zeros(self.shape, dtype=np.int64)
        dist[start] = 0
        n_paths[start] = 1

        q = deque([start])
        while q:
            cur = q.popleft()
            ci, cj = cur
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = ci + di, cj + dj
                if not (0 <= ni < H and 0 <= nj < W):
                    continue
                if not self.maze[ni, nj]:
                    continue
                if dist[ni, nj] == -1:
                    dist[ni, nj] = dist[cur] + 1
                    n_paths[ni, nj] = n_paths[cur]
                    q.append((ni, nj))
                elif dist[ni, nj] == dist[cur] + 1:
                    n_paths[ni, nj] += n_paths[cur]
        return int(n_paths[goal])

    # --------------------------------------------------------
    def display(self):
        """Display with a header noting the braiding fraction and path counts."""
        n_loops = self.count_distinct_paths(max_paths=99) - 1
        n_short = self.count_shortest_path_alternatives()
        print(f"[MultiPathMaze braid={self.braid:.2f}, "
              f"{n_loops} loops in graph, "
              f"{n_short} distinct shortest path(s) start->goal]")
        super().display()


# ============================================================
# Standalone demo
# ============================================================
if __name__ == "__main__":
    SEED = 42
    SIZE = 10

    print("=" * 60)
    print("PERFECT maze (single path):")
    print("=" * 60)
    np.random.seed(SEED)
    perfect = MultiPathMaze(size=SIZE, braid=0.0, seed=SEED,
                            force_path_diversity=False)
    perfect.display()

    print("\n" + "=" * 60)
    print(f"BRAIDED maze (braid=0.3, multiple paths):")
    print("=" * 60)
    braided = MultiPathMaze(size=SIZE, braid=0.3, seed=SEED)
    braided.display()

    print("\n" + "=" * 60)
    print("HEAVILY BRAIDED maze (braid=0.7):")
    print("=" * 60)
    heavy = MultiPathMaze(size=SIZE, braid=0.7, seed=SEED)
    heavy.display()
