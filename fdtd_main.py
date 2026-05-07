"""
fdtd_main.py
============
Entry point for the FDTD EM Maze Solver.

Prompts the user to generate a random maze (using the existing Maze class
from maze.py), runs a 2D Yee-grid FDTD simulation of an EM pulse launched
at the maze's start position, and animates the result.  Optionally
overlays the BFS shortest-path solution from maze_solver.py for visual
comparison between the wave's spread and the optimal route.

Place this file in the same directory as:
    maze.py
    maze_solver.py
    fdtd_engine.py

Run with:
    python fdtd_main.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from maze import Maze
from maze_solver import solve_maze
from multipath_maze import MultiPathMaze
from fdtd_engine import build_pec_mask, maze_cell_to_pixel, run_fdtd
from maze_em_solver import (trace_em_path, pixel_path_length,
                            pixel_path_to_maze_cells)
from fdtd_setup_ui import get_setup, show_restart_dialog


# Note: Setup prompts have been replaced by the Tkinter UI in
# fdtd_setup_ui.py.  The run_one_simulation() function below
# accepts a parameter dict from that UI.


# ============================================================
# Visualization
# ============================================================
def visualize(snapshots, times, pec_mask, src_pix, goal_pix,
              solution_path=None, em_path=None, meta=None,
              slowdown=2.0, material=None, source_mode='burst',
              export_format=None, export_basename='fdtd_animation'):
    """
    Animate the FDTD result.

    Parameters
    ----------
    snapshots, times : from run_fdtd
    pec_mask : bool array of wall locations
    src_pix, goal_pix : (i, j) pixel coords for source / goal markers
    solution_path : optional list of maze (r, c) tuples from BFS
    em_path : optional list of (i, j) pixel coords from FDTD solver
    meta : geometry metadata from build_pec_mask (needed for path overlay)
    slowdown : playback slowdown factor (1.0 = baseline 30 ms/frame,
               2.0 = 50% slower playback at 60 ms/frame)
    material : optional dict, used only for the figure title
    source_mode : 'burst' or 'continuous', used only for the title
    export_format : None, 'gif', or 'mp4'.  If set, the animation is
        saved to ./animations/<export_basename>_<timestamp>.<ext> before
        being displayed interactively.
    export_basename : string prefix for the saved filename
    """
    Nx, Ny = pec_mask.shape

    # Color scale chosen from later-time fields (after the source has rung up)
    late = snapshots[len(snapshots) // 3:]
    vmax = 0.3 * max(np.abs(s).max() for s in late)
    if vmax == 0:
        vmax = 1.0  # safety guard for trivial cases

    # Pre-build a transparent overlay for walls (black where wall)
    wall_rgba = np.zeros((Nx, Ny, 4))
    wall_rgba[..., 3] = pec_mask.astype(float)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(snapshots[0].T, origin='lower', cmap='RdBu',
                   vmin=-vmax, vmax=vmax, interpolation='nearest')
    ax.imshow(wall_rgba.transpose(1, 0, 2), origin='lower')

    # Optional BFS solution path overlay
    if solution_path is not None and meta is not None:
        pix_coords = np.array([maze_cell_to_pixel(rc, meta)
                               for rc in solution_path])
        ax.plot(pix_coords[:, 0], pix_coords[:, 1],
                color='lime', linewidth=2.0, alpha=0.7,
                label='BFS shortest path')

    # Optional FDTD-derived path overlay
    if em_path is not None:
        em_arr = np.array(em_path)
        ax.plot(em_arr[:, 0], em_arr[:, 1],
                color='magenta', linewidth=2.0, alpha=0.85,
                linestyle='--', label='EM-traced path')

    if solution_path is not None or em_path is not None:
        ax.legend(loc='upper left', fontsize=9, framealpha=0.85)

    ax.plot(src_pix[0],  src_pix[1],  marker='*', color='lime',
            markersize=16, markeredgecolor='black', label='Source')
    ax.plot(goal_pix[0], goal_pix[1], marker='X', color='gold',
            markersize=14, markeredgecolor='black', label='Goal')
    ax.set_xticks([]); ax.set_yticks([])
    title = ax.set_title(f't = {times[0]:.2f}')

    # Playback interval: 30 ms baseline; slowdown=2.0 -> 60 ms (50% slower)
    interval_ms = int(30 * slowdown)
    mat_label = material['name'] if material else 'pec'

    def update(frame):
        im.set_data(snapshots[frame].T)
        title.set_text(f't = {times[frame]:.2f}   '
                       f'(frame {frame+1}/{len(snapshots)},  '
                       f'walls: {mat_label},  '
                       f'src: {source_mode},  '
                       f'playback {1.0/slowdown:.0%} of baseline)')
        return [im, title]

    ani = FuncAnimation(fig, update, frames=len(snapshots),
                        interval=interval_ms, blit=False, repeat=True)
    plt.tight_layout()

    # Save to disk BEFORE showing interactively (plt.show blocks until close)
    if export_format in ('gif', 'mp4'):
        saved_path = _save_animation(ani, export_format, export_basename,
                                     interval_ms, n_frames=len(snapshots))
        if saved_path:
            print(f"  Animation saved to: {saved_path}")

    plt.show()
    return ani


def _save_animation(ani, fmt, basename, interval_ms, n_frames):
    """
    Render the FuncAnimation to disk as a GIF or MP4 file.

    Parameters
    ----------
    ani : matplotlib.animation.FuncAnimation
    fmt : 'gif' or 'mp4'
    basename : filename prefix (no extension)
    interval_ms : playback interval in ms, used to set the output FPS
    n_frames : used for a rough progress estimate message

    Returns
    -------
    pathlib.Path of the saved file, or None if saving failed.
    """
    import os
    from datetime import datetime
    from pathlib import Path
    from matplotlib.animation import PillowWriter, FFMpegWriter, writers

    out_dir = Path('animations')
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = out_dir / f"{basename}_{timestamp}.{fmt}"

    fps = max(1, int(round(1000.0 / max(1, interval_ms))))

    print(f"\nSaving animation ({fmt.upper()}, {n_frames} frames at {fps} fps).")
    print("  This may take a moment...")

    try:
        if fmt == 'gif':
            writer = PillowWriter(fps=fps)
        elif fmt == 'mp4':
            if 'ffmpeg' not in writers.list():
                print("  ERROR: MP4 export needs ffmpeg, but it was not")
                print("         found on your system.  Install ffmpeg from")
                print("         https://ffmpeg.org/download.html and put it")
                print("         on your PATH, or choose GIF instead.")
                return None
            writer = FFMpegWriter(fps=fps, bitrate=2400)
        else:
            print(f"  Unknown format: {fmt!r}")
            return None

        ani.save(str(out_path), writer=writer, dpi=110)
        return out_path

    except Exception as e:
        print(f"  ERROR saving animation: {type(e).__name__}: {e}")
        if out_path.exists():
            try:
                os.remove(out_path)
            except OSError:
                pass
        return None


def plot_arrival_field(T_arrival, pec_mask, src_pix, goal_pix,
                       em_path=None, solution_path=None, meta=None,
                       n_contours=15):
    """
    Static figure showing the first-arrival-time field as an
    isochrone (topographic-style) map.  Cells the wave never
    reached are shown blank.
    """
    Nx, Ny = T_arrival.shape

    # Mask unreached cells and walls so they don't affect the colormap range
    T_plot = np.where(np.isinf(T_arrival), np.nan, T_arrival)
    T_plot = np.where(pec_mask, np.nan, T_plot)

    fig, ax = plt.subplots(figsize=(8, 8))
    finite_max = np.nanmax(T_plot)

    if not np.isfinite(finite_max):
        ax.text(0.5, 0.5, "No cells were reached by the wave.",
                ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        plt.show()
        return

    # Filled colormap of arrival time
    im = ax.imshow(T_plot.T, origin='lower', cmap='plasma',
                   vmin=0, vmax=finite_max, interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('First-arrival time t (normalized)')

    # Isochrone contours on top of the colormap
    levels = np.linspace(0, finite_max, n_contours + 2)[1:-1]
    xs = np.arange(Nx)
    ys = np.arange(Ny)
    ax.contour(xs, ys, T_plot.T, levels=levels,
               colors='white', linewidths=0.6, alpha=0.5)

    # Walls in dark grey
    wall_rgba = np.zeros((Nx, Ny, 4))
    wall_rgba[..., 0] = 0.15
    wall_rgba[..., 1] = 0.15
    wall_rgba[..., 2] = 0.15
    wall_rgba[..., 3] = pec_mask.astype(float)
    ax.imshow(wall_rgba.transpose(1, 0, 2), origin='lower')

    # Path overlays
    if solution_path is not None and meta is not None:
        pix = np.array([maze_cell_to_pixel(rc, meta) for rc in solution_path])
        ax.plot(pix[:, 0], pix[:, 1], color='lime', linewidth=2.0,
                alpha=0.85, label='BFS shortest path')
    if em_path is not None:
        em_arr = np.array(em_path)
        ax.plot(em_arr[:, 0], em_arr[:, 1], color='cyan', linewidth=2.0,
                alpha=0.95, linestyle='--', label='EM-traced path')

    ax.plot(src_pix[0],  src_pix[1],  marker='*', color='lime',
            markersize=16, markeredgecolor='black', label='Source')
    ax.plot(goal_pix[0], goal_pix[1], marker='X', color='gold',
            markersize=14, markeredgecolor='black', label='Goal')

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('First-arrival time field (isochrones).\n'
                 'Wave-derived path = gradient descent on this field.')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.85)
    plt.tight_layout()
    plt.show()


# ============================================================
# Single simulation runner (parameters come from the UI dict)
# ============================================================
def run_one_simulation(params):
    """
    Execute one full FDTD simulation given a parameter dict from the UI.

    Parameter dict keys:
        size           int
        braid          float in [0, 1]
        show_bfs       bool
        material       dict (one of MATERIAL_PRESETS values)
        source_mode    'burst' or 'continuous'
        use_em_solver  bool
        duration_mult  float (multiplier on default n_steps)
        slowdown       float (animation playback slowdown)
    """
    print()
    print("=" * 60)
    print("  FDTD ELECTROMAGNETIC MAZE SOLVER")
    print("=" * 60)

    # ---- Maze ----
    size = params['size']
    braid = params['braid']
    print(f"\nGenerating a {size}x{size} maze (braid={braid:.2f})...")
    if braid > 0.0:
        maze = MultiPathMaze(size, braid=braid)
        n_loops = maze.count_distinct_paths(max_paths=99) - 1
        n_short = maze.count_shortest_path_alternatives()
        print(f"  Graph contains {n_loops} loops; "
              f"{n_short} distinct shortest path(s) from start to goal.")
    else:
        maze = Maze(size)
    maze.display()

    # ---- BFS overlay ----
    solution_path = solve_maze(maze) if params['show_bfs'] else None
    if solution_path is not None:
        print(f"\nBFS shortest path: {len(solution_path)} cells.")

    # ---- FDTD geometry ----
    print("\nBuilding FDTD grid...")
    pec_mask, meta = build_pec_mask(maze, pix_per_cell=8, pad=10)
    start_disp = tuple(2 * x + 1 for x in maze.start_position)
    goal_disp  = tuple(2 * x + 1 for x in maze.goal_position)
    src_pix  = maze_cell_to_pixel(start_disp, meta)
    goal_pix = maze_cell_to_pixel(goal_disp,  meta)
    print(f"  Grid size:  {meta['Nx']} x {meta['Ny']} cells")
    print(f"  Source at maze cell {start_disp}  -> pixel {src_pix}")
    print(f"  Goal   at maze cell {goal_disp}   -> pixel {goal_pix}")

    # ---- Material + source mode ----
    material = params['material']
    source_mode = params['source_mode']
    use_em_solver = params['use_em_solver']
    print(f"\nUsing wall material: {material['name']}  ({material['desc']})")
    print(f"Source mode:         {source_mode}")
    if use_em_solver and source_mode == 'continuous':
        print()
        print("  NOTE: continuous-wave source can corrupt arrival-time")
        print("  detection because reflected energy from the steady-state")
        print("  field may trip the threshold before the true wavefront.")
        print("  The EM-derived path may not be reliable.")

    # ---- Duration ----
    default_steps = max(1500, 200 * size)
    n_steps = max(100, int(round(default_steps * params['duration_mult'])))
    print(f"\nTime steps: {n_steps}  "
          f"({params['duration_mult']:.1f}x default of {default_steps})")

    # ---- Run FDTD ----
    print("\nRunning FDTD simulation...")
    snapshots, times, dt, T_arrival = run_fdtd(
        pec_mask, src_pix,
        n_steps=n_steps, snap_every=6,
        material=material,
        track_arrival=use_em_solver,
        source_mode=source_mode)
    print(f"Done. dt = {dt:.4f}, captured {len(snapshots)} snapshots.")

    # ---- EM path tracing ----
    em_path = None
    if use_em_solver:
        print("\nTracing path via gradient descent on arrival-time field...")
        em_path = trace_em_path(T_arrival, src_pix, goal_pix, pec_mask,
                                connectivity=8, smooth=True, verbose=True)
        if em_path is None:
            print("  WAVE NEVER REACHED THE GOAL.")
            print("  Try: a less lossy wall material, a longer duration,")
            print("       or a smaller maze size.")
        else:
            em_len_pix = pixel_path_length(em_path)
            em_cells = pixel_path_to_maze_cells(em_path, meta)
            print(f"  EM path: {len(em_path)} pixels, "
                  f"Euclidean length {em_len_pix:.1f} px "
                  f"(~{em_len_pix / meta['pix_per_cell']:.1f} maze cells)")
            print(f"  Unique maze cells visited: {len(em_cells)}")
            if solution_path is not None:
                print(f"  BFS path:                  {len(solution_path)} cells")

    # ---- Animate ----
    slowdown = params['slowdown']
    export_format = params.get('export_format')
    print(f"\nLaunching animation (close the window to exit).")
    print(f"Playback slowdown: {slowdown:.1f}x  "
          f"({1.0 / slowdown:.0%} of baseline rate).")
    if export_format:
        print(f"Export format: {export_format.upper()}")

    # Build a descriptive filename prefix from the current parameters
    export_basename = (
        f"maze_s{params['size']}_b{int(params['braid']*100):02d}"
        f"_{params['material']['name']}_{params['source_mode']}"
    )

    visualize(snapshots, times, pec_mask, src_pix, goal_pix,
              solution_path=solution_path, em_path=em_path, meta=meta,
              slowdown=slowdown, material=material, source_mode=source_mode,
              export_format=export_format, export_basename=export_basename)

    # ---- Static isochrone map ----
    if use_em_solver and T_arrival is not None:
        print("Showing arrival-time isochrone map...")
        plot_arrival_field(T_arrival, pec_mask, src_pix, goal_pix,
                           em_path=em_path, solution_path=solution_path,
                           meta=meta)


# ============================================================
# Driver loop (UI -> simulation -> restart prompt -> repeat)
# ============================================================
def main():
    params = get_setup()
    if params is None:
        print("Setup cancelled. Goodbye!")
        return

    while True:
        run_one_simulation(params)

        action, next_params = show_restart_dialog(params)
        if action == 'quit':
            print("\nGoodbye!")
            return
        params = next_params  # 'rerun' returns same dict; 'new' returns new


if __name__ == "__main__":
    main()
