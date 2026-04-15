"""
fdtd_engine.py
==============
Pure-NumPy 2D FDTD (Yee grid) electromagnetic engine for use with the
Maze class defined in maze.py.

Implements:
  - Yee staggered grid for TM_z mode (Ez, Hx, Hy)
  - Leapfrog time-stepping at the 2D Courant limit Sc = 1/sqrt(2)
  - Mur 1st-order absorbing boundary condition on the outer domain
  - PEC (perfect electric conductor) walls (Ez forced to 0 inside walls)
  - Gaussian-modulated sinusoidal soft source

Maxwell's curl equations in 2D TM_z (normalized: c = eps0 = mu0 = 1):
    dEz/dt =  dHy/dx - dHx/dy
    dHx/dt = -dEz/dy
    dHy/dt =  dEz/dx
"""

import numpy as np


# ============================================================
# Geometry: convert a Maze object into a PEC mask on the FDTD grid
# ============================================================
def build_pec_mask(maze, pix_per_cell=8, pad=10):
    """
    Convert a Maze object into a fine-grained boolean PEC mask
    on the FDTD pixel grid.

    Each 'cell' of the maze.maze array (which is True for floor,
    False for wall) is upsampled into a pix_per_cell x pix_per_cell
    block on the FDTD grid, padded by `pad` cells of free space on
    every side for the absorbing boundary.

    Parameters
    ----------
    maze : Maze
        The Maze object (uses maze.maze, where True = floor).
    pix_per_cell : int
        FDTD grid cells per maze cell (also corridor / wall width).
    pad : int
        Free-space padding cells around the maze for the ABC.

    Returns
    -------
    pec_mask : (Nx, Ny) bool ndarray
        True where the cell is inside a PEC wall.
    meta : dict
        {'Nx', 'Ny', 'pix_per_cell', 'pad', 'maze_shape'}
    """
    wall = ~maze.maze                      # True for wall in maze coords
    inner = np.kron(wall, np.ones((pix_per_cell, pix_per_cell),
                                  dtype=bool))
    Nx = inner.shape[0] + 2 * pad
    Ny = inner.shape[1] + 2 * pad
    pec_mask = np.zeros((Nx, Ny), dtype=bool)
    pec_mask[pad:pad + inner.shape[0],
             pad:pad + inner.shape[1]] = inner

    meta = dict(Nx=Nx, Ny=Ny, pix_per_cell=pix_per_cell,
                pad=pad, maze_shape=maze.shape)
    return pec_mask, meta


def maze_cell_to_pixel(rc, meta):
    """Map a maze (row, col) display-coord to the FDTD pixel center."""
    r, c = rc
    p, pad = meta['pix_per_cell'], meta['pad']
    return (pad + r * p + p // 2, pad + c * p + p // 2)


# ============================================================
# Main FDTD time-loop
# ============================================================
def run_fdtd(pec_mask, src_pix, n_steps=900, snap_every=6,
             pix_per_wavelength=6, material=None,
             track_arrival=False, arrival_threshold=0.01,
             source_mode='burst', verbose=True):
    """
    Run the 2D TM_z FDTD simulation.

    Parameters
    ----------
    pec_mask : (Nx, Ny) bool ndarray
        True where the cell is inside a "wall" of the chosen material.
    src_pix : (i, j) tuple of ints
        Pixel location of the soft Gaussian-modulated sinusoidal source.
    n_steps : int
        Number of FDTD time steps.
    snap_every : int
        Save an Ez snapshot every `snap_every` steps.
    pix_per_wavelength : int
        Spatial sampling of the source wavelength (>=10 is typical;
        kept low here so the wave fits in narrow corridors).
    material : None or dict
        Wall material specification (see lossy-dielectric note above).
    track_arrival : bool
        If True, build a (Nx, Ny) field T_arrival recording the first
        time |Ez| at each cell exceeds `arrival_threshold`.  Cells the
        wave never reaches stay at np.inf.  Used for EM-based maze
        solving via gradient descent on T_arrival.
    arrival_threshold : float
        |Ez| level that counts as "wavefront has arrived" at a cell.
        Default 0.01 = 1% of source peak amplitude.  Lower is more
        sensitive to weak signals (after cylindrical spread and
        wall losses) but more sensitive to noise.
    source_mode : 'burst' or 'continuous'
        'burst':      single Gaussian-modulated sinusoidal pulse,
                      then silence.  Best for arrival-time tracking
                      and EM-based maze solving (clean wavefront).
        'continuous': ramped-up continuous sinusoid that keeps
                      pumping energy into the simulation.  Best for
                      visualizing steady-state interference patterns,
                      standing waves in dead-end corridors, and
                      resonant modes.  Less ideal for arrival tracking
                      because reflections from continuously pumped
                      energy can trip the threshold before the true
                      wavefront actually arrives at distant cells.
    verbose : bool
        Print progress at each 25% mark.

    Returns
    -------
    snapshots : list of (Nx, Ny) ndarrays
        Captured Ez fields.
    times : list of float
        Simulation time of each snapshot (normalized units).
    dt : float
        The time step used.
    T_arrival : (Nx, Ny) ndarray or None
        First-arrival time at each cell, or None if track_arrival=False.
    """
    Nx, Ny = pec_mask.shape

    # -------- Yee field arrays (note staggered shapes) --------
    Ez = np.zeros((Nx, Ny))
    Hx = np.zeros((Nx, Ny - 1))     # half-step in y
    Hy = np.zeros((Nx - 1, Ny))     # half-step in x

    # -------- Stability: 2D Courant condition ----------------
    dx = 1.0
    c = 1.0
    Sc = 1.0 / np.sqrt(2.0)         # the "magic" 2D timestep
    dt = Sc * dx / c
    Ce = dt / dx                    # Ez update coefficient
    Ch = dt / dx                    # H update coefficient

    # -------- Material model: build Ca, Cb arrays ------------
    # PEC: keep the original fast path (Ez forced to 0 inside walls).
    # Lossy dielectric: Ca, Cb arrays vary by cell.
    is_pec = material is None or material.get('name') == 'pec'
    if not is_pec:
        eps_r = float(material['eps_r'])
        loss  = float(material['loss'])
        if eps_r < 1.0:
            raise ValueError("eps_r must be >= 1.0")
        if loss < 0.0:
            raise ValueError("loss must be >= 0.0")
        Ca_wall = (1.0 - loss / 2.0) / (1.0 + loss / 2.0)
        Cb_wall = (Ce / eps_r)       / (1.0 + loss / 2.0)
        Ca = np.ones((Nx, Ny))
        Cb = np.full((Nx, Ny), Ce)
        Ca[pec_mask] = Ca_wall
        Cb[pec_mask] = Cb_wall
        if verbose:
            print(f"  Material '{material['name']}': "
                  f"eps_r={eps_r:.2f}, loss={loss:.4f}  "
                  f"-> Ca_wall={Ca_wall:+.3f}, Cb_wall={Cb_wall:.3f}")

    # -------- Mur 1st-order ABC ------------------------------
    mur_k = (c * dt - dx) / (c * dt + dx)
    prev_xlo = np.zeros(Ny); prev_xlo_1 = np.zeros(Ny)
    prev_xhi = np.zeros(Ny); prev_xhi_1 = np.zeros(Ny)
    prev_ylo = np.zeros(Nx); prev_ylo_1 = np.zeros(Nx)
    prev_yhi = np.zeros(Nx); prev_yhi_1 = np.zeros(Nx)

    # -------- Source: Gaussian-modulated sinusoid ------------
    wavelength = float(pix_per_wavelength)
    freq = c / wavelength
    omega = 2.0 * np.pi * freq
    period = 1.0 / freq
    t0 = 3.0 * period               # pulse center
    spread = 1.0 * period           # Gaussian envelope width
    src_i, src_j = src_pix

    # Choose the source waveform based on source_mode:
    #   burst:      one Gaussian-modulated cycle, then quiet
    #   continuous: smoothly ramp up and then sustain a sinusoid forever
    #               (the ramp is a 1-cosine soft start over a few periods,
    #               which avoids the spectral splatter of a hard turn-on).
    if source_mode == 'burst':
        def src_waveform(t):
            return np.exp(-((t - t0) / spread) ** 2) * np.sin(omega * t)
    elif source_mode == 'continuous':
        ramp_time = 4.0 * period   # smooth turn-on duration

        def src_waveform(t):
            if t < ramp_time:
                envelope = 0.5 * (1.0 - np.cos(np.pi * t / ramp_time))
            else:
                envelope = 1.0
            return envelope * np.sin(omega * t)
    else:
        raise ValueError(
            f"source_mode must be 'burst' or 'continuous', got {source_mode!r}")

    snapshots, times = [], []
    quarter_marks = {n_steps // 4, n_steps // 2, 3 * n_steps // 4}

    # -------- Arrival-time tracking --------------------------
    if track_arrival:
        T_arrival = np.full((Nx, Ny), np.inf)
    else:
        T_arrival = None

    # =========================================================
    # MAIN TIME LOOP (leapfrog: H first, then E)
    # =========================================================
    for n in range(n_steps):
        t = n * dt

        # ---- Update H from curl of E ------------------------
        # Hx at (i, j+1/2):  dHx/dt = -dEz/dy
        Hx -= Ch * (Ez[:, 1:] - Ez[:, :-1])
        # Hy at (i+1/2, j):  dHy/dt = +dEz/dx
        Hy += Ch * (Ez[1:, :] - Ez[:-1, :])

        # ---- Cache pre-update Ez slices for Mur ABC --------
        prev_xlo[:] = Ez[0, :];   prev_xlo_1[:] = Ez[1, :]
        prev_xhi[:] = Ez[-1, :];  prev_xhi_1[:] = Ez[-2, :]
        prev_ylo[:] = Ez[:, 0];   prev_ylo_1[:] = Ez[:, 1]
        prev_yhi[:] = Ez[:, -1];  prev_yhi_1[:] = Ez[:, -2]

        # ---- Update Ez interior from curl of H -------------
        if is_pec:
            # Free space everywhere (walls are zeroed below)
            Ez[1:-1, 1:-1] += Ce * (
                (Hy[1:,   1:-1] - Hy[:-1,  1:-1])
              - (Hx[1:-1, 1:  ] - Hx[1:-1, :-1 ])
            )
        else:
            # Lossy dielectric: Ca, Cb vary spatially
            curl = ((Hy[1:,   1:-1] - Hy[:-1,  1:-1])
                  - (Hx[1:-1, 1:  ] - Hx[1:-1, :-1 ]))
            Ez[1:-1, 1:-1] = (Ca[1:-1, 1:-1] * Ez[1:-1, 1:-1]
                            + Cb[1:-1, 1:-1] * curl)

        # ---- Mur ABC on the four outer edges ---------------
        Ez[0,  1:-1] = prev_xlo_1[1:-1] + mur_k * (Ez[1,  1:-1] - prev_xlo[1:-1])
        Ez[-1, 1:-1] = prev_xhi_1[1:-1] + mur_k * (Ez[-2, 1:-1] - prev_xhi[1:-1])
        Ez[1:-1,  0] = prev_ylo_1[1:-1] + mur_k * (Ez[1:-1,  1] - prev_ylo[1:-1])
        Ez[1:-1, -1] = prev_yhi_1[1:-1] + mur_k * (Ez[1:-1, -2] - prev_yhi[1:-1])

        # ---- PEC: force Ez = 0 inside metal walls ----------
        if is_pec:
            Ez[pec_mask] = 0.0
        # (lossy materials are handled by Ca, Cb above; no zeroing needed)

        # ---- Soft source injection -------------------------
        Ez[src_i, src_j] += src_waveform(t)

        # ---- Record first-arrival time at each cell --------
        if track_arrival:
            detected = (np.abs(Ez) > arrival_threshold) & np.isinf(T_arrival)
            T_arrival[detected] = t

        # ---- Capture snapshot ------------------------------
        if n % snap_every == 0:
            snapshots.append(Ez.copy())
            times.append(t)

        if verbose and n in quarter_marks:
            pct = 100 * (n + 1) // n_steps
            print(f"  ... {pct}% complete (step {n+1}/{n_steps})")

    return snapshots, times, dt, T_arrival