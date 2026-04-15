"""
fdtd_setup_ui.py
================
Tkinter setup window for the FDTD maze solver.

Replaces the sequential terminal prompts with a single dialog that
collects every parameter at once, then lets the user re-run more
simulations without restarting the script.

Public API
----------
get_setup() -> dict | None
    Show the setup window.  Returns a dict of parameters when the user
    clicks "Run Simulation", or None if they close the window.

show_restart_dialog(prev_params) -> ('rerun' | 'new' | 'quit', dict | None)
    Shown after a simulation completes.  Returns a 2-tuple:
      ('rerun', prev_params)      run the same setup again
      ('new',   new_params)       run with a freshly chosen setup
      ('quit',  None)             exit the program

The dict returned has keys:
    size           int
    braid          float in [0, 1]
    show_bfs       bool
    material       dict (one of MATERIAL_PRESETS values)
    source_mode    'burst' or 'continuous'
    use_em_solver  bool
    duration_mult  float (multiplier on the auto-scaled default n_steps)
    slowdown       float (animation playback slowdown factor)
"""

import tkinter as tk
from tkinter import ttk, messagebox


# ============================================================
# Material catalog (mirrors the one in fdtd_main.py for self-containment)
# ============================================================
MATERIAL_PRESETS = {
    'pec': {
        'name': 'pec', 'eps_r': 1.0, 'loss': None,
        'desc': 'Perfect electric conductor - total reflection'
    },
    'metal': {
        'name': 'metal', 'eps_r': 1.0, 'loss': 1.5,
        'desc': 'Lossy conductor (e.g. aluminum)'
    },
    'concrete': {
        'name': 'concrete', 'eps_r': 6.0, 'loss': 0.08,
        'desc': 'Building wall - partial transmission'
    },
    'wood': {
        'name': 'wood', 'eps_r': 3.0, 'loss': 0.02,
        'desc': 'Wooden walls - significant transmission'
    },
    'glass': {
        'name': 'glass', 'eps_r': 4.5, 'loss': 0.002,
        'desc': 'Low-loss dielectric - refraction'
    },
    'absorber': {
        'name': 'absorber', 'eps_r': 2.0, 'loss': 0.4,
        'desc': 'Radar-absorbing material'
    },
}

BRAID_PRESETS = {
    'Regular maze (single path)':            0.0,
    'Lightly braided (~20%, a few loops)':   0.2,
    'Moderately braided (~40%, several)':    0.4,
    'Heavily braided (~70%, many loops)':    0.7,
}

DURATION_PRESETS = {
    'Short    (0.5x default)':     0.5,
    'Default  (1.0x)':             1.0,
    'Long     (2.0x default)':     2.0,
    'Very long (4.0x default)':    4.0,
}

SLOWDOWN_PRESETS = {
    'Fast (1.0x)':         1.0,
    'Normal (1.5x)':       1.5,
    'Slow (2.0x)':         2.0,    # the original default
    'Very slow (3.0x)':    3.0,
    'Crawl (5.0x)':        5.0,
}


# ============================================================
# Setup window
# ============================================================
class FDTDSetupUI:
    """Tkinter window that gathers all simulation parameters."""

    def __init__(self, defaults=None, title='FDTD Maze Solver - Setup'):
        self.result = None  # populated on "Run", left None on close
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry('580x720')

        # Pre-fill from defaults dict (used for the restart loop)
        d = defaults or {}

        # Tkinter Var holders for each input
        self.var_size       = tk.IntVar(value=d.get('size', 8))
        self.var_braid      = tk.StringVar(value=d.get(
            '_braid_label', 'Regular maze (single path)'))
        self.var_show_bfs   = tk.BooleanVar(value=d.get('show_bfs', True))
        self.var_material   = tk.StringVar(value=d.get(
            '_material_key', 'pec'))
        self.var_source     = tk.StringVar(value=d.get('source_mode', 'burst'))
        self.var_use_em     = tk.BooleanVar(value=d.get('use_em_solver', True))
        self.var_duration   = tk.StringVar(value=d.get(
            '_duration_label', 'Default  (1.0x)'))
        self.var_slowdown   = tk.StringVar(value=d.get(
            '_slowdown_label', 'Slow (2.0x)'))

        self._build_widgets()
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)

    # --------------------------------------------------------
    def _build_widgets(self):
        """Lay out all the input controls on the window."""
        pad = {'padx': 10, 'pady': 4}

        # Title
        title = ttk.Label(
            self.root, text='2D FDTD Electromagnetic Maze Solver',
            font=('TkDefaultFont', 13, 'bold'))
        title.pack(pady=(12, 2))
        subtitle = ttk.Label(
            self.root, text='Choose simulation parameters, then click Run.',
            foreground='#555')
        subtitle.pack(pady=(0, 10))

        # ---- Maze section -----------------------------------
        maze_frame = ttk.LabelFrame(self.root, text='  Maze  ')
        maze_frame.pack(fill='x', **pad)

        row = ttk.Frame(maze_frame); row.pack(fill='x', padx=8, pady=6)
        ttk.Label(row, text='Maze size:').pack(side='left')
        size_spin = ttk.Spinbox(row, from_=3, to=30, width=6,
                                textvariable=self.var_size)
        size_spin.pack(side='left', padx=(8, 0))
        ttk.Label(row, text='cells per side  (recommended 5-15)',
                  foreground='#666').pack(side='left', padx=(8, 0))

        row = ttk.Frame(maze_frame); row.pack(fill='x', padx=8, pady=6)
        ttk.Label(row, text='Maze type:').pack(side='left')
        braid_combo = ttk.Combobox(
            row, textvariable=self.var_braid,
            values=list(BRAID_PRESETS.keys()),
            state='readonly', width=42)
        braid_combo.pack(side='left', padx=(8, 0))

        ttk.Checkbutton(maze_frame,
                        text='Overlay BFS shortest path on animation',
                        variable=self.var_show_bfs
                        ).pack(anchor='w', padx=8, pady=(0, 6))

        # ---- Physics section --------------------------------
        phys_frame = ttk.LabelFrame(self.root, text='  Physics  ')
        phys_frame.pack(fill='x', **pad)

        row = ttk.Frame(phys_frame); row.pack(fill='x', padx=8, pady=6)
        ttk.Label(row, text='Wall material:').pack(side='left')
        mat_combo = ttk.Combobox(
            row, textvariable=self.var_material,
            values=list(MATERIAL_PRESETS.keys()),
            state='readonly', width=14)
        mat_combo.pack(side='left', padx=(8, 0))
        self.mat_desc_label = ttk.Label(phys_frame, text='', foreground='#666')
        self.mat_desc_label.pack(anchor='w', padx=8)
        mat_combo.bind('<<ComboboxSelected>>', self._update_mat_desc)
        self._update_mat_desc()

        ttk.Label(phys_frame, text='Source waveform:').pack(
            anchor='w', padx=8, pady=(8, 0))
        ttk.Radiobutton(
            phys_frame,
            text='Single burst  (clean wavefront, best for solving)',
            variable=self.var_source, value='burst'
        ).pack(anchor='w', padx=24)
        ttk.Radiobutton(
            phys_frame,
            text='Continuous   (standing waves, best for visualizing)',
            variable=self.var_source, value='continuous'
        ).pack(anchor='w', padx=24, pady=(0, 6))

        # ---- Solver section ---------------------------------
        solv_frame = ttk.LabelFrame(self.root, text='  Solver  ')
        solv_frame.pack(fill='x', **pad)
        ttk.Checkbutton(
            solv_frame,
            text='Use the EM wave to solve the maze '
                 '(arrival-time tracking + path tracing)',
            variable=self.var_use_em
        ).pack(anchor='w', padx=8, pady=6)

        # ---- Timing section ---------------------------------
        time_frame = ttk.LabelFrame(self.root, text='  Timing  ')
        time_frame.pack(fill='x', **pad)

        row = ttk.Frame(time_frame); row.pack(fill='x', padx=8, pady=6)
        ttk.Label(row, text='Simulation duration:').pack(side='left')
        ttk.Combobox(row, textvariable=self.var_duration,
                     values=list(DURATION_PRESETS.keys()),
                     state='readonly', width=24
                     ).pack(side='left', padx=(8, 0))

        row = ttk.Frame(time_frame); row.pack(fill='x', padx=8, pady=6)
        ttk.Label(row, text='Animation playback:').pack(side='left')
        ttk.Combobox(row, textvariable=self.var_slowdown,
                     values=list(SLOWDOWN_PRESETS.keys()),
                     state='readonly', width=24
                     ).pack(side='left', padx=(8, 0))

        # ---- Buttons ----------------------------------------
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(side='bottom', pady=14)

        ttk.Button(btn_frame, text='Run Simulation', width=18,
                   command=self._on_run).pack(side='left', padx=8)
        ttk.Button(btn_frame, text='Cancel', width=10,
                   command=self._on_close).pack(side='left', padx=8)

    # --------------------------------------------------------
    def _update_mat_desc(self, _event=None):
        key = self.var_material.get()
        m = MATERIAL_PRESETS.get(key)
        if m:
            loss_str = ('infinite (PEC)' if m['loss'] is None
                        else f"{m['loss']:.3f}")
            self.mat_desc_label.config(
                text=f"  -> eps_r={m['eps_r']:.2f}, loss={loss_str}: "
                     f"{m['desc']}")

    # --------------------------------------------------------
    def _on_run(self):
        """Validate inputs and pack them into self.result, then close."""
        try:
            size = int(self.var_size.get())
        except (tk.TclError, ValueError):
            messagebox.showerror('Invalid input',
                                 'Maze size must be an integer.')
            return
        if size < 3:
            messagebox.showerror('Invalid input', 'Maze size must be >= 3.')
            return
        if size > 30:
            if not messagebox.askyesno(
                    'Large maze',
                    f'Size {size} may be slow.  Continue?'):
                return

        braid_label = self.var_braid.get()
        duration_label = self.var_duration.get()
        slowdown_label = self.var_slowdown.get()
        material_key = self.var_material.get()

        self.result = {
            'size':           size,
            'braid':          BRAID_PRESETS[braid_label],
            'show_bfs':       bool(self.var_show_bfs.get()),
            'material':       MATERIAL_PRESETS[material_key],
            'source_mode':    self.var_source.get(),
            'use_em_solver':  bool(self.var_use_em.get()),
            'duration_mult':  DURATION_PRESETS[duration_label],
            'slowdown':       SLOWDOWN_PRESETS[slowdown_label],
            # Internal labels so the restart dialog can pre-select correctly
            '_braid_label':    braid_label,
            '_material_key':   material_key,
            '_duration_label': duration_label,
            '_slowdown_label': slowdown_label,
        }
        self.root.destroy()

    def _on_close(self):
        self.result = None
        self.root.destroy()

    # --------------------------------------------------------
    def show(self):
        """Display the window and block until the user closes it."""
        self.root.mainloop()
        return self.result


# ============================================================
# Restart dialog (after a simulation completes)
# ============================================================
class RestartDialog:
    """Three-way prompt: run again with same setup, change setup, or quit."""

    def __init__(self, prev_params):
        self.choice = 'quit'
        self.prev_params = prev_params

        self.root = tk.Tk()
        self.root.title('Simulation complete')
        self.root.geometry('420x260')

        ttk.Label(
            self.root, text='Simulation complete.',
            font=('TkDefaultFont', 12, 'bold')
        ).pack(pady=(18, 8))

        # Brief summary of what was just run
        summary = (
            f"  Maze size:     {prev_params['size']}\n"
            f"  Braid:         {prev_params['_braid_label']}\n"
            f"  Material:      {prev_params['_material_key']}\n"
            f"  Source:        {prev_params['source_mode']}\n"
            f"  Duration:      {prev_params['_duration_label']}\n"
            f"  EM solver:     {'on' if prev_params['use_em_solver'] else 'off'}"
        )
        ttk.Label(self.root, text=summary, font=('TkFixedFont', 9),
                  foreground='#444', justify='left'
                  ).pack(padx=20, pady=(0, 12))

        ttk.Label(self.root, text='What now?').pack()

        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text='Run Same Again', width=18,
                   command=self._rerun).pack(side='left', padx=4)
        ttk.Button(btn_frame, text='Change Setup', width=18,
                   command=self._new).pack(side='left', padx=4)
        ttk.Button(btn_frame, text='Quit', width=10,
                   command=self._quit).pack(side='left', padx=4)

        self.root.protocol('WM_DELETE_WINDOW', self._quit)

    def _rerun(self):
        self.choice = 'rerun'
        self.root.destroy()

    def _new(self):
        self.choice = 'new'
        self.root.destroy()

    def _quit(self):
        self.choice = 'quit'
        self.root.destroy()

    def show(self):
        self.root.mainloop()
        return self.choice


# ============================================================
# Public API
# ============================================================
def get_setup(defaults=None):
    """
    Show the setup window.  Returns the parameter dict on Run, or None
    if the user closed/cancelled.
    """
    return FDTDSetupUI(defaults=defaults).show()


def show_restart_dialog(prev_params):
    """
    After a simulation completes, ask the user what to do next.

    Returns
    -------
    (action, params) tuple where:
        action == 'rerun'  -> params is the previous params dict
        action == 'new'    -> params is a freshly chosen dict (or None
                              if the user cancelled the new-setup window)
        action == 'quit'   -> params is None
    """
    choice = RestartDialog(prev_params).show()
    if choice == 'rerun':
        return 'rerun', prev_params
    if choice == 'new':
        new_params = get_setup(defaults=prev_params)
        if new_params is None:
            return 'quit', None
        return 'new', new_params
    return 'quit', None


# ============================================================
# Standalone test
# ============================================================
if __name__ == '__main__':
    params = get_setup()
    if params is None:
        print('User cancelled.')
    else:
        print('Got setup:')
        for k, v in params.items():
            if not k.startswith('_'):
                print(f'  {k} = {v}')
