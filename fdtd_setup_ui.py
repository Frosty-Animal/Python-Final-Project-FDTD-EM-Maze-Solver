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
from tkinter import ttk, messagebox, simpledialog

from fdtd_presets import (list_presets, save_preset, load_preset,
                           delete_preset, preset_exists, preset_metadata)


# ============================================================
# Material catalog (mirrors the one in fdtd_main.py for self-containment)
# ============================================================
MATERIAL_PRESETS = {
    # ---- Perfect conductor (baseline) ---
    'pec': {
        'name': 'pec', 'eps_r': 1.0, 'loss': None,
        'desc': 'Perfect electric conductor — total reflection, no penetration',
        'category': 'Ideal',
    },

    # ---- Metals & conductors ---
    'silver (Ag)': {
        'name': 'silver (Ag)', 'eps_r': 1.0, 'loss': 1.8,
        'desc': 'σ ≈ 6.3×10⁷ S/m — highest conductivity of any element',
        'category': 'Metal',
    },
    'copper (Cu)': {
        'name': 'copper (Cu)', 'eps_r': 1.0, 'loss': 1.7,
        'desc': 'σ ≈ 5.96×10⁷ S/m — standard PCB trace / waveguide material',
        'category': 'Metal',
    },
    'aluminum (Al)': {
        'name': 'aluminum (Al)', 'eps_r': 1.0, 'loss': 1.5,
        'desc': 'σ ≈ 3.77×10⁷ S/m — lightweight RF shielding, enclosures',
        'category': 'Metal',
    },
    'ITO': {
        'name': 'ITO', 'eps_r': 3.8, 'loss': 0.5,
        'desc': 'Indium tin oxide — transparent conductor, εᵣ ≈ 3.8, σ ≈ 10⁴ S/m',
        'category': 'Metal',
    },

    # ---- Semiconductors ---
    'a-Si': {
        'name': 'a-Si', 'eps_r': 11.7, 'loss': 0.15,
        'desc': 'Amorphous silicon — solar cells, TFTs, εᵣ ≈ 11.7',
        'category': 'Semiconductor',
    },
    'c-Si': {
        'name': 'c-Si', 'eps_r': 11.7, 'loss': 0.02,
        'desc': 'Crystalline silicon (intrinsic) — εᵣ ≈ 11.7, very low loss',
        'category': 'Semiconductor',
    },
    'GaAs': {
        'name': 'GaAs', 'eps_r': 12.9, 'loss': 0.006,
        'desc': 'Gallium arsenide — MMIC substrates, εᵣ ≈ 12.9, tan δ ≈ 0.0004',
        'category': 'Semiconductor',
    },

    # ---- Dielectrics & substrates ---
    'SiO2': {
        'name': 'SiO2', 'eps_r': 3.9, 'loss': 0.003,
        'desc': 'Fused silica / quartz — IC gate oxide, εᵣ ≈ 3.9, tan δ < 0.001',
        'category': 'Dielectric',
    },
    'alumina (Al2O3)': {
        'name': 'alumina (Al2O3)', 'eps_r': 9.8, 'loss': 0.005,
        'desc': 'Aluminum oxide ceramic — LTCC, εᵣ ≈ 9.8, tan δ ≈ 0.0001',
        'category': 'Dielectric',
    },
    'FR-4': {
        'name': 'FR-4', 'eps_r': 4.4, 'loss': 0.04,
        'desc': 'Standard PCB laminate — εᵣ ≈ 4.4, tan δ ≈ 0.02',
        'category': 'Dielectric',
    },
    'Rogers RT5880': {
        'name': 'Rogers RT5880', 'eps_r': 2.2, 'loss': 0.002,
        'desc': 'High-frequency PCB — εᵣ ≈ 2.2, tan δ ≈ 0.0009',
        'category': 'Dielectric',
    },
    'PTFE (Teflon)': {
        'name': 'PTFE (Teflon)', 'eps_r': 2.1, 'loss': 0.001,
        'desc': 'Low-k low-loss — coax insulation, εᵣ ≈ 2.1, tan δ ≈ 0.0002',
        'category': 'Dielectric',
    },
    'BaTiO3': {
        'name': 'BaTiO3', 'eps_r': 150.0, 'loss': 0.08,
        'desc': 'Barium titanate — high-k ceramic capacitor, εᵣ ≈ 150–1700',
        'category': 'Dielectric',
    },

    # ---- Magnetic & specialty ---
    'ferrite (NiZn)': {
        'name': 'ferrite (NiZn)', 'eps_r': 13.0, 'loss': 0.12,
        'desc': 'Nickel-zinc ferrite — EMI absorber, chokes, εᵣ ≈ 13',
        'category': 'Magnetic',
    },
    'ferrite (MnZn)': {
        'name': 'ferrite (MnZn)', 'eps_r': 15.0, 'loss': 0.18,
        'desc': 'Manganese-zinc ferrite — power cores, εᵣ ≈ 15, higher loss',
        'category': 'Magnetic',
    },
    'radar absorber (RAM)': {
        'name': 'radar absorber (RAM)', 'eps_r': 3.0, 'loss': 0.6,
        'desc': 'Carbon-loaded foam absorber — anechoic chamber linings',
        'category': 'Magnetic',
    },

    # ---- Liquids & biological ---
    'DMSO': {
        'name': 'DMSO', 'eps_r': 47.0, 'loss': 0.1,
        'desc': 'Dimethyl sulfoxide — high-ε solvent, εᵣ ≈ 47, microwave research',
        'category': 'Liquid',
    },
    'DI water': {
        'name': 'DI water', 'eps_r': 80.0, 'loss': 0.05,
        'desc': 'Deionized water — εᵣ ≈ 80 at low freq, σ ≈ 5.5 µS/m',
        'category': 'Liquid',
    },
    'saline (0.9%)': {
        'name': 'saline (0.9%)', 'eps_r': 76.0, 'loss': 0.35,
        'desc': 'Physiological saline — εᵣ ≈ 76, σ ≈ 1.5 S/m, bio-EM reference',
        'category': 'Liquid',
    },
    'skin tissue': {
        'name': 'skin tissue', 'eps_r': 38.0, 'loss': 0.25,
        'desc': 'Human skin at 2.4 GHz — εᵣ ≈ 38, σ ≈ 1.46 S/m',
        'category': 'Liquid',
    },

    # ---- Common building / shielding ---
    'concrete': {
        'name': 'concrete', 'eps_r': 6.0, 'loss': 0.08,
        'desc': 'Reinforced concrete — indoor propagation, εᵣ ≈ 5–8',
        'category': 'Building',
    },
    'gypsum (drywall)': {
        'name': 'gypsum (drywall)', 'eps_r': 2.8, 'loss': 0.01,
        'desc': 'Interior partition wall — εᵣ ≈ 2.8, low loss',
        'category': 'Building',
    },
    'plate glass': {
        'name': 'plate glass', 'eps_r': 6.3, 'loss': 0.005,
        'desc': 'Soda-lime window glass — εᵣ ≈ 6.3, very low loss',
        'category': 'Building',
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

EXPORT_FORMAT_PRESETS = {
    "Don't save (view only)": None,
    'GIF (always works, large file)': 'gif',
    'MP4 (needs ffmpeg, smaller file)': 'mp4',
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
        self.root.geometry('640x860')

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
        self.var_export     = tk.StringVar(value=d.get(
            '_export_label', "Don't save (view only)"))

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

        # ---- Preset bar -------------------------------------
        preset_frame = ttk.LabelFrame(self.root, text='  Presets  ')
        preset_frame.pack(fill='x', **pad)

        row = ttk.Frame(preset_frame); row.pack(fill='x', padx=8, pady=6)
        ttk.Label(row, text='Saved setup:').pack(side='left')
        self.preset_combo = ttk.Combobox(
            row, values=list_presets(), state='readonly', width=28)
        self.preset_combo.pack(side='left', padx=(8, 4))
        self.preset_combo.bind('<<ComboboxSelected>>', self._on_preset_pick)
        ttk.Button(row, text='Load', width=6,
                   command=self._on_load_preset
                   ).pack(side='left', padx=2)
        ttk.Button(row, text='Save As...', width=10,
                   command=self._on_save_preset
                   ).pack(side='left', padx=2)
        ttk.Button(row, text='Delete', width=8,
                   command=self._on_delete_preset
                   ).pack(side='left', padx=2)

        self.preset_info = ttk.Label(preset_frame, text='', foreground='#666')
        self.preset_info.pack(anchor='w', padx=8, pady=(0, 4))

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
            state='readonly', width=24)
        mat_combo.pack(side='left', padx=(8, 0))
        self.mat_desc_label = ttk.Label(phys_frame, text='', foreground='#666',
                                        wraplength=580)
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

        row = ttk.Frame(time_frame); row.pack(fill='x', padx=8, pady=6)
        ttk.Label(row, text='Export animation:').pack(side='left')
        ttk.Combobox(row, textvariable=self.var_export,
                     values=list(EXPORT_FORMAT_PRESETS.keys()),
                     state='readonly', width=30
                     ).pack(side='left', padx=(8, 0))
        ttk.Label(time_frame,
                  text='   Saved to ./animations/<name>_<timestamp>.<ext>',
                  foreground='#666'
                  ).pack(anchor='w', padx=8, pady=(0, 4))

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
            cat = m.get('category', '')
            self.mat_desc_label.config(
                text=f"  [{cat}] εᵣ={m['eps_r']:.1f}, loss={loss_str}: "
                     f"{m['desc']}")

    # --------------------------------------------------------
    # Preset handling
    # --------------------------------------------------------
    def _on_preset_pick(self, _event=None):
        """Show metadata for the selected preset (without loading it yet)."""
        name = self.preset_combo.get()
        if not name:
            self.preset_info.config(text='')
            return
        saved_at, summary = preset_metadata(name)
        if saved_at:
            self.preset_info.config(
                text=f"   {summary}    (saved {saved_at})")
        else:
            self.preset_info.config(text='   (could not read preset)')

    def _on_load_preset(self):
        """Load the selected preset into all UI fields."""
        name = self.preset_combo.get()
        if not name:
            messagebox.showinfo('Load preset',
                                'Pick a preset from the dropdown first.')
            return
        try:
            params = load_preset(name)
        except (KeyError, ValueError) as e:
            messagebox.showerror('Load failed', str(e))
            return
        self._apply_params(params)
        self.preset_info.config(text=f"   Loaded preset {name!r}")

    def _on_save_preset(self):
        """Save the current UI state as a new preset."""
        # Use whatever's currently selected in the dropdown as the default name
        default = self.preset_combo.get() or ''
        name = simpledialog.askstring(
            'Save preset',
            'Name for this preset:\n'
            '(letters, digits, spaces, hyphens, underscores, parentheses)',
            initialvalue=default, parent=self.root)
        if name is None:
            return  # cancelled
        name = name.strip()
        if not name:
            messagebox.showerror('Save failed', 'Name cannot be empty.')
            return

        # Confirm overwrite
        if preset_exists(name):
            if not messagebox.askyesno(
                    'Overwrite?',
                    f"A preset called {name!r} already exists. Overwrite?"):
                return

        # Build a params dict from current UI state without closing the window
        try:
            params = self._collect_params()
        except ValueError as e:
            messagebox.showerror('Cannot save', str(e))
            return

        try:
            save_preset(name, params)
        except (ValueError, OSError) as e:
            messagebox.showerror('Save failed', str(e))
            return

        # Refresh dropdown and select the just-saved entry
        self.preset_combo['values'] = list_presets()
        self.preset_combo.set(name)
        self._on_preset_pick()
        messagebox.showinfo('Saved',
                            f"Preset {name!r} saved to ./presets/{name}.json")

    def _on_delete_preset(self):
        """Delete the selected preset after confirmation."""
        name = self.preset_combo.get()
        if not name:
            messagebox.showinfo('Delete preset',
                                'Pick a preset from the dropdown first.')
            return
        if not messagebox.askyesno(
                'Confirm delete',
                f"Delete preset {name!r}?\nThis cannot be undone."):
            return
        if delete_preset(name):
            self.preset_combo['values'] = list_presets()
            self.preset_combo.set('')
            self.preset_info.config(text=f"   Deleted preset {name!r}")
        else:
            messagebox.showerror('Delete failed',
                                 f"Could not find preset {name!r}.")

    def _collect_params(self):
        """
        Read the current UI state and produce a params dict, without
        closing the window.  Raises ValueError if any field is invalid.
        Used by both the Save Preset button and (indirectly) by _on_run.
        """
        try:
            size = int(self.var_size.get())
        except (tk.TclError, ValueError):
            raise ValueError('Maze size must be an integer.')
        if size < 3:
            raise ValueError('Maze size must be >= 3.')

        braid_label = self.var_braid.get()
        duration_label = self.var_duration.get()
        slowdown_label = self.var_slowdown.get()
        material_key = self.var_material.get()
        export_label = self.var_export.get()

        return {
            'size':            size,
            'braid':           BRAID_PRESETS[braid_label],
            'show_bfs':        bool(self.var_show_bfs.get()),
            'material':        MATERIAL_PRESETS[material_key],
            'source_mode':     self.var_source.get(),
            'use_em_solver':   bool(self.var_use_em.get()),
            'duration_mult':   DURATION_PRESETS[duration_label],
            'slowdown':        SLOWDOWN_PRESETS[slowdown_label],
            'export_format':   EXPORT_FORMAT_PRESETS[export_label],
            '_braid_label':    braid_label,
            '_material_key':   material_key,
            '_duration_label': duration_label,
            '_slowdown_label': slowdown_label,
            '_export_label':   export_label,
        }

    def _apply_params(self, params):
        """Update every UI widget to reflect the values in `params`."""
        self.var_size.set(int(params.get('size', 8)))
        self.var_show_bfs.set(bool(params.get('show_bfs', True)))
        self.var_source.set(params.get('source_mode', 'burst'))
        self.var_use_em.set(bool(params.get('use_em_solver', True)))

        # Dropdowns: prefer the saved label; fall back to value lookup
        braid_label = params.get('_braid_label')
        if not braid_label:
            braid_label = self._closest_label(BRAID_PRESETS,
                                              params.get('braid', 0.0))
        self.var_braid.set(braid_label)

        material_key = params.get('_material_key')
        if not material_key or material_key not in MATERIAL_PRESETS:
            material_key = params.get('material', {}).get('name', 'pec')
        self.var_material.set(material_key)
        self._update_mat_desc()

        duration_label = params.get('_duration_label')
        if not duration_label:
            duration_label = self._closest_label(DURATION_PRESETS,
                                                 params.get('duration_mult', 1.0))
        self.var_duration.set(duration_label)

        slowdown_label = params.get('_slowdown_label')
        if not slowdown_label:
            slowdown_label = self._closest_label(SLOWDOWN_PRESETS,
                                                 params.get('slowdown', 2.0))
        self.var_slowdown.set(slowdown_label)

        export_label = params.get('_export_label')
        if not export_label:
            # Reverse-lookup by export_format value
            fmt = params.get('export_format', None)
            for label, val in EXPORT_FORMAT_PRESETS.items():
                if val == fmt:
                    export_label = label
                    break
            else:
                export_label = "Don't save (view only)"
        self.var_export.set(export_label)

    @staticmethod
    def _closest_label(preset_dict, target):
        """Return the dict key whose value is numerically closest to `target`."""
        return min(preset_dict, key=lambda k: abs(preset_dict[k] - target))

    # --------------------------------------------------------
    def _on_run(self):
        """Validate inputs and pack them into self.result, then close."""
        try:
            params = self._collect_params()
        except ValueError as e:
            messagebox.showerror('Invalid input', str(e))
            return
        if params['size'] > 30:
            if not messagebox.askyesno(
                    'Large maze',
                    f"Size {params['size']} may be slow.  Continue?"):
                return
        self.result = params
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
