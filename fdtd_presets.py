"""
fdtd_presets.py
===============
Save and load FDTD simulation setups as JSON presets.

Presets are stored as files in a ./presets/ folder relative to wherever
the script is run from.  Each file is a single JSON object containing
the parameter dict produced by the setup UI, plus minimal metadata.

Public API
----------
list_presets()                  -> list of preset names (no extension)
save_preset(name, params)       -> writes presets/<name>.json
load_preset(name)               -> returns params dict, or raises KeyError
delete_preset(name)             -> removes presets/<name>.json
preset_exists(name)             -> bool
ensure_presets_dir()            -> creates presets/ if missing
PRESETS_DIR                     -> the directory path (Path object)

The on-disk format:
{
    "_format_version": 1,
    "_saved_at": "2026-04-15T19:42:00",
    "params": { ...the parameter dict... }
}
"""

import json
import re
from datetime import datetime
from pathlib import Path


PRESETS_DIR = Path('presets')
FORMAT_VERSION = 1

# Required keys a valid preset must contain after loading.
REQUIRED_KEYS = {
    'size', 'braid', 'show_bfs', 'material', 'source_mode',
    'use_em_solver', 'duration_mult', 'slowdown',
}

# Filename safety: only letters, digits, hyphen, underscore, period,
# space, and a few common punctuation marks for descriptive names.
_VALID_NAME_RE = re.compile(r'^[A-Za-z0-9._\- ()]+$')


# ============================================================
# File-system helpers
# ============================================================
def ensure_presets_dir():
    """Create the presets directory if it doesn't exist."""
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)


def _validate_name(name):
    """Raise ValueError if `name` isn't safe to use as a filename."""
    name = name.strip()
    if not name:
        raise ValueError("Preset name cannot be empty.")
    if len(name) > 80:
        raise ValueError("Preset name too long (max 80 characters).")
    if not _VALID_NAME_RE.match(name):
        raise ValueError(
            "Preset name may only contain letters, digits, spaces, "
            "hyphens, underscores, periods, and parentheses.")
    return name


def _path_for(name):
    return PRESETS_DIR / f"{_validate_name(name)}.json"


# ============================================================
# Public API
# ============================================================
def list_presets():
    """Return sorted list of preset names (filenames without .json)."""
    if not PRESETS_DIR.exists():
        return []
    return sorted(p.stem for p in PRESETS_DIR.glob('*.json'))


def preset_exists(name):
    """Return True if a preset file by this name exists."""
    try:
        return _path_for(name).exists()
    except ValueError:
        return False


def save_preset(name, params):
    """
    Save `params` to presets/<name>.json.  Overwrites if it exists.

    The internal label keys (those starting with '_') are preserved so
    the UI can pre-fill dropdowns correctly when the preset is loaded.
    """
    ensure_presets_dir()
    blob = {
        '_format_version': FORMAT_VERSION,
        '_saved_at': datetime.now().isoformat(timespec='seconds'),
        'params': params,
    }
    path = _path_for(name)
    with open(path, 'w') as f:
        json.dump(blob, f, indent=2)
    return path


def load_preset(name):
    """
    Load and return the params dict for the named preset.

    Raises
    ------
    KeyError    : preset does not exist
    ValueError  : preset file is malformed or missing required keys
    """
    path = _path_for(name)
    if not path.exists():
        raise KeyError(f"No preset named {name!r}")
    with open(path) as f:
        blob = json.load(f)
    if not isinstance(blob, dict) or 'params' not in blob:
        raise ValueError(f"Preset {name!r} is malformed (missing 'params').")
    params = blob['params']
    missing = REQUIRED_KEYS - set(params)
    if missing:
        raise ValueError(
            f"Preset {name!r} is missing required keys: "
            f"{sorted(missing)}")
    return params


def delete_preset(name):
    """Remove the named preset file. Returns True if a file was deleted."""
    path = _path_for(name)
    if path.exists():
        path.unlink()
        return True
    return False


def preset_metadata(name):
    """
    Return (saved_at_iso_str, brief_summary) for display in the UI.
    Returns (None, '') if the file is unreadable.
    """
    path = _path_for(name)
    try:
        with open(path) as f:
            blob = json.load(f)
        params = blob.get('params', {})
        saved_at = blob.get('_saved_at', '')
        summary = (
            f"size={params.get('size', '?')}, "
            f"braid={params.get('braid', 0):.2f}, "
            f"{params.get('material', {}).get('name', '?')}, "
            f"{params.get('source_mode', '?')}"
        )
        return saved_at, summary
    except Exception:
        return None, ''


# ============================================================
# Standalone test
# ============================================================
if __name__ == '__main__':
    # Quick round-trip test
    test_params = {
        'size': 8,
        'braid': 0.3,
        'show_bfs': True,
        'material': {'name': 'concrete', 'eps_r': 6.0, 'loss': 0.08,
                     'desc': 'Building wall'},
        'source_mode': 'burst',
        'use_em_solver': True,
        'duration_mult': 2.0,
        'slowdown': 2.0,
        '_braid_label':    'Moderately braided (~40%, several)',
        '_material_key':   'concrete',
        '_duration_label': 'Long     (2.0x default)',
        '_slowdown_label': 'Slow (2.0x)',
    }
    print('Saving test preset...')
    p = save_preset('test_demo', test_params)
    print(f'  Saved to {p}')
    print(f'\nAll presets: {list_presets()}')
    print(f'\nLoading back...')
    loaded = load_preset('test_demo')
    print(f"  size = {loaded['size']}, material = {loaded['material']['name']}")
    print(f'\nMetadata: {preset_metadata("test_demo")}')
    print(f'\nDeleting...')
    delete_preset('test_demo')
    print(f'After delete: {list_presets()}')
