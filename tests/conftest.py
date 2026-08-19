"""Test configuration for hermes-lcm plugin tests.

Patches the plugin modules so they can be imported both as a package
(relative imports during plugin loading) and directly during testing.
"""
import sys
import importlib
import os
import tempfile
from pathlib import Path

# --- Live-database isolation guard -------------------------------------
# Some tests construct LCM config without monkeypatching HERMES_HOME, so the
# plugin default (config.py: Path(HERMES_HOME or ~/.hermes)/lcm.db) resolves
# to the REAL production database. A full-suite pytest run then holds POSIX
# locks on the live lcm.db and blocks gateway ingest for the run's duration
# (observed 2026-08-18 and 2026-08-19: PIDs 2987553/3306236, inode 553579).
#
# Guard: if HERMES_HOME points at (or inside) the invoking user's real home,
# redirect it to a per-run temp dir BEFORE any plugin module imports config.
# Tests that explicitly monkeypatch HERMES_HOME to a tmp_path are unaffected
# (their value is outside the real home). Set LCM_TESTS_ALLOW_REAL_HOME=1 to
# opt out deliberately (e.g. a diagnostic run that must inspect live state).
def _isolate_hermes_home_if_live() -> None:
    if os.environ.get("LCM_TESTS_ALLOW_REAL_HOME") == "1":
        return
    home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
    try:
        real_home = Path.home().resolve()
        live = home.resolve()
    except OSError:
        return
    if live == real_home or real_home in live.parents or live in (real_home / ".hermes",):
        isolated = Path(tempfile.mkdtemp(prefix="lcm-tests-hermes-home-"))
        os.environ["HERMES_HOME"] = str(isolated)
        print(
            f"conftest: HERMES_HOME pointed at the real home; "
            f"isolated to {isolated} (set LCM_TESTS_ALLOW_REAL_HOME=1 to override)",
            file=sys.stderr,
        )


_isolate_hermes_home_if_live()

# Make the repo root importable (for agent.context_engine etc.)
repo_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Register the plugin directory as a proper package
plugin_dir = Path(__file__).resolve().parent.parent
pkg_name = "hermes_lcm"

if pkg_name not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        pkg_name,
        str(plugin_dir / "__init__.py"),
        submodule_search_locations=[str(plugin_dir)],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__path__ = [str(plugin_dir)]
    mod.__package__ = pkg_name
    sys.modules[pkg_name] = mod
    # Don't exec the module (it tries to register with ctx)
    # Just make submodules importable

    # Register each submodule
    for py_file in plugin_dir.glob("*.py"):
        if py_file.name == "__init__.py":
            continue
        sub_name = f"{pkg_name}.{py_file.stem}"
        if sub_name not in sys.modules:
            sub_spec = importlib.util.spec_from_file_location(
                sub_name, str(py_file),
                submodule_search_locations=[],
            )
            sub_mod = importlib.util.module_from_spec(sub_spec)
            sub_mod.__package__ = pkg_name
            sys.modules[sub_name] = sub_mod
            setattr(mod, py_file.stem, sub_mod)
            try:
                sub_spec.loader.exec_module(sub_mod)
            except Exception:
                pass  # some modules may fail (e.g. engine needs agent)
