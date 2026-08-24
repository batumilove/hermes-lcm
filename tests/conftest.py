"""Test configuration for hermes-lcm plugin tests.

Patches the plugin modules so they can be imported both as a package
(relative imports during plugin loading) and directly during testing.
"""
import atexit
import importlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Collection-time safety boundary: tests that instantiate LCMConfig() with its
# defaults must never inherit an operator's live Hermes state.  This must run
# before any plugin module is imported because config/path constants can bind
# during collection, before pytest fixtures execute.
INHERITED_OPERATOR_HERMES_HOME = os.environ.get("HERMES_HOME", "")
DISPOSABLE_HERMES_HOME = tempfile.mkdtemp(prefix="hermes-lcm-tests-")
os.chmod(DISPOSABLE_HERMES_HOME, 0o700)
os.environ["HERMES_HOME"] = DISPOSABLE_HERMES_HOME
# Engine fallback paths use Path.home()/.hermes when no explicit hermes_home is
# supplied, so isolate HOME as well as HERMES_HOME.
os.environ["HOME"] = DISPOSABLE_HERMES_HOME
os.environ["HERMES_TEST_INHERITED_HERMES_HOME"] = INHERITED_OPERATOR_HERMES_HOME
os.environ["HERMES_TEST_DISPOSABLE_HERMES_HOME"] = DISPOSABLE_HERMES_HOME
os.environ["HERMES_TEST_HOME_ISOLATED"] = "1"
HERMES_HOME_ISOLATED = True


def _cleanup_disposable_hermes_home() -> None:
    shutil.rmtree(DISPOSABLE_HERMES_HOME, ignore_errors=True)


atexit.register(_cleanup_disposable_hermes_home)

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
    # Execute only the package initializer. It defines register() but does not
    # invoke it, so this is side-effect safe. Let Python import submodules on
    # demand instead of eagerly creating every module and swallowing import
    # failures: a half-initialized entry left in sys.modules breaks dotted
    # monkeypatch targets such as ``hermes_lcm.engine.<name>``.
    spec.loader.exec_module(mod)
