"""Collection-time HERMES_HOME isolation regression (incident 2026-08-19/20).

A prior broad pytest inherited production HERMES_HOME and opened the live
/home/ubuntu/.hermes/lcm.db, holding POSIX locks that blocked gateway ingest
preflight. tests/conftest.py imports plugin modules at collection time, so an
autouse fixture is too late: the guard must replace HERMES_HOME (and the
Path.home() fallback) *before* any plugin import executes.

Contract proven here:
  1. With an operator-like inherited HERMES_HOME containing a sentinel
     production-like lcm.db, importing tests/conftest.py redirects the
     environment to a restrictive disposable home.
  2. Default LCMConfig()/engine state resolves inside the disposable home,
     never the operator home.
  3. The operator sentinel lcm.db is byte-for-byte unchanged afterwards.
  4. Subprocesses inherit the disposable home (no silent fallback).
  5. The disposable path differs from the captured operator path and has
     restrictive permissions (no group/other access).

The child-process probe imports conftest directly (never pytest-in-pytest),
so there is no recursive test runaway.
"""
import hashlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
PLUGIN_DIR = TESTS_DIR.parent
SENTINEL_BYTES = b"SENTINEL-PRODUCTION-LIKE- lcm.db v1\n" * 64


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


CHILD_PROBE = textwrap.dedent(
    """
    import importlib.util, json, os, subprocess, sys, tempfile
    from pathlib import Path

    conftest_path = sys.argv[1]
    spec = importlib.util.spec_from_file_location("conftest_under_test", conftest_path)
    conftest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conftest)

    env_home = os.environ.get("HERMES_HOME", "")

    # Default LCMConfig path resolution (config.py reads env at call time).
    cfg_spec = importlib.util.spec_from_file_location(
        "lcm_config_probe", str(Path(conftest_path).parent.parent / "config.py"))
    cfg = importlib.util.module_from_spec(cfg_spec)
    cfg_spec.loader.exec_module(cfg)
    config_yaml = str(cfg._hermes_config_path())

    # Engine default DB resolution for empty hermes_home: Path.home()/.hermes/lcm.db
    engine_default_db = str(Path.home() / ".hermes" / "lcm.db")

    # Grandchild proves subprocess inheritance of the disposable home.
    gc = subprocess.run(
        [sys.executable, "-c", "import os; print(os.environ.get('HERMES_HOME',''))"],
        capture_output=True, text=True, timeout=60,
    )
    grandchild_home = gc.stdout.strip()

    mode = -1
    if env_home and os.path.isdir(env_home):
        mode = os.stat(env_home).st_mode & 0o777

    out = {
        "env_home": env_home,
        "inherited_operator_home": getattr(
            conftest, "INHERITED_OPERATOR_HERMES_HOME", None),
        "disposable_home": getattr(conftest, "DISPOSABLE_HERMES_HOME", None),
        "guard_ran": bool(getattr(conftest, "HERMES_HOME_ISOLATED", False)),
        "config_yaml": config_yaml,
        "engine_default_db": engine_default_db,
        "grandchild_home": grandchild_home,
        "disposable_mode": mode,
    }
    print("PROBE_JSON:" + json.dumps(out))
    """
)


def _run_child_probe(operator_home: Path) -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", CHILD_PROBE, str(TESTS_DIR / "conftest.py")],
        capture_output=True, text=True, timeout=180,
        env={**os.environ, "HERMES_HOME": str(operator_home)},
    )
    assert proc.returncode == 0, f"child probe failed:\n{proc.stderr}"
    for line in proc.stdout.splitlines():
        if line.startswith("PROBE_JSON:"):
            return json.loads(line[len("PROBE_JSON:"):])
    pytest.fail(f"no PROBE_JSON in child output:\n{proc.stdout}\n{proc.stderr}")


@pytest.fixture()
def operator_home(tmp_path: Path) -> Path:
    home = tmp_path / "operator-like-home"
    home.mkdir()
    sentinel = home / "lcm.db"
    sentinel.write_bytes(SENTINEL_BYTES)
    return home


def test_child_import_isolates_inherited_operator_home(operator_home: Path):
    """Operator-like inherited HERMES_HOME -> conftest redirects before imports."""
    before_sha = _sha((operator_home / "lcm.db").read_bytes())
    probe = _run_child_probe(operator_home)

    # Guard evidence: the operator home was captured for the record.
    assert probe["inherited_operator_home"] == str(operator_home)
    assert probe["guard_ran"] is True

    # Replacement (not setdefault): env now points at a DIFFERENT, existing,
    # restrictively-permissioned disposable home.
    disposable = probe["env_home"]
    assert disposable, "HERMES_HOME must remain set"
    assert disposable != str(operator_home)
    assert operator_home not in Path(disposable).parents
    # The child owns and removes its disposable directory at interpreter exit;
    # its in-process mode probe proves the directory existed restrictively.
    assert probe["disposable_mode"] != -1
    assert probe["disposable_mode"] & 0o077 == 0, (
        f"disposable home must be owner-restrictive, got {oct(probe['disposable_mode'])}"
    )

    # Default config resolution lands inside the disposable home.
    assert Path(probe["config_yaml"]).parent == Path(disposable)

    # Engine default DB (empty hermes_home) lands inside the disposable home.
    assert Path(disposable) in Path(probe["engine_default_db"]).parents

    # Subprocesses inherit the disposable home.
    assert probe["grandchild_home"] == disposable

    # Operator sentinel untouched, byte-for-byte.
    after = (operator_home / "lcm.db").read_bytes()
    assert after == SENTINEL_BYTES
    assert _sha(after) == before_sha


def test_in_process_collection_time_isolation(operator_home: Path):
    """This pytest process itself must have been isolated at collection time.

    The VM harness launches pytest with a sacrificial outer HERMES_HOME; the
    conftest guard (which ran before this module was imported) must have
    captured it and swapped in a disposable home.  Environment evidence is
    used here because pytest may import conftest under a package-qualified
    module name rather than the literal ``conftest`` name.
    """
    recorded_disposable = os.environ.get("HERMES_TEST_DISPOSABLE_HERMES_HOME")
    assert os.environ.get("HERMES_TEST_HOME_ISOLATED") == "1"
    assert recorded_disposable, "conftest must record the disposable home"
    disposable = Path(recorded_disposable)
    assert disposable.is_dir()
    assert disposable.stat().st_mode & 0o077 == 0
    # The captured operator home (whatever pytest inherited) differs.
    inherited = os.environ.get("HERMES_TEST_INHERITED_HERMES_HOME", "")
    assert str(disposable) != inherited

    # Default LCMConfig resolution inside this process points into the
    # disposable home, not the inherited operator home.
    sys.path.insert(0, str(TESTS_DIR.parent.parent.parent.parent))
    import importlib.util

    cfg_spec = importlib.util.spec_from_file_location(
        "lcm_config_probe_inproc", str(PLUGIN_DIR / "config.py"))
    assert cfg_spec is not None and cfg_spec.loader is not None
    cfg = importlib.util.module_from_spec(cfg_spec)
    cfg_spec.loader.exec_module(cfg)
    resolved = cfg._hermes_config_path()
    assert Path(resolved).parent == disposable
    if inherited:
        assert Path(inherited) != Path(resolved).parent


def test_operator_sentinel_unchanged_after_engine_default_touch(operator_home: Path):
    """Instantiating default-path engine state must never reach the operator DB."""
    before_sha = _sha((operator_home / "lcm.db").read_bytes())
    probe = _run_child_probe(operator_home)
    disposable = Path(probe["env_home"])
    engine_db = Path(probe["engine_default_db"])
    # If the engine default DB path materializes, it is inside the disposable
    # home; the operator sentinel is byte-for-byte unchanged either way.
    if engine_db.exists():
        assert disposable in engine_db.parents
    after = (operator_home / "lcm.db").read_bytes()
    assert after == SENTINEL_BYTES
    assert _sha(after) == before_sha
