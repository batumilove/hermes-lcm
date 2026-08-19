"""Guard: the conftest isolation must redirect a live-home HERMES_HOME to a
temp dir, leave an explicit non-live HERMES_HOME untouched, and honor the
LCM_TESTS_ALLOW_REAL_HOME opt-out. Run with the worktree's venv python:

    cd <worktree> && python tests/test_conftest_db_isolation.py
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

WORKTREE = Path(__file__).resolve().parent.parent
CONFTEST_SNIPPET = "_isolate_hermes_home_if_live"


def run_probe(extra_env: dict[str, str]) -> str:
    code = (
        "import sys; sys.path.insert(0, 'tests'); import conftest; "
        "import os; print(os.environ.get('HERMES_HOME', '<unset>'))"
    )
    env = {k: v for k, v in os.environ.items() if k != "HERMES_HOME"}
    env.update(extra_env)
    cp = subprocess.run([sys.executable, "-c", code], cwd=WORKTREE,
                        capture_output=True, text=True, env=env, timeout=120)
    if cp.returncode != 0:
        raise SystemExit(f"probe failed: {cp.stderr[-800:]}")
    return cp.stdout.strip().splitlines()[-1]


def main() -> None:
    real_home = str(Path.home())

    # 1. Unset HERMES_HOME -> must be redirected off the real home.
    got = run_probe({})
    assert got and not got.startswith(real_home), f"case1 leaked real home: {got}"
    print(f"case1 unset -> isolated: OK ({got})")

    # 2. HERMES_HOME = real home -> redirected.
    got = run_probe({"HERMES_HOME": real_home})
    assert got and not got.startswith(real_home), f"case2 leaked real home: {got}"
    print(f"case2 real-home -> isolated: OK ({got})")

    # 3. HERMES_HOME = explicit tmp dir (like tests that monkeypatch) -> unchanged.
    explicit = tempfile.mkdtemp(prefix="explicit-hermes-home-")
    got = run_probe({"HERMES_HOME": explicit})
    assert got == explicit, f"case3 rewrote explicit home: {got}"
    print(f"case3 explicit tmp -> unchanged: OK")

    # 4. Opt-out honored.
    got = run_probe({"LCM_TESTS_ALLOW_REAL_HOME": "1"})
    assert got == "<unset>" or got.startswith(real_home), f"case4 opt-out broken: {got}"
    print("case4 opt-out -> unchanged: OK")

    print("ALL PASS")


if __name__ == "__main__":
    main()
