"""Verify wheel contents, entry point, import path, and packaged Web runtime."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import venv
import zipfile
from pathlib import Path


REQUIRED = {
    "paper_agent/web/index.html",
    "paper_agent/web/app.js",
    "paper_agent/web/style.css",
}


def _run_installed_smoke(wheel: Path) -> None:
    dependency_site = next(
        (Path(path) for path in sys.path if path and (Path(path) / "fastapi").is_dir()),
        None,
    )
    if dependency_site is None:
        raise SystemExit("cannot locate the validated runtime dependencies for wheel smoke")
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = str(dependency_site)
    child_env["PYTHONNOUSERSITE"] = "1"
    with tempfile.TemporaryDirectory(prefix="paper-agent-wheel-check-") as raw:
        environment = Path(raw) / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        if os.name == "nt":
            python = environment / "Scripts" / "python.exe"
            paper = environment / "Scripts" / "paper.exe"
        else:
            python = environment / "bin" / "python"
            paper = environment / "bin" / "paper"
        subprocess.run(
            [str(python), "-m", "pip", "install", "--no-deps", str(wheel.resolve())],
            check=True,
        )
        version = subprocess.run(
            [str(paper), "--version"],
            check=True,
            capture_output=True,
            text=True,
            env=child_env,
        ).stdout.strip()
        probe = """
from pathlib import Path
import sys
import paper_agent
from fastapi.testclient import TestClient
from paper_agent.webapp import _web_directory, create_app

assert str(Path(paper_agent.__file__).resolve()).startswith(str(Path(sys.prefix).resolve()))
assert Path(_web_directory()).is_dir()
response = TestClient(create_app()).get('/')
assert response.status_code == 200
assert '<html' in response.text.lower()
"""
        subprocess.run([str(python), "-c", probe], check=True, env=child_env)
        print(f"isolated wheel smoke passed: {version}")


def main() -> None:
    wheel_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "dist")
    wheels = sorted(wheel_dir.glob("paper_agent-*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected exactly one paper-agent wheel in {wheel_dir}, found {len(wheels)}")
    with zipfile.ZipFile(wheels[0]) as archive:
        names = set(archive.namelist())
    missing = sorted(REQUIRED - names)
    if missing:
        raise SystemExit("wheel is missing runtime assets: " + ", ".join(missing))
    print(f"wheel assets verified: {wheels[0].name}")
    _run_installed_smoke(wheels[0])


if __name__ == "__main__":
    main()
