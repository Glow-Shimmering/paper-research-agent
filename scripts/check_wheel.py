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
    "pragent/web/legacy/index.html",
    "pragent/web/legacy/app.js",
    "pragent/web/legacy/style.css",
    "pragent/web/templates/base.html",
    "pragent/web/templates/projects.html",
    "pragent/web/templates/project_workspace.html",
    "pragent/web/templates/discover.html",
    "pragent/web/templates/library.html",
    "pragent/web/templates/fragments/action_result.html",
    "pragent/web/templates/fragments/discovery_results.html",
    "pragent/web/templates/fragments/questions.html",
    "pragent/web/templates/fragments/sources.html",
    "pragent/web/static/app.css",
    "pragent/web/static/htmx.min.js",
    "pragent/web/static/HTMX-LICENSE.txt",
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
    with tempfile.TemporaryDirectory(prefix="pra-wheel-check-") as raw:
        environment = Path(raw) / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        if os.name == "nt":
            python = environment / "Scripts" / "python.exe"
            pra = environment / "Scripts" / "pra.exe"
        else:
            python = environment / "bin" / "python"
            pra = environment / "bin" / "pra"
        subprocess.run(
            [str(python), "-m", "pip", "install", "--no-deps", str(wheel.resolve())],
            check=True,
        )
        version = subprocess.run(
            [str(pra), "--version"],
            check=True,
            capture_output=True,
            text=True,
            env=child_env,
        ).stdout.strip()
        probe = """
from pathlib import Path
import sys
import tempfile
import pragent
from fastapi.testclient import TestClient
from pragent.store import Store
from pragent.webapp import _web_directory, create_app

assert str(Path(pragent.__file__).resolve()).startswith(str(Path(sys.prefix).resolve()))
assert Path(_web_directory()).is_dir()
with tempfile.TemporaryDirectory() as raw:
    store = Store(Path(raw) / 'wheel.db')
    with TestClient(
        create_app(store=store), base_url='http://127.0.0.1'
    ) as client:
        response = client.get('/')
        assert response.status_code == 200
        assert '<html' in response.text.lower()
        workspace = client.get('/ui/projects')
        assert workspace.status_code == 200
        assert '研究项目' in workspace.text
        discover = client.get('/ui/discover')
        assert discover.status_code == 200
        assert '多来源论文发现' in discover.text
        library = client.get('/ui/library')
        assert library.status_code == 200
        assert '统一来源库' in library.text
    store.close()
"""
        subprocess.run([str(python), "-c", probe], check=True, env=child_env)
        print(f"isolated wheel smoke passed: {version}")


def main() -> None:
    wheel_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "dist")
    wheels = sorted(wheel_dir.glob("paper_research_agent-*.whl"))
    if len(wheels) != 1:
        raise SystemExit(
            f"expected exactly one paper-research-agent wheel in {wheel_dir}, "
            f"found {len(wheels)}"
        )
    with zipfile.ZipFile(wheels[0]) as archive:
        names = set(archive.namelist())
    missing = sorted(REQUIRED - names)
    if missing:
        raise SystemExit("wheel is missing runtime assets: " + ", ".join(missing))
    print(f"wheel assets verified: {wheels[0].name}")
    _run_installed_smoke(wheels[0])


if __name__ == "__main__":
    main()
