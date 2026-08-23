from pathlib import Path

from pragent.config import _env_or_default, _find_env_file


def test_empty_environment_value_uses_default(monkeypatch):
    monkeypatch.setenv("PRA_TEST_DEFAULT", "")

    assert _env_or_default("PRA_TEST_DEFAULT", "fallback") == "fallback"


def test_find_env_file_prefers_explicit_relative_path(tmp_path):
    explicit = tmp_path / "custom.env"
    explicit.write_text("PRA_LLM_MODEL=test\n", encoding="utf-8")
    cwd_env = tmp_path / ".env"
    cwd_env.write_text("PRA_LLM_MODEL=cwd\n", encoding="utf-8")

    found = _find_env_file("custom.env", tmp_path, Path(__file__))

    assert found == explicit.resolve()


def test_missing_explicit_env_does_not_fall_back(tmp_path):
    (tmp_path / ".env").write_text("PRA_LLM_MODEL=cwd\n", encoding="utf-8")

    assert _find_env_file("missing.env", tmp_path, Path(__file__)) is None


def test_find_env_file_uses_current_working_directory(tmp_path):
    cwd_env = tmp_path / ".env"
    cwd_env.write_text("PRA_LLM_MODEL=cwd\n", encoding="utf-8")

    assert _find_env_file(None, tmp_path, Path(__file__)) == cwd_env.resolve()


def test_find_env_file_discovers_editable_repository_root(tmp_path):
    repo = tmp_path / "checkout"
    package = repo / "src" / "pragent"
    package.mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='paper-research-agent'\n", encoding="utf-8")
    repo_env = repo / ".env"
    repo_env.write_text("PRA_LLM_MODEL=repo\n", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    found = _find_env_file(None, elsewhere, package / "config.py")

    assert found == repo_env.resolve()
