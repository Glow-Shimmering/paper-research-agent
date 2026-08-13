"""配置：环境变量 + 可发现的 ``.env`` 文件。

查找顺序：``PAPER_ENV_FILE`` 显式路径、当前工作目录的 ``.env``、
editable 源码仓库根目录的 ``.env``。非空环境变量优先；环境变量缺失或
为空时才使用文件值（部分宿主环境会注入空值变量）。
"""
import os
from pathlib import Path
from typing import Optional

from dotenv import dotenv_values


def _env_or_default(name: str, default: str) -> str:
    """把宿主注入的空环境变量视为未设置。"""
    return os.getenv(name) or default


def _find_env_file(explicit: Optional[str], cwd: Path, module_file: Path) -> Optional[Path]:
    """返回应加载的环境文件，不依赖包在文件系统中的固定层级。"""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = cwd / path
        path = path.resolve()
        return path if path.is_file() else None

    cwd_env = (cwd / ".env").resolve()
    if cwd_env.is_file():
        return cwd_env

    package_dir = module_file.resolve().parent
    for parent in (package_dir, *package_dir.parents):
        if (parent / "pyproject.toml").is_file():
            repo_env = parent / ".env"
            return repo_env.resolve() if repo_env.is_file() else None
    return None


ENV_FILE = _find_env_file(os.getenv("PAPER_ENV_FILE"), Path.cwd(), Path(__file__))

for _k, _v in (dotenv_values(ENV_FILE) if ENV_FILE else {}).items():
    if _v is not None and (_k not in os.environ or not os.environ[_k]):
        os.environ[_k] = _v

def _default_library_dir() -> Path:
    """数据目录：优先 ``~/.pagent``；存在旧版 ``~/.paper-agent`` 时无缝沿用。"""
    home = Path.home()
    legacy = home / ".paper-agent"
    if legacy.is_dir():
        return legacy
    return home / ".pagent"


LIBRARY_DIR = Path(_env_or_default("PAPER_DATA_DIR", str(_default_library_dir())))
DB_PATH = LIBRARY_DIR / "library.db"
LLM_BASE_URL = _env_or_default("PAPER_LLM_BASE_URL", "https://api.deepseek.com")
LLM_API_KEY = os.getenv("PAPER_LLM_API_KEY", "")
LLM_MODEL = _env_or_default("PAPER_LLM_MODEL", "deepseek-chat")
WEB_API_KEY = os.getenv("PAPER_WEB_API_KEY", "")
EMBED_MODEL = _env_or_default("PAPER_EMBED_MODEL", "BAAI/bge-small-zh-v1.5")


def download_dir_override() -> Optional[Path]:
    """下载论文的显式目录：PAPER_DOWNLOAD_DIR > PAPER_DATA_DIR；都未显式设置返回 None。

    仅接受显式设置（env 中存在即生效），避免默认值目录被误用作下载目标。
    """
    for key in ("PAPER_DOWNLOAD_DIR", "PAPER_DATA_DIR"):
        raw = os.getenv(key)
        if raw:
            return Path(raw).resolve()
    return None


def notes_dir() -> Path:
    """笔记保存目录：PAPER_NOTE_DIR 显式设置优先，否则 PAPER_DATA_DIR 下 notes。"""
    raw = os.getenv("PAPER_NOTE_DIR")
    if raw:
        return Path(raw).resolve()
    return LIBRARY_DIR / "notes"


def ensure_data_dir() -> Path:
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    return LIBRARY_DIR
