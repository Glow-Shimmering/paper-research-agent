"""配置：环境变量 + .env 加载（.env 固定取项目根，与运行目录无关）。

优先级：环境变量非空时优先；环境变量缺失或为空时用 .env 的值
（宿主环境可能注入空值变量，load_dotenv 默认不覆盖它们）。
"""
import os
from pathlib import Path

from dotenv import dotenv_values

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

for _k, _v in (dotenv_values(_PROJECT_ROOT / ".env") or {}).items():
    if _v is not None and (_k not in os.environ or not os.environ[_k]):
        os.environ[_k] = _v

LIBRARY_DIR = Path(os.getenv("PAPER_DATA_DIR", str(Path.home() / ".paper-agent")))
DB_PATH = LIBRARY_DIR / "library.db"
LLM_BASE_URL = os.getenv("PAPER_LLM_BASE_URL", "https://api.deepseek.com")
LLM_API_KEY = os.getenv("PAPER_LLM_API_KEY", "")
LLM_MODEL = os.getenv("PAPER_LLM_MODEL", "deepseek-chat")
EMBED_MODEL = os.getenv("PAPER_EMBED_MODEL", "BAAI/bge-small-zh-v1.5")


def ensure_data_dir() -> Path:
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    return LIBRARY_DIR
