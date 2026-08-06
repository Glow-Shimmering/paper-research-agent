"""配置：环境变量 + .env 加载（.env 固定取项目根，与运行目录无关）。"""
import os
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

LIBRARY_DIR = Path(os.getenv("PAPER_DATA_DIR", str(Path.home() / ".paper-agent")))
DB_PATH = LIBRARY_DIR / "library.db"
LLM_BASE_URL = os.getenv("PAPER_LLM_BASE_URL", "https://api.deepseek.com")
LLM_API_KEY = os.getenv("PAPER_LLM_API_KEY", "")
LLM_MODEL = os.getenv("PAPER_LLM_MODEL", "deepseek-chat")
EMBED_MODEL = os.getenv("PAPER_EMBED_MODEL", "BAAI/bge-small-zh-v1.5")


def ensure_data_dir() -> Path:
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    return LIBRARY_DIR
