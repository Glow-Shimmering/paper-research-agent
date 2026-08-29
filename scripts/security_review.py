"""Step 28 确定性隐私/安全审查：扫描仓库与 wheel 中的敏感信息泄漏。

离线、确定性的静态检查，对应 roadmap「Security/privacy/data checks」中可自动化的部分：
- Git 跟踪文件与 wheel 内容中不得出现可用的 key 字面量（DeepSeek/Semantic Scholar/Web API）；
- `.env` 不得被 Git 跟踪，`.env.example` 中 key 值必须为空；
- `src/pragent/**` 与 wheel 内不得出现本机绝对路径；
- wheel 内不得打包 `.env`、snapshot（*.html.gz）或原始 PDF；
- 模板中 `| safe` 的使用必须逐个列出让审查者复核（不自动判定为失败）。

已由合同测试覆盖、此处不再重复的边界（见 docs/evaluation.md 的映射表）：
公开响应脱敏、snapshot 不注入 DOM、CSRF、远程监听 API key + TLS、SSRF。

用法：
    python scripts/security_review.py                 # 只扫描仓库
    python scripts/security_review.py --wheel dist    # 同时扫描 dist 下的 wheel
退出码：0 = 全部通过；1 = 存在失败项。
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# 可用凭据的形态：非空、非占位符赋值。占位符（空值/示例占位）不算泄漏。
_SECRET_ASSIGNMENTS = (
    (re.compile(r"^PRA_LLM_API_KEY=(?!\s*$)\S+", re.MULTILINE), "PRA_LLM_API_KEY"),
    (
        re.compile(
            r"^PRA_SEMANTIC_SCHOLAR_API_KEY=(?!\s*$)\S+", re.MULTILINE
        ),
        "PRA_SEMANTIC_SCHOLAR_API_KEY",
    ),
    (re.compile(r"^PRA_WEB_API_KEY=(?!\s*$)\S+", re.MULTILINE), "PRA_WEB_API_KEY"),
    (re.compile(r"^PRA_CROSSREF_EMAIL=(?!\s*$)\S+", re.MULTILINE), "PRA_CROSSREF_EMAIL"),
)
# 常见云 provider key 形态（sk- 前缀长随机串）。
bearer_like = re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")

_PLACEHOLDER_VALUES = frozenset(
    {
        "your-api-key",
        "your_api_key",
        "changeme",
        "example",
        "you@example.com",
        "test-key",
        "placeholder",
    }
)

_HOST_PATHS = (
    re.compile(r"/Users/[A-Za-z0-9_.-]+/"),
    re.compile(r"/home/[A-Za-z0-9_.-]+/"),
    re.compile(r"\b[A-Za-z]:\\\\Users\\\\"),
)

_SCAN_SKIP_PARTS = frozenset(
    {".git", ".venv", ".pytest-tmp", "dist", "build", ".mimosa", ".v2c", "__pycache__"}
)


def _placeholder(value: str) -> bool:
    lowered = value.strip().lower()
    return lowered in _PLACEHOLDER_VALUES or lowered.startswith("<")


def _tracked_files() -> list[Path]:
    import subprocess

    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return [
        REPO_ROOT / raw.decode("utf-8")
        for raw in result.stdout.split(b"\0")
        if raw
    ]


def _iter_repo_files():
    for path in _tracked_files():
        if path.is_file():
            yield path


def _iter_wheel_files(wheel_dir: Path):
    for wheel in sorted(wheel_dir.glob("*.whl")):
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                yield wheel, name, archive.read(name)


def _scan_text(text: str, findings: list[str], location: str) -> None:
    for pattern, label in _SECRET_ASSIGNMENTS:
        for match in pattern.finditer(text):
            value = match.group(0).split("=", 1)[1]
            if not _placeholder(value):
                findings.append(f"{location}: 可能可用的 {label} 字面量")
    if bearer_like.search(text):
        findings.append(f"{location}: 疑似 sk- 形态的 key 字面量")


def review_repo(findings: list[str], notes: list[str]) -> None:
    env_example = REPO_ROOT / ".env.example"
    if env_example.is_file():
        text = env_example.read_text(encoding="utf-8")
        for pattern, label in _SECRET_ASSIGNMENTS:
            for match in pattern.finditer(text):
                value = match.group(0).split("=", 1)[1]
                if not _placeholder(value):
                    findings.append(
                        f".env.example: {label} 必须保持空值或显式占位符"
                    )
    else:
        notes.append(".env.example 不存在（跳过占位符检查）")

    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        import subprocess

        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", ".env"],
            cwd=REPO_ROOT,
            capture_output=True,
        )
        if tracked.returncode == 0:
            findings.append(".env 被 Git 跟踪；本地配置文件不得入库")

    for path in _iter_repo_files():
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative == ".env.example":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        _scan_text(text, findings, relative)

    for path in (REPO_ROOT / "src" / "pragent").rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".html", ".css", ".js"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in _HOST_PATHS:
            for match in pattern.finditer(text):
                findings.append(
                    f"src/pragent/{path.relative_to(REPO_ROOT / 'src' / 'pragent')}: "
                    f"本机绝对路径 {match.group(0)!r}"
                )

    templates = REPO_ROOT / "src" / "pragent" / "web" / "templates"
    if templates.is_dir():
        for template in templates.rglob("*.html"):
            text = template.read_text(encoding="utf-8", errors="ignore")
            for index, line in enumerate(text.splitlines(), start=1):
                if "|safe" in line or "| safe" in line:
                    notes.append(
                        "模板 |safe 使用需人工复核："
                        f"{template.relative_to(REPO_ROOT)}:{index}: {line.strip()[:120]}"
                    )


def review_wheel(wheel_dir: Path, findings: list[str]) -> list[str]:
    checked = []
    for wheel, name, payload in _iter_wheel_files(wheel_dir):
        checked.append(f"{wheel.name}!/{name}")
        base = name.rsplit("/", 1)[-1]
        if base == ".env" or base.endswith(".html.gz") or base.endswith(".pdf"):
            findings.append(f"{wheel.name}!/{name}: wheel 不应打包 .env/snapshot/PDF")
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            continue
        _scan_text(text, findings, f"{wheel.name}!/{name}")
        if name.startswith("pragent/"):
            for pattern in _HOST_PATHS:
                for match in pattern.finditer(text):
                    findings.append(
                        f"{wheel.name}!/{name}: 本机绝对路径 {match.group(0)!r}"
                    )
    return checked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wheel",
        type=Path,
        help="可选：同时扫描该目录下的 *.whl",
    )
    args = parser.parse_args()

    findings: list[str] = []
    notes: list[str] = []
    review_repo(findings, notes)

    checked_count = 0
    if args.wheel is not None:
        checked = review_wheel(args.wheel, findings)
        checked_count = len(checked)
        if checked_count == 0:
            findings.append(f"--wheel {args.wheel}: 目录中没有找到 wheel")

    print("== 隐私/安全静态审查（离线确定性检查）==")
    print(f"发现：{len(findings)} 项；人工复核提示：{len(notes)} 项")
    for finding in findings:
        print(f"[FAIL] {finding}")
    for note in notes:
        print(f"[REVIEW] {note}")
    if args.wheel is not None:
        print(f"wheel 条目扫描数：{checked_count}")
    if findings:
        print("审查未通过")
        return 1
    print("审查通过：无可自动检出的 key 泄漏、.env 跟踪、本机绝对路径或敏感打包内容")
    return 0


if __name__ == "__main__":
    sys.exit(main())
