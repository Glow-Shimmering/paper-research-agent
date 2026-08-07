"""检查 pytest 临时目录与已知下载残留，防止死循环写入占满磁盘。

用法：.venv/Scripts/python scripts/check_tmp_space.py
历史事故：测试 fake 死循环曾在系统 Temp 的 pytest-of-Glow 写入 78G 残留。
pytest basetemp 已移到项目内 .pytest-tmp，但系统 Temp 旧残留仍需监控。
"""
import sys
from pathlib import Path

THRESHOLD_MB = 500  # 超过此大小告警

TARGETS = [
    Path(__file__).resolve().parent.parent / ".pytest-tmp",
    Path.home() / "AppData" / "Local" / "Temp" / "pytest-of-Glow",
    Path.home() / "AppData" / "Local" / "Temp" / "pa-dl-test",
]


def dir_size_mb(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for f in path.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return total // (1024 * 1024)


def main() -> None:
    ok = True
    for target in TARGETS:
        size = dir_size_mb(target)
        if size == 0:
            continue
        flag = "⚠ 过大" if size > THRESHOLD_MB else "正常"
        print(f"{target}: {size} MB {flag}")
        if size > THRESHOLD_MB:
            ok = False
    if not ok:
        print("发现异常占用的临时目录，请检查是否存在死循环写入残留，确认后删除。")
        sys.exit(1)
    print("临时目录检查通过。")


if __name__ == "__main__":
    main()
