"""Content-addressed, deterministic gzip storage for untrusted raw HTML."""

from __future__ import annotations

import gzip
import hashlib
import io
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

_SNAPSHOT_NAME_RE = re.compile(r"^[0-9a-f]{64}\.html\.gz$")


class SnapshotError(RuntimeError):
    pass


@dataclass(frozen=True)
class SnapshotRef:
    sha256: str
    relative_path: str
    size_bytes: int
    compressed_size_bytes: int


class SnapshotStore:
    def __init__(self, root: str | Path, *, max_bytes: int = 10 * 1024 * 1024) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes 必须大于 0")
        self.root = Path(root).expanduser().resolve(strict=False)
        self.max_bytes = int(max_bytes)

    def save(self, html: bytes) -> SnapshotRef:
        if not isinstance(html, bytes):
            raise TypeError("snapshot HTML 必须是 bytes")
        if not html:
            raise SnapshotError("不能保存空 HTML snapshot")
        if len(html) > self.max_bytes:
            raise SnapshotError("HTML snapshot 超过大小限制")
        digest = hashlib.sha256(html).hexdigest()
        relative_path = f"{digest}.html.gz"
        destination = self.root / relative_path
        compressed = _gzip_deterministic(html)
        self.root.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            existing = self.read(relative_path)
            if existing != html:
                raise SnapshotError("已有 content-addressed snapshot 内容不一致")
            return SnapshotRef(
                sha256=digest,
                relative_path=relative_path,
                size_bytes=len(html),
                compressed_size_bytes=destination.stat().st_size,
            )

        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{digest}.", suffix=".tmp", dir=self.root
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as file:
                file.write(compressed)
                file.flush()
                os.fsync(file.fileno())
            temporary.chmod(0o600)
            os.replace(temporary, destination)
            _fsync_directory(self.root)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return SnapshotRef(
            sha256=digest,
            relative_path=relative_path,
            size_bytes=len(html),
            compressed_size_bytes=len(compressed),
        )

    def read(self, relative_path: str) -> bytes:
        if not _SNAPSHOT_NAME_RE.fullmatch(str(relative_path)):
            raise SnapshotError("snapshot 相对路径无效")
        path = self.root / relative_path
        try:
            with gzip.open(path, "rb") as file:
                content = file.read(self.max_bytes + 1)
        except (OSError, EOFError) as exc:
            raise SnapshotError("snapshot gzip 损坏或无法读取") from exc
        if len(content) > self.max_bytes:
            raise SnapshotError("解压后的 snapshot 超过大小限制")
        expected = relative_path[:64]
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected:
            raise SnapshotError("snapshot 文件名 hash 与内容不一致")
        return content


def _gzip_deterministic(content: bytes) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as file:
        file.write(content)
    return buffer.getvalue()


def _fsync_directory(path: Path) -> None:
    # Windows 的 CRT 不支持目录 descriptor 上的 ``fsync``；snapshot 文件本身
    # 已在原子替换前同步。POSIX 上继续同步目录项以获得崩溃一致性。
    if os.name == "nt":
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
