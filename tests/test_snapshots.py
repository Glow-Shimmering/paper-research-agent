import gzip
import os
import stat

import pytest

from pragent.ingestion.snapshots import SnapshotError, SnapshotStore


def test_content_addressed_snapshot_is_deterministic_idempotent_and_private(tmp_path):
    store = SnapshotStore(tmp_path / "snapshots")
    html = b"<html><article>Evidence report</article></html>"

    first = store.save(html)
    compressed_once = (store.root / first.relative_path).read_bytes()
    second = store.save(html)
    compressed_twice = (store.root / second.relative_path).read_bytes()

    assert first == second
    assert first.relative_path == f"{first.sha256}.html.gz"
    assert compressed_once == compressed_twice
    assert gzip.decompress(compressed_once) == html
    assert store.read(first.relative_path) == html
    if os.name == "posix":
        # Windows 的 st_mode 不表达 NTFS ACL，不能用 POSIX mode bits 验证。
        mode = stat.S_IMODE((store.root / first.relative_path).stat().st_mode)
        assert mode == stat.S_IRUSR | stat.S_IWUSR


def test_snapshot_rejects_traversal_oversize_corruption_and_hash_mismatch(tmp_path):
    store = SnapshotStore(tmp_path, max_bytes=20)
    with pytest.raises(SnapshotError, match="大小限制"):
        store.save(b"x" * 21)
    with pytest.raises(SnapshotError, match="相对路径"):
        store.read("../secret.html.gz")

    valid = store.save(b"small html")
    path = store.root / valid.relative_path
    path.write_bytes(b"not gzip")
    with pytest.raises(SnapshotError, match="损坏"):
        store.read(valid.relative_path)

    other_name = "f" * 64 + ".html.gz"
    with gzip.open(store.root / other_name, "wb") as file:
        file.write(b"different")
    with pytest.raises(SnapshotError, match="hash"):
        store.read(other_name)


def test_snapshot_existing_corruption_fails_closed_instead_of_overwriting(tmp_path):
    store = SnapshotStore(tmp_path)
    html = b"<html>same identity</html>"
    saved = store.save(html)
    path = store.root / saved.relative_path
    path.write_bytes(b"corrupt")

    with pytest.raises(SnapshotError):
        store.save(html)
    assert path.read_bytes() == b"corrupt"
