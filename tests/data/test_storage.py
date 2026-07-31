from pathlib import Path

import pytest

from solarflare.data.storage import (
    bytes_to_gb,
    ensure_storage_capacity,
    get_directory_size,
    has_storage_capacity,
)


def test_empty_directory_size(tmp_path: Path) -> None:
    assert get_directory_size(tmp_path) == 0


def test_directory_size(tmp_path: Path) -> None:
    test_file = tmp_path / "sample.bin"
    test_file.write_bytes(b"1234567890")

    assert get_directory_size(tmp_path) == 10


def test_bytes_to_gb() -> None:
    assert bytes_to_gb(1024**3) == 1.0


def test_storage_capacity(tmp_path: Path) -> None:
    assert has_storage_capacity(
        data_directory=tmp_path,
        incoming_bytes=1024,
        limit_gb=1,
    )


def test_storage_limit_rejected(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        ensure_storage_capacity(
            data_directory=tmp_path,
            incoming_bytes=2 * 1024**3,
            limit_gb=1,
        )