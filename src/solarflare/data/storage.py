from pathlib import Path

BYTES_PER_GB = 1024**3


def get_directory_size(directory: Path) -> int:
    """Return the total size of all files inside a directory in bytes."""

    if not directory.exists():
        return 0

    total_size = 0

    for path in directory.rglob("*"):
        if path.is_file():
            total_size += path.stat().st_size

    return total_size


def bytes_to_gb(size_bytes: int) -> float:
    """Convert bytes to gibibytes."""

    return size_bytes / BYTES_PER_GB


def has_storage_capacity(
    data_directory: Path,
    incoming_bytes: int,
    limit_gb: float,
) -> bool:
    """Check whether a new download would remain within the storage limit."""

    current_size = get_directory_size(data_directory)
    maximum_size = int(limit_gb * BYTES_PER_GB)

    return current_size + incoming_bytes <= maximum_size


def ensure_storage_capacity(
    data_directory: Path,
    incoming_bytes: int,
    limit_gb: float,
) -> None:
    """Raise an error if a download would exceed the dataset storage limit."""

    if incoming_bytes < 0:
        raise ValueError("incoming_bytes cannot be negative.")

    if limit_gb <= 0:
        raise ValueError("limit_gb must be greater than zero.")

    if not has_storage_capacity(
        data_directory=data_directory,
        incoming_bytes=incoming_bytes,
        limit_gb=limit_gb,
    ):
        current_gb = bytes_to_gb(
            get_directory_size(data_directory)
        )

        incoming_gb = bytes_to_gb(incoming_bytes)

        raise RuntimeError(
            "Storage limit exceeded. "
            f"Current: {current_gb:.2f} GB, "
            f"incoming: {incoming_gb:.2f} GB, "
            f"limit: {limit_gb:.2f} GB."
        )