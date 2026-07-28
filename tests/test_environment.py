import sys


def test_python_version() -> None:
    """Ensure the project runs on Python 3.11 or newer."""

    assert sys.version_info >= (3, 11)