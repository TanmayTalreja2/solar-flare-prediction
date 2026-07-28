import importlib
import platform
import sys

REQUIRED_PACKAGES = [
    "numpy",
    "pandas",
    "sklearn",
    "matplotlib",
    "plotly",
    "requests",
    "httpx",
    "fastapi",
    "streamlit",
    "pydantic",
]


def check_python_version() -> bool:
    """Check whether the active Python version is supported."""

    version = sys.version_info

    print(
        f"Python: {version.major}.{version.minor}.{version.micro}"
    )

    return version >= (3, 11)


def check_packages() -> bool:
    """Check whether required project packages can be imported."""

    all_available = True

    for package in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "unknown")

            print(f"[OK] {package}: {version}")

        except ImportError:
            print(f"[MISSING] {package}")
            all_available = False

    return all_available


def main() -> None:
    """Run environment validation checks."""

    print("=" * 60)
    print("Solar Flare Prediction System - Environment Check")
    print("=" * 60)

    print(f"Operating system: {platform.system()}")
    print(f"Platform: {platform.platform()}")
    print()

    python_ok = check_python_version()

    print()
    packages_ok = check_packages()

    print()
    print("=" * 60)

    if python_ok and packages_ok:
        print("Environment check PASSED.")
        return

    print("Environment check FAILED.")
    sys.exit(1)


if __name__ == "__main__":
    main()