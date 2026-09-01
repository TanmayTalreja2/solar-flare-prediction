from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MAGNETOGRAM_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "magnetograms"
)

print()
print("=" * 60)
print("MAGNETOGRAM FILENAME INSPECTION")
print("=" * 60)

files = list(MAGNETOGRAM_DIR.glob("*.npz"))

print(f"\nTotal NPZ files: {len(files)}")

pattern = re.compile(
    r"harp_(\d+)_(\d{8})_(\d{6})_t[01]\.npz"
)

parsed = []
unparsed = []

for file_path in files:

    if pattern.fullmatch(file_path.name):
        parsed.append(file_path.name)
    else:
        unparsed.append(file_path.name)

print(f"Parsed filenames:   {len(parsed)}")
print(f"Unparsed filenames: {len(unparsed)}")

print()
print("=" * 60)
print("FIRST 30 PARSED FILES")
print("=" * 60)

for filename in parsed[:30]:
    print(filename)

print()
print("=" * 60)
print("FIRST 50 UNPARSED FILES")
print("=" * 60)

for filename in unparsed[:50]:
    print(filename)

print()
print("=" * 60)
print("LAST 50 UNPARSED FILES")
print("=" * 60)

for filename in unparsed[-50:]:
    print(filename)