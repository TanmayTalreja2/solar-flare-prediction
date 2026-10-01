from pathlib import Path
import pandas as pd

INPUT_DIR = Path("data/raw/sharp/daily_full_params")
OUTPUT_PATH = Path("data/raw/sharp/sharp_2012_full_year_25params.parquet")


def main():

    print("=" * 60)
    print(" COMBINING FULL-PARAMETER SHARP DATA (2012)")
    print("=" * 60)
    files = sorted(INPUT_DIR.glob("sharp_2012_*_*_full_params.parquet"))

    print(f"Found {len(files)} daily files.")

    if not files:
        print("No files found. Run download_sharp_full_params.py first.")
        return

    frames = []

    for file in files:
        print(f"Loading {file.name}...")
        frames.append(pd.read_parquet(file))

    data = pd.concat(frames, ignore_index=True)

    print(f"Rows before deduplication: {len(data)}")

    duplicates = data.duplicated(subset=["HARPNUM", "T_REC"]).sum()
    print(f"Duplicate observations: {duplicates}")

    data = data.drop_duplicates(subset=["HARPNUM", "T_REC"])

    data = data.sort_values(["T_REC", "HARPNUM"]).reset_index(drop=True)

    print()
    print(f"Total rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")
    print(f"Unique HARPNUM: {data['HARPNUM'].nunique()}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(OUTPUT_PATH, index=False)

    print()
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()