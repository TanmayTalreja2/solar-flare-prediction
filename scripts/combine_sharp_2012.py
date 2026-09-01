from pathlib import Path

import pandas as pd


Q1_PATH = Path(
    "data/raw/sharp/"
    "sharp_2012_01_01_90d.parquet"
)

APR_DEC_PATH = Path(
    "data/raw/sharp/"
    "sharp_2012_04_01_2012_12_31.parquet"
)

OUTPUT_PATH = Path(
    "data/raw/sharp/"
    "sharp_2012_full_year.parquet"
)


def main() -> None:

    print("========================================")
    print(" COMBINING FULL 2012 SHARP DATA")
    print("========================================")

    print()
    print("Loading Q1...")

    q1 = pd.read_parquet(
        Q1_PATH
    )

    print(
        f"Q1 rows: {len(q1)}"
    )

    print()
    print("Loading April-December...")

    apr_dec = pd.read_parquet(
        APR_DEC_PATH
    )

    print(
        f"April-December rows: "
        f"{len(apr_dec)}"
    )

    # --------------------------------------------------
    # Combine
    # --------------------------------------------------

    print()
    print("Combining datasets...")

    data = pd.concat(
        [
            q1,
            apr_dec,
        ],
        ignore_index=True,
    )

    print(
        f"Rows before deduplication: "
        f"{len(data)}"
    )

    # --------------------------------------------------
    # Duplicate check
    # --------------------------------------------------

    duplicates = data.duplicated(
        subset=[
            "HARPNUM",
            "T_REC",
        ]
    ).sum()

    print(
        f"Duplicate observations: "
        f"{duplicates}"
    )

    data = data.drop_duplicates(
        subset=[
            "HARPNUM",
            "T_REC",
        ]
    )

    # --------------------------------------------------
    # Sort
    # --------------------------------------------------

    data = data.sort_values(
        [
            "T_REC",
            "HARPNUM",
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print()
    print("========================================")
    print(" FULL YEAR SHARP SUMMARY")
    print("========================================")

    print(
        f"Total rows: {len(data)}"
    )

    print(
        f"Columns: {len(data.columns)}"
    )

    print(
        f"Unique HARPNUM: "
        f"{data['HARPNUM'].nunique()}"
    )

    print(
        f"Unique NOAA AR: "
        f"{data['NOAA_AR'].nunique()}"
    )

    print()

    print("Time range:")

    print(
        f"{data['T_REC'].min()} "
        f"→ "
        f"{data['T_REC'].max()}"
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("========================================")
    print(" COMBINATION COMPLETE")
    print("========================================")

    print(
        f"Output: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()