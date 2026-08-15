from pathlib import Path

import pandas as pd


DATA_PATH = Path(
    "data/processed/goes/goes_flares_2012.parquet"
)


def main() -> None:
    """Inspect the processed GOES flare dataset."""

    data = pd.read_parquet(DATA_PATH)

    print("========== GOES DATASET ==========")
    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")
    print()

    print("========== FLARE CATEGORIES ==========")
    print(
        data["flare_category"]
        .value_counts()
        .sort_index()
    )
    print()

    print("========== M/X FLARES ==========")

    mx_flares = data[
        data["flare_category"].isin(
            ["M", "X"]
        )
    ]

    print(f"M/X flares: {len(mx_flares)}")
    print()

    print(
        mx_flares[
            [
                "start_time",
                "peak_time",
                "end_time",
                "flare_class",
                "active_region",
            ]
        ].head(20)
    )

    print()

    print("========== MARCH 7, 2012 ==========")

    march_7 = data[
        data["peak_time"].dt.date
        == pd.Timestamp("2012-03-07").date()
    ]

    print(
        march_7[
            [
                "start_time",
                "peak_time",
                "end_time",
                "flare_class",
                "active_region",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
    