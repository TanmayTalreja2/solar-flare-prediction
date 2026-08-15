from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "data/raw/goes/goes_flares_2012.csv"
)

OUTPUT_PATH = Path(
    "data/processed/goes/goes_flares_2012.parquet"
)


def load_goes_data(path: Path) -> pd.DataFrame:
    """Load the NOAA GOES flare CSV."""

    print(f"Loading GOES data from: {path}")

    # NOAA's CSV contains some fields that can confuse the
    # default pandas C parser. The Python parser is more tolerant.
    data = pd.read_csv(
        path,
        engine="python",
    )

    print(f"Rows loaded: {len(data)}")
    print(f"Columns loaded: {len(data.columns)}")

    return data


def clean_goes_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize GOES flare records."""

    data = data.copy()

    # Normalize column names.
    data.columns = [
        column.strip().lower()
        for column in data.columns
    ]

    print()
    print("Available columns:")
    print(data.columns.tolist())

    # NOAA's current GOES Flare Report uses "time"
    # for the flare peak time.
    if "time" in data.columns:
        data = data.rename(
            columns={"time": "peak_time"}
        )

    # Convert timestamps.
    datetime_columns = [
        "start_time",
        "peak_time",
        "end_time",
    ]

    for column in datetime_columns:
        if column in data.columns:
            data[column] = pd.to_datetime(
                data[column],
                errors="coerce",
            )

    # Standardize flare class.
    data["flare_class"] = (
        data["flare_class"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    # Extract A/B/C/M/X category.
    data["flare_category"] = (
        data["flare_class"].str[0]
    )

    # Extract numerical magnitude.
    data["flare_magnitude"] = pd.to_numeric(
        data["flare_class"].str[1:],
        errors="coerce",
    )

    # Convert active region to numeric.
    data["active_region"] = pd.to_numeric(
        data["active_region"],
        errors="coerce",
    )

    # Remove records without a valid peak time.
    data = data.dropna(
        subset=["peak_time"]
    )

    # Sort chronologically.
    data = (
        data.sort_values("peak_time")
        .reset_index(drop=True)
    )

    return data

def save_data(
    data: pd.DataFrame,
    path: Path,
) -> None:
    """Save processed GOES data as Parquet."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_parquet(
        path,
        index=False,
    )

    print()
    print("========== SAVE COMPLETE ==========")
    print(f"Output: {path}")
    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")


def main() -> None:
    """Run the GOES parsing pipeline."""

    data = load_goes_data(
        INPUT_PATH
    )

    data = clean_goes_data(
        data
    )

    save_data(
        data,
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()