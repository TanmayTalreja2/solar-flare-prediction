from pathlib import Path

import requests


URL = (
    "https://data.ngdc.noaa.gov/platforms/"
    "solar-space-observing-satellites/goes/"
    "multi/l2/data/xrsf-l2-flrpt_science/"
    "csv/sci_xrsf-l2-flrpt_geo_y2012_v1-0-1.csv"
)

OUTPUT_PATH = Path(
    "data/raw/goes/goes_flares_2012.csv"
)


def download_file(url: str, output_path: Path) -> None:
    """Download the NOAA GOES flare report."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Connecting to NOAA...")
    print(f"URL: {url}")
    print()

    response = requests.get(
        url,
        timeout=120,
    )

    response.raise_for_status()

    output_path.write_bytes(
        response.content
    )

    print("Download successful.")
    print(f"Saved to: {output_path}")
    print(
        f"Size: {output_path.stat().st_size:,} bytes"
    )


def main() -> None:
    """Download the 2012 GOES flare dataset."""

    print("Downloading NOAA GOES flare report...")
    print()

    download_file(
        URL,
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()