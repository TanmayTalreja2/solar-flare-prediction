"""
Download 3-channel HMI SHARP magnetograms from JSOC.

Channels:
    Br = radial magnetic field
    Bp = photospheric field component
    Bt = transverse magnetic field

Uses the EXISTING dataset_labels.csv so that the 3-channel
dataset contains exactly the same observations as the current
Br-only CNN dataset.

Existing Br-only dataset is NOT modified.

Output:
    data/processed/magnetograms_3ch/*.npz

Each NPZ contains:
    img -> (3, 224, 224)
"""

import io
import time
import warnings
from pathlib import Path

import drms
import numpy as np
import pandas as pd
import requests

from astropy.io import fits
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent

LABELS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "magnetograms"
    / "dataset_labels.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "magnetograms_3ch"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# JSOC SETTINGS
# ============================================================

JSOC_BASE_URL = "http://jsoc.stanford.edu"

SERIES = "hmi.sharp_cea_720s"

SEGMENTS = [
    "Br",
    "Bp",
    "Bt",
]

TARGET_SIZE = (224, 224)


# ============================================================
# FITS PROCESSING
# ============================================================

def read_fits_bytes(
    fits_bytes,
):
    """
    Read the image data from FITS bytes.
    """

    try:

        hdul = fits.open(
            io.BytesIO(fits_bytes)
        )

        data = (
            hdul[1].data
            if len(hdul) > 1
            else hdul[0].data
        )

        hdul.close()

        if data is None:
            return None

        data = np.asarray(
            data,
            dtype=np.float32
        )

        # Replace NaN / Inf
        data = np.nan_to_num(
            data,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        return data

    except Exception as e:

        print(
            f"FITS processing error: {e}"
        )

        return None


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_component(
    data,
):
    """
    Normalize magnetic field component.

    Same basic normalization used for the existing Br-only
    dataset:

        clip ±3000 Gauss
        scale to [-1, 1]
    """

    data = np.clip(
        data,
        -3000.0,
        3000.0
    )

    data = data / 3000.0

    return data.astype(
        np.float32
    )


# ============================================================
# CROP / PAD
# ============================================================

def crop_or_pad(
    data,
    target_size=TARGET_SIZE,
):
    """
    Center crop or zero-pad image to 224x224.
    """

    target_h, target_w = (
        target_size
    )

    h, w = data.shape

    # --------------------------------------------------------
    # Padding
    # --------------------------------------------------------

    if h < target_h or w < target_w:

        pad_h = max(
            0,
            target_h - h
        )

        pad_w = max(
            0,
            target_w - w
        )

        data = np.pad(
            data,
            (
                (
                    pad_h // 2,
                    pad_h - pad_h // 2
                ),
                (
                    pad_w // 2,
                    pad_w - pad_w // 2
                ),
            ),
            mode="constant",
            constant_values=0,
        )

        h, w = data.shape

    # --------------------------------------------------------
    # Center crop
    # --------------------------------------------------------

    y_start = (
        h - target_h
    ) // 2

    x_start = (
        w - target_w
    ) // 2

    cropped = data[
        y_start:
        y_start + target_h,
        x_start:
        x_start + target_w
    ]

    return cropped.astype(
        np.float32
    )


# ============================================================
# PROCESS 3 COMPONENTS
# ============================================================

def process_three_components(
    component_bytes,
    output_path,
):
    """
    Convert Br/Bp/Bt FITS files into one
    (3, 224, 224) compressed NPZ.
    """

    processed = []

    for component in SEGMENTS:

        fits_bytes = component_bytes.get(
            component
        )

        if fits_bytes is None:
            return False

        data = read_fits_bytes(
            fits_bytes
        )

        if data is None:
            return False

        data = normalize_component(
            data
        )

        data = crop_or_pad(
            data
        )

        processed.append(
            data
        )

    # --------------------------------------------------------
    # Stack channels
    # --------------------------------------------------------

    image = np.stack(
        processed,
        axis=0
    )

    # Expected:
    # (3, 224, 224)

    if image.shape != (
        3,
        224,
        224
    ):

        print(
            f"Unexpected shape: "
            f"{image.shape}"
        )

        return False

    np.savez_compressed(
        output_path,
        img=image
    )

    return True


# ============================================================
# BUILD FILENAME
# ============================================================

def build_output_filename(
    row,
):

    harpnum = int(
        row["HARPNUM"]
    )

    timestamp = pd.to_datetime(
        row["observation_time"]
    )

    target = int(
        row["target_24h"]
    )

    return (
        f"harp_{harpnum}_"
        f"{timestamp.strftime('%Y%m%d_%H%M%S')}_"
        f"t{target}.npz"
    )


# ============================================================
# QUERY JSOC
# ============================================================

def query_jsoc_segments(
    client,
    harpnum,
    trec,
):
    """
    Query all three magnetic field segments
    for one observation.
    """

    query = (
        f"{SERIES}"
        f"[{harpnum}]"
        f"[{trec}]"
    )

    try:

        keys, segs = client.query(
            query,
            key="T_REC",
            seg=",".join(SEGMENTS),
        )

        if len(segs) == 0:
            return None

        result = {}

        for component in SEGMENTS:

            if component not in segs.columns:
                return None

            value = segs.iloc[0][
                component
            ]

            if (
                value is None
                or not str(value).strip()
                or str(value).lower()
                == "nan"
            ):

                return None

            result[
                component
            ] = str(value)

        return result

    except Exception as e:

        return None


# ============================================================
# DOWNLOAD ONE FITS
# ============================================================

def download_fits(
    session,
    segment_path,
):
    """
    Download one FITS segment.
    """

    url = (
        JSOC_BASE_URL
        + str(segment_path)
    )

    try:

        response = session.get(
            url,
            timeout=90,
            allow_redirects=True
        )

        if (
            response.status_code == 200
            and len(response.content) > 1000
        ):

            return response.content

    except Exception:
        pass

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(" 3-CHANNEL MAGNETOGRAM DOWNLOAD")
    print("=" * 60)

    print()
    print(
        f"Labels: {LABELS_PATH}"
    )

    print(
        f"Output: {OUTPUT_DIR}"
    )

    # --------------------------------------------------------
    # Load existing labels
    # --------------------------------------------------------

    print()
    print(
        "Loading existing magnetogram labels..."
    )

    if not LABELS_PATH.exists():

        raise FileNotFoundError(
            f"Labels file not found:\n"
            f"{LABELS_PATH}"
        )

    labels = pd.read_csv(
        LABELS_PATH
    )

    labels[
        "observation_time"
    ] = pd.to_datetime(
        labels[
            "observation_time"
        ]
    )

    print(
        f"Total observations: "
        f"{len(labels)}"
    )

    # --------------------------------------------------------
    # Check existing 3-channel files
    # --------------------------------------------------------

    pending = []

    already_done = 0

    for _, row in labels.iterrows():

        filename = (
            build_output_filename(
                row
            )
        )

        output_path = (
            OUTPUT_DIR
            / filename
        )

        if output_path.exists():

            already_done += 1

        else:

            pending.append(
                {
                    "HARPNUM":
                        int(row["HARPNUM"]),

                    "T_REC":
                        row["T_REC"],

                    "target_24h":
                        int(
                            row["target_24h"]
                        ),

                    "observation_time":
                        row[
                            "observation_time"
                        ],

                    "output_path":
                        output_path,

                    "filename":
                        filename,
                }
            )

    print()
    print(
        f"Already downloaded: "
        f"{already_done}"
    )

    print(
        f"Remaining: "
        f"{len(pending)}"
    )

    if len(pending) == 0:

        print()
        print(
            "All 3-channel magnetograms "
            "already exist."
        )

        return

    # --------------------------------------------------------
    # JSOC client
    # --------------------------------------------------------

    print()
    print(
        "Connecting to JSOC..."
    )

    client = drms.Client(
        email="jsoc@sunpy.org"
    )

    session = requests.Session()

    # --------------------------------------------------------
    # Cache JSOC segment paths
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(" QUERYING JSOC")
    print("=" * 60)

    segment_records = []

    query_failures = 0

    for item in tqdm(
        pending,
        desc="Querying JSOC"
    ):

        segment_paths = (
            query_jsoc_segments(
                client,
                item["HARPNUM"],
                item["T_REC"],
            )
        )

        if segment_paths is None:

            query_failures += 1

            continue

        item["segments"] = (
            segment_paths
        )

        segment_records.append(
            item
        )

    print()
    print(
        f"Successful JSOC queries: "
        f"{len(segment_records)}"
    )

    print(
        f"Failed JSOC queries: "
        f"{query_failures}"
    )

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(" DOWNLOADING Br + Bp + Bt")
    print("=" * 60)

    success = already_done

    failures = 0

    for item in tqdm(
        segment_records,
        desc="Downloading 3-channel data"
    ):

        component_bytes = {}

        failed = False

        for component in SEGMENTS:

            segment_path = (
                item["segments"][
                    component
                ]
            )

            content = download_fits(
                session,
                segment_path
            )

            if content is None:

                failed = True

                break

            component_bytes[
                component
            ] = content

            # Small delay between requests
            time.sleep(0.05)

        if failed:

            failures += 1

            continue

        output_path = (
            item["output_path"]
        )

        success_flag = (
            process_three_components(
                component_bytes,
                output_path
            )
        )

        if success_flag:

            success += 1

        else:

            failures += 1

        time.sleep(0.05)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(" 3-CHANNEL DOWNLOAD COMPLETE")
    print("=" * 60)

    print()
    print(
        f"Requested observations : "
        f"{len(labels)}"
    )

    print(
        f"Already existed        : "
        f"{already_done}"
    )

    print(
        f"Successfully processed : "
        f"{success}"
    )

    print(
        f"Failed                  : "
        f"{failures}"
    )

    print()
    print(
        f"Output directory:\n"
        f"{OUTPUT_DIR}"
    )

    # --------------------------------------------------------
    # Verify a sample
    # --------------------------------------------------------

    files = list(
        OUTPUT_DIR.glob(
            "*.npz"
        )
    )

    print()
    print(
        f"3-channel files present: "
        f"{len(files)}"
    )

    if files:

        sample_path = files[0]

        try:

            sample = np.load(
                sample_path
            )

            print()
            print(
                "Sample verification:"
            )

            print(
                f"File: "
                f"{sample_path.name}"
            )

            print(
                f"Keys: "
                f"{sample.files}"
            )

            print(
                f"Shape: "
                f"{sample['img'].shape}"
            )

            print(
                f"Dtype: "
                f"{sample['img'].dtype}"
            )

            print(
                f"Min: "
                f"{sample['img'].min():.4f}"
            )

            print(
                f"Max: "
                f"{sample['img'].max():.4f}"
            )

        except Exception as e:

            print(
                f"Verification failed: {e}"
            )


if __name__ == "__main__":
    main()