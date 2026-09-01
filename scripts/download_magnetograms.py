"""
Download and preprocess magnetogram images from JSOC.

Strategy:
  1. Sample 1000 positive + 1000 negative observations from the dataset.
  2. Batch-query JSOC for segment file paths (fast, no export needed).
  3. Download FITS files directly via HTTP.
  4. Normalize, crop/pad to 224x224, save as compressed .npz.
  5. No raw FITS files are kept on disk.

This approach is much faster than the JSOC export system because
it skips the server-side export queue entirely.
"""

import os
import io
import pandas as pd
import numpy as np
import drms
import requests
from pathlib import Path
from tqdm import tqdm
from astropy.io import fits
import warnings
import time

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

project_root = Path(__file__).parent.parent
data_path = (
    project_root / "data" / "processed" / "features"
    / "sharp_goes_temporal_features_2012_full.parquet"
)
out_dir = project_root / "data" / "processed" / "magnetograms"
out_dir.mkdir(parents=True, exist_ok=True)

JSOC_BASE_URL = "http://jsoc.stanford.edu"
SERIES = "hmi.sharp_cea_720s"
SEGMENT = "Br"  # Radial magnetic field component


# ============================================================
# IMAGE PROCESSING
# ============================================================

def process_fits_bytes(fits_bytes, out_npz_path, target_size=(224, 224)):
    """
    Read FITS from bytes, normalize, crop/pad to target_size,
    and save as compressed .npz.
    """
    try:
        hdul = fits.open(io.BytesIO(fits_bytes))
        data = hdul[1].data if len(hdul) > 1 else hdul[0].data
        hdul.close()

        if data is None:
            return False

        # Handle NaN / Inf
        data = np.nan_to_num(
            data.astype(np.float32),
            nan=0.0, posinf=0.0, neginf=0.0,
        )

        # Normalize: clip to ±3000 Gauss, scale to [-1, 1]
        data = np.clip(data, -3000, 3000)
        data = data / 3000.0

        # Crop / pad to target_size
        h, w = data.shape
        th, tw = target_size

        if h < th or w < tw:
            pad_h = max(0, th - h)
            pad_w = max(0, tw - w)
            data = np.pad(
                data,
                ((pad_h // 2, pad_h - pad_h // 2),
                 (pad_w // 2, pad_w - pad_w // 2)),
                mode="constant",
            )
            h, w = data.shape

        y_start = (h - th) // 2
        x_start = (w - tw) // 2
        cropped = data[y_start:y_start + th, x_start:x_start + tw]

        np.savez_compressed(out_npz_path, img=cropped)
        return True

    except Exception as e:
        print(f"  Error processing FITS: {e}")
        return False


# ============================================================
# BATCH QUERY JSOC FOR SEGMENT PATHS
# ============================================================

def query_segment_paths(client, harpnums, t_recs):
    """
    Query JSOC for segment file paths in batches.
    Returns a dict mapping (harpnum, t_rec) -> segment_path.
    """
    # Group by HARPNUM for efficient batch queries
    from collections import defaultdict
    harp_to_trecs = defaultdict(list)
    for harp, trec in zip(harpnums, t_recs):
        harp_to_trecs[harp].append(trec)

    seg_map = {}
    unique_harps = list(harp_to_trecs.keys())

    print(f"Querying JSOC for segment paths across {len(unique_harps)} unique HARPNUMs...")

    for harp in tqdm(unique_harps, desc="Querying JSOC"):
        trecs = harp_to_trecs[harp]

        # Build a query that fetches all needed timestamps for this HARPNUM
        # We query one at a time since each T_REC is a specific timestamp
        for trec in trecs:
            query = f"{SERIES}[{harp}][{trec}]"
            try:
                keys, segs = client.query(query, key="T_REC", seg=SEGMENT)
                if len(segs) > 0 and SEGMENT in segs.columns:
                    seg_path = segs.iloc[0][SEGMENT]
                    if seg_path and str(seg_path).strip():
                        seg_map[(harp, trec)] = seg_path
            except Exception as e:
                pass  # Skip failed queries silently

    return seg_map


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    df["observation_time"] = pd.to_datetime(df["observation_time"])

    # --------------------------------------------------------
    # Subsample: 1000 positive + 1000 negative
    # --------------------------------------------------------
    pos_samples = df[df["target_24h"] == 1].sample(n=2500, random_state=42)
    neg_samples = df[df["target_24h"] == 0].sample(n=2500, random_state=42)
    samples = (
        pd.concat([pos_samples, neg_samples])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    print(
        f"Selected {len(samples)} total samples "
        f"({len(pos_samples)} positive, {len(neg_samples)} negative)"
    )

    # Save labels
    labels_path = out_dir / "dataset_labels.csv"
    samples[
        ["HARPNUM", "NOAA_AR", "observation_time", "T_REC", "target_24h"]
    ].to_csv(labels_path, index=False)

    # --------------------------------------------------------
    # Check which files already exist
    # --------------------------------------------------------
    to_download = []
    already_done = 0

    for idx, row in samples.iterrows():
        harpnum = int(row["HARPNUM"])
        ts = row["observation_time"]
        target = int(row["target_24h"])
        t_rec = row["T_REC"]

        out_npz_name = (
            f"harp_{harpnum}_"
            f"{ts.strftime('%Y%m%d_%H%M%S')}_"
            f"t{target}.npz"
        )
        out_npz_path = out_dir / out_npz_name

        if out_npz_path.exists():
            already_done += 1
        else:
            to_download.append({
                "harpnum": harpnum,
                "t_rec": t_rec,
                "target": target,
                "ts": ts,
                "out_path": out_npz_path,
            })

    print(f"Already downloaded: {already_done}")
    print(f"Remaining to download: {len(to_download)}")

    if len(to_download) == 0:
        print("All files already downloaded!")
        return

    # --------------------------------------------------------
    # Query JSOC for segment paths
    # --------------------------------------------------------
    client = drms.Client(email="jsoc@sunpy.org")

    harpnums = [d["harpnum"] for d in to_download]
    t_recs = [d["t_rec"] for d in to_download]

    seg_map = query_segment_paths(client, harpnums, t_recs)

    print(f"Found segment paths for {len(seg_map)}/{len(to_download)} records.")

    # --------------------------------------------------------
    # Download and process
    # --------------------------------------------------------
    success_count = already_done
    fail_count = 0
    session = requests.Session()

    for item in tqdm(to_download, desc="Downloading magnetograms"):
        key = (item["harpnum"], item["t_rec"])

        if key not in seg_map:
            fail_count += 1
            continue

        seg_path = seg_map[key]
        url = JSOC_BASE_URL + seg_path

        try:
            resp = session.get(url, timeout=60, allow_redirects=True)

            if resp.status_code == 200 and len(resp.content) > 1000:
                if process_fits_bytes(resp.content, item["out_path"]):
                    success_count += 1
                else:
                    fail_count += 1
            else:
                fail_count += 1

        except Exception as e:
            fail_count += 1

        # Small delay to be polite to JSOC servers
        time.sleep(0.1)

    print(f"\n{'='*50}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'='*50}")
    print(f"Successfully processed: {success_count}/{len(samples)}")
    print(f"Failed: {fail_count}")
    print(f"Data saved to: {out_dir}")


if __name__ == "__main__":
    main()
