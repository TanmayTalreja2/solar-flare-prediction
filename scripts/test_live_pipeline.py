import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from solarflare.realtime.live_pipeline import run_live_predictions

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 140)
pd.set_option("display.max_columns", 20)

if __name__ == "__main__":

    print("Fetching live predictions (this can take a minute or two)...")

    results, error = run_live_predictions()

    if error:
        print(f"ERROR: {error}")
        sys.exit(1)

    if len(results) == 0:
        print("No active regions returned.")
        sys.exit(0)

    print(f"\n{len(results)} active regions:\n")
    print(results.drop(columns=["note"]).to_string(index=False))

    print("\nNotes:")
    for _, r in results.iterrows():
        print(f"  NOAA {r['NOAA_AR']}: {r['note'] or 'ok'}")