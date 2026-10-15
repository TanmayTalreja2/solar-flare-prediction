import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st

st.title("About")

st.markdown(
    """
## What this project is

A research prototype that estimates the risk of an M- or X-class solar flare in an active region
within the next 24 hours, from magnetic-field measurements and magnetogram images of the Sun.

## Data

- **Magnetic measurements and images:** SDO/HMI SHARP data, 12-minute cadence, from the JSOC at Stanford. Models were trained on 2012. Of the 25 standard SHARP parameters, 17 were available from the queried series and are used.
- **Flare labels:** GOES M- and X-class flare start times, matched to active regions. An observation is labelled positive if a strong flare starts in the same region within 24 hours.
- **Live Mode:** the near-real-time SHARP series from JSOC.

## Models

- **XGBoost** on SHARP parameters plus how they changed over the preceding 1, 3, 6 and 12 hours.
- **ResNet18 CNN** on magnetogram images.
- **Fusion:** a logistic regression combining the two outputs.

## Reports

The Predictor page generates a plain-language PDF for general readers and a technical PDF
(plus a JSON export) for researchers. Every number in them comes from the live model and evaluation files.

## Honest limitations

- Trained and tested on a single year (2012); results may not carry over to other phases of the solar cycle.
- Scores are relative warning levels, not calibrated probabilities.
- Precision and recall are low: most alerts are false alarms and many flares are missed.
- Live scores have not been validated against outcomes.
- Not validated for operational use.

## Acknowledgements

Data courtesy of NASA/SDO and the HMI science team (JSOC, Stanford University) and NOAA's
Space Weather Prediction Center (GOES).
"""
)