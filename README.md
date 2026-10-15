# 🌞 Hybrid Solar Flare Prediction System

A hybrid machine learning system for predicting solar flare activity using both **solar active-region features** and **solar magnetogram images**.

The project combines:

- XGBoost for temporal and tabular solar features
- ResNet18 CNN for solar magnetogram images
- A weighted hybrid ensemble
- A real-time prediction pipeline
- Human-readable solar flare risk classification

---

# 📌 Project Overview

Solar flares are sudden releases of energy from the Sun that can affect space weather, satellite systems, communication systems, and other technological infrastructure.

This project develops a hybrid prediction system that analyzes two different types of solar observations:

1. Numerical active-region features
2. Magnetogram images of solar active regions

The outputs of two machine learning models are combined to produce a final solar flare probability and risk level.

## Hybrid Architecture

```text
                         SOLAR OBSERVATION
                                │
                 ┌──────────────┴──────────────┐
                 │                             │
                 ▼                             ▼
        SHARP / Temporal Features        Magnetogram Image
                 │                             │
                 ▼                             ▼
              XGBoost                      ResNet18 CNN
                 │                             │
                 ▼                             ▼
        Tabular Probability           Image Probability
                 │                             │
                 └──────────────┬──────────────┘
                                │
                                ▼
                         HYBRID ENSEMBLE
                                │
                                ▼
                    SOLAR FLARE RISK PREDICTION


## Running the dashboard

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run dashboard\app.py
```

Pages: Home, Predictor (with plain-language and technical PDF reports), Live Mode, Model Insights, RL Experiment, About.

## Restoring the gitignored models and data

`models/` and `data/` are excluded from git. The dashboard needs these files in place:

- `models/xgboost_2012_temporal_features_25params.joblib`
- `models/cnn_magnetogram.pt` (needed for Live Mode and the fusion scripts)
- `data/processed/features/sharp_goes_temporal_features_2012_full_25params.parquet`
- `results/cnn/final_metrics.csv`
- `results/ensemble_25params/` (`fusion_results.csv`, `fusion_predictions.csv`, `fusion_model.joblib`)
- `results/xgboost_25params/hyperparameter_results.csv`

To regenerate the tabular data and model from scratch, run in order:

```powershell
python scripts\download_sharp_full_params.py
python scripts\combine_sharp_2012_full_params.py
python scripts\align_sharp_goes.py
python scripts\clean_full_dataset.py
python scripts\engineer_features_full.py
python scripts\engineer_temporal_features.py
python scripts\train_xgboost_temporal_features_25params.py
python scripts\train_fusion_25params.py
python scripts\rl_dry_run.py
```

Check the input and output filenames inside the align, clean and feature scripts: they were edited to use the `_25params` files. The CNN needs the magnetogram image dataset from `scripts\download_magnetograms.py`.

## Live Mode

Live Mode queries JSOC's near-real-time SHARP series (`hmi.sharp_cea_720s_nrt`) and needs no registration for the queries it uses. It needs `astropy` for the FITS images. Each fetch is logged to `data/live_log/live_predictions.csv`; `scripts\verify_live_log.py` compares logged predictions against observed flares once 24 hours have passed.

## Tests

```powershell
pytest -q
```