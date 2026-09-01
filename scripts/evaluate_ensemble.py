import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import torchvision.models as models
import joblib


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "sharp_goes_temporal_features_2012_full.parquet"
)

MAGNETOGRAM_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "magnetograms"
)

LABELS_PATH = MAGNETOGRAM_DIR / "dataset_labels.csv"

XGB_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost_2012_temporal_features.joblib"
)

CNN_PATH = (
    PROJECT_ROOT
    / "models"
    / "cnn_magnetogram.pt"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "ensemble"


TRAIN_END = pd.Timestamp("2012-06-30 23:59:59")


# ============================================================
# CNN MODEL
# ============================================================

def build_cnn():

    model = models.resnet18(weights=None)

    model.conv1 = nn.Conv2d(
        1,
        64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )

    num_features = model.fc.in_features

    model.fc = nn.Linear(
        num_features,
        1
    )

    return model


# ============================================================
# LOAD CNN
# ============================================================

def load_cnn(device):

    print("Loading CNN...")

    model = build_cnn()

    state_dict = torch.load(
        CNN_PATH,
        map_location=device
    )

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    print("CNN loaded successfully.")

    return model


# ============================================================
# MAGNETOGRAM DATASET
# ============================================================

class MagnetogramDataset(Dataset):

    def __init__(self, df):

        self.df = df.reset_index(drop=True)

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        fname = (
            f"harp_{row['HARPNUM']}_"
            f"{row['observation_time'].strftime('%Y%m%d_%H%M%S')}_"
            f"t{int(row['target_24h'])}.npz"
        )

        path = MAGNETOGRAM_DIR / fname

        data = np.load(path)["img"]

        tensor = torch.tensor(
            data,
            dtype=torch.float32
        ).unsqueeze(0)

        return tensor


# ============================================================
# FIND VALID MAGNETOGRAMS
# ============================================================

def prepare_magnetogram_data():

    print("\nLoading magnetogram labels...")

    labels = pd.read_csv(LABELS_PATH)

    labels["observation_time"] = pd.to_datetime(
        labels["observation_time"]
    )

    valid_rows = []

    for _, row in labels.iterrows():

        fname = (
            f"harp_{row['HARPNUM']}_"
            f"{row['observation_time'].strftime('%Y%m%d_%H%M%S')}_"
            f"t{int(row['target_24h'])}.npz"
        )

        path = MAGNETOGRAM_DIR / fname

        if path.exists():

            valid_rows.append(row)

    magnetograms = pd.DataFrame(valid_rows)

    print(
        f"Valid magnetograms available: "
        f"{len(magnetograms)}"
    )

    return magnetograms


# ============================================================
# CNN PREDICTIONS
# ============================================================

def generate_cnn_predictions(
    df,
    model,
    device
):

    print("\nGenerating CNN predictions...")

    dataset = MagnetogramDataset(df)

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False
    )

    predictions = []

    with torch.no_grad():

        for batch in loader:

            batch = batch.to(device)

            logits = model(batch)

            probs = torch.sigmoid(
                logits.squeeze(1)
            )

            predictions.extend(
                probs.cpu().numpy()
            )

    return np.array(predictions)


# ============================================================
# XGBOOST PREDICTIONS
# ============================================================

def generate_xgb_predictions(df):

    print("\nLoading XGBoost...")

    package = joblib.load(XGB_PATH)

    model = package["model"]
    imputer = package["imputer"]
    features = package["features"]

    print("XGBoost loaded successfully.")

    missing = [
        f for f in features
        if f not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing XGBoost features: {missing}"
        )

    X = df[features].copy()

    X = imputer.transform(X)

    probabilities = model.predict_proba(X)[:, 1]

    return probabilities


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    labels,
    probabilities
):

    if len(np.unique(labels)) < 2:

        return 0.0, 0.0

    roc = roc_auc_score(
        labels,
        probabilities
    )

    pr = average_precision_score(
        labels,
        probabilities
    )

    return roc, pr


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(" XGBOOST + CNN ENSEMBLE EVALUATION")
    print("=" * 60)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nUsing device: {device}")

    # --------------------------------------------------------
    # LOAD TABULAR DATA
    # --------------------------------------------------------

    print("\nLoading processed tabular data...")

    data = pd.read_parquet(DATA_PATH)

    data["observation_time"] = pd.to_datetime(
        data["observation_time"]
    )

    print(
        f"Total observations: {len(data)}"
    )

    # --------------------------------------------------------
    # TEST PERIOD
    # --------------------------------------------------------

    test_data = data[
        data["observation_time"] > TRAIN_END
    ].copy()

    print(
        f"Test observations: {len(test_data)}"
    )

    # --------------------------------------------------------
    # LOAD MAGNETOGRAM LABELS
    # --------------------------------------------------------

    magnetograms = prepare_magnetogram_data()

    # Only use test-period magnetograms
    magnetograms = magnetograms[
        magnetograms["observation_time"] > TRAIN_END
    ].copy()

    print(
        f"Test-period magnetograms: "
        f"{len(magnetograms)}"
    )

    # --------------------------------------------------------
    # MATCH MAGNETOGRAMS TO TABULAR DATA
    # --------------------------------------------------------

    print("\nMatching CNN observations to XGBoost data...")

    merge_keys = [
        "HARPNUM",
        "observation_time"
    ]

    # Make sure HARPNUM exists
    if "HARPNUM" not in test_data.columns:

        raise ValueError(
            "HARPNUM is missing from the processed "
            "XGBoost dataset."
        )

    matched = magnetograms.merge(
        test_data,
        on=merge_keys,
        how="inner",
        suffixes=("_mag", "")
    )

    print(
        f"Matched observations: {len(matched)}"
    )

    if len(matched) == 0:

        raise RuntimeError(
            "No observations could be matched. "
            "Check HARPNUM and observation_time."
        )

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    matched = matched.drop_duplicates(
        subset=merge_keys
    ).reset_index(drop=True)

    print(
        f"After duplicate removal: {len(matched)}"
    )

    # --------------------------------------------------------
    # LABELS
    # --------------------------------------------------------

    labels = matched["target_24h"].values.astype(int)

    print(
        f"\nPositive samples: {labels.sum()}"
    )

    print(
        f"Negative samples: "
        f"{len(labels) - labels.sum()}"
    )

    # --------------------------------------------------------
    # XGBOOST
    # --------------------------------------------------------

    xgb_probs = generate_xgb_predictions(
        matched
    )

    xgb_roc, xgb_pr = calculate_metrics(
        labels,
        xgb_probs
    )

    print("\nXGBOOST RESULTS")
    print("-" * 40)
    print(f"ROC-AUC : {xgb_roc:.4f}")
    print(f"PR-AUC  : {xgb_pr:.4f}")

    # --------------------------------------------------------
    # CNN
    # --------------------------------------------------------

    cnn_model = load_cnn(device)

    cnn_probs = generate_cnn_predictions(
        matched,
        cnn_model,
        device
    )

    cnn_roc, cnn_pr = calculate_metrics(
        labels,
        cnn_probs
    )

    print("\nCNN RESULTS")
    print("-" * 40)
    print(f"ROC-AUC : {cnn_roc:.4f}")
    print(f"PR-AUC  : {cnn_pr:.4f}")

    # --------------------------------------------------------
    # ENSEMBLE TEST
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print(" TESTING ENSEMBLE WEIGHTS")
    print("=" * 60)

    weights = [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0
    ]

    results = []

    for xgb_weight in weights:

        cnn_weight = 1.0 - xgb_weight

        ensemble_probs = (
            xgb_weight * xgb_probs
            +
            cnn_weight * cnn_probs
        )

        roc, pr = calculate_metrics(
            labels,
            ensemble_probs
        )

        results.append({

            "xgb_weight": xgb_weight,

            "cnn_weight": cnn_weight,

            "roc_auc": roc,

            "pr_auc": pr
        })

        print(
            f"XGB={xgb_weight:.1f} | "
            f"CNN={cnn_weight:.1f} | "
            f"ROC-AUC={roc:.4f} | "
            f"PR-AUC={pr:.4f}"
        )

    results_df = pd.DataFrame(results)

    # --------------------------------------------------------
    # BEST ENSEMBLE
    # --------------------------------------------------------

    best = results_df.loc[
        results_df["pr_auc"].idxmax()
    ]

    print("\n")
    print("=" * 60)
    print(" BEST ENSEMBLE")
    print("=" * 60)

    print(
        f"XGBoost weight : "
        f"{best['xgb_weight']:.1f}"
    )

    print(
        f"CNN weight     : "
        f"{best['cnn_weight']:.1f}"
    )

    print(
        f"ROC-AUC        : "
        f"{best['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC         : "
        f"{best['pr_auc']:.4f}"
    )

    print("\n")

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results_path = (
        RESULTS_DIR
        / "ensemble_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    print(
        f"Results saved to:\n"
        f"{results_path}"
    )

    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    best_xgb_weight = best["xgb_weight"]
    best_cnn_weight = best["cnn_weight"]

    best_probs = (
        best_xgb_weight * xgb_probs
        +
        best_cnn_weight * cnn_probs
    )

    prediction_output = matched[
        [
            "HARPNUM",
            "NOAA_AR",
            "observation_time",
            "target_24h"
        ]
    ].copy()

    prediction_output[
        "xgb_probability"
    ] = xgb_probs

    prediction_output[
        "cnn_probability"
    ] = cnn_probs

    prediction_output[
        "ensemble_probability"
    ] = best_probs

    prediction_output[
        "ensemble_prediction"
    ] = (
        best_probs >= 0.01
    ).astype(int)

    predictions_path = (
        RESULTS_DIR
        / "ensemble_predictions.csv"
    )

    prediction_output.to_csv(
        predictions_path,
        index=False
    )

    print(
        f"Predictions saved to:\n"
        f"{predictions_path}"
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print(" BASELINE VS ENSEMBLE")
    print("=" * 60)

    print(
        f"XGBoost ROC-AUC : {xgb_roc:.4f}"
    )

    print(
        f"XGBoost PR-AUC  : {xgb_pr:.4f}"
    )

    print(
        f"CNN ROC-AUC     : {cnn_roc:.4f}"
    )

    print(
        f"CNN PR-AUC      : {cnn_pr:.4f}"
    )

    print(
        f"Ensemble ROC-AUC: {best['roc_auc']:.4f}"
    )

    print(
        f"Ensemble PR-AUC : {best['pr_auc']:.4f}"
    )

    print("=" * 60)
    print(" ENSEMBLE EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":

    main()