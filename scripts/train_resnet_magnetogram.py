from pathlib import Path
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MAGNETOGRAM_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "magnetograms"
)

LABELS_PATH = (
    MAGNETOGRAM_DIR
    / "dataset_labels_with_paths.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "resnet_magnetogram.pt"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "resnet"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

RANDOM_SEED = 42

BATCH_SIZE = 16

EPOCHS = 15

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

PATIENCE = 5

NUM_WORKERS = 0


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=RANDOM_SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# DATASET
# ============================================================

class MagnetogramDataset(Dataset):

    def __init__(
        self,
        dataframe,
        transform=None,
    ):

        self.dataframe = (
            dataframe
            .reset_index(drop=True)
            .copy()
        )

        self.transform = transform


    def __len__(self):

        return len(
            self.dataframe
        )


    def __getitem__(
        self,
        index,
    ):

        row = self.dataframe.iloc[index]

        file_path = Path(
            str(row["file_path"])
        )


        # ====================================================
        # LOAD MAGNETOGRAM
        # ====================================================

        with np.load(
            file_path
        ) as data:

            magnetogram = data[
                "img"
            ]


        magnetogram = np.asarray(
            magnetogram,
            dtype=np.float32,
        )


        # ====================================================
        # REMOVE NAN / INF
        # ====================================================

        magnetogram = np.nan_to_num(
            magnetogram,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )


        # ====================================================
        # NORMALIZATION
        # ====================================================

        mean = float(
            np.mean(
                magnetogram
            )
        )

        std = float(
            np.std(
                magnetogram
            )
        )


        if (
            not np.isfinite(std)
            or std < 1e-6
        ):

            std = 1.0


        magnetogram = (
            magnetogram - mean
        ) / std


        magnetogram = np.nan_to_num(
            magnetogram,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )


        # ====================================================
        # CLIP EXTREME VALUES
        # ====================================================

        magnetogram = np.clip(
            magnetogram,
            -10.0,
            10.0,
        )


        # ====================================================
        # ENSURE 224 x 224
        # ====================================================

        image = torch.tensor(
            magnetogram,
            dtype=torch.float32,
        )


        # ====================================================
        # CONVERT TO 3 CHANNELS
        # ====================================================

        if image.ndim == 2:

            image = image.unsqueeze(0)

        if image.shape[0] == 1:

            image = image.repeat(
                3,
                1,
                1,
            )


        # ====================================================
        # TRANSFORM
        # ====================================================

        if self.transform is not None:

            image = self.transform(
                image
            )


        image = torch.nan_to_num(
            image,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )


        # ====================================================
        # LABEL
        # ====================================================

        label = torch.tensor(
            float(
                row["target"]
            ),
            dtype=torch.float32,
        )


        return (
            image,
            label,
        )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print(
        "Loading labels..."
    )


    if not LABELS_PATH.exists():

        raise FileNotFoundError(

            "Corrected labels file not found:\n"

            f"{LABELS_PATH}"

        )


    data = pd.read_csv(
        LABELS_PATH
    )


    print(
        f"Total label rows: {len(data)}"
    )

    print()

    print(
        "Using corrected magnetogram paths."
    )


    # ========================================================
    # TARGET
    # ========================================================

    if "target_24h" in data.columns:

        data = data.rename(
            columns={
                "target_24h": "target"
            }
        )


    if "target" not in data.columns:

        raise ValueError(
            "Labels file must contain "
            "'target_24h' or 'target'."
        )


    # ========================================================
    # REQUIRED COLUMNS
    # ========================================================

    required_columns = [

        "HARPNUM",

        "observation_time",

        "target",

        "file_path",

    ]


    for column in required_columns:

        if column not in data.columns:

            raise ValueError(

                f"Missing required column: "
                f"{column}"

            )


    # ========================================================
    # TIME
    # ========================================================

    data[
        "observation_time"
    ] = pd.to_datetime(

        data[
            "observation_time"
        ],

        errors="coerce",

    )


    # ========================================================
    # REMOVE INVALID ROWS
    # ========================================================

    data = data.dropna(

        subset=[

            "HARPNUM",

            "observation_time",

            "target",

            "file_path",

        ]

    ).copy()


    # ========================================================
    # TARGET CLEANING
    # ========================================================

    data[
        "target"
    ] = data[
        "target"
    ].astype(int)


    data = data[

        data[
            "target"
        ].isin(
            [0, 1]
        )

    ].copy()


    # ========================================================
    # FIX FILE PATHS
    #
    # Handles:
    #
    # filename.npz
    #
    # relative/path/filename.npz
    #
    # incorrect/old/absolute/path/filename.npz
    #
    # and rebuilds everything using MAGNETOGRAM_DIR.
    # ========================================================

    def resolve_file_path(
        path_value,
    ):

        raw_path = Path(
            str(path_value)
        )


        # Only the filename matters because
        # all NPZ files are in MAGNETOGRAM_DIR.

        filename = raw_path.name


        corrected_path = (

            MAGNETOGRAM_DIR

            / filename

        )


        return str(
            corrected_path.resolve()
        )


    data[
        "file_path"
    ] = data[
        "file_path"
    ].apply(
        resolve_file_path
    )


    # ========================================================
    # SORT CHRONOLOGICALLY
    # ========================================================

    data = (

        data

        .sort_values(
            "observation_time"
        )

        .reset_index(
            drop=True
        )

    )


    # ========================================================
    # PRINT DISTRIBUTION
    # ========================================================

    positives = int(

        (
            data["target"] == 1
        ).sum()

    )


    negatives = int(

        (
            data["target"] == 0
        ).sum()

    )


    print(
        f"Positive samples: {positives}"
    )

    print(
        f"Negative samples: {negatives}"
    )


    # ========================================================
    # PATH DEBUG
    # ========================================================

    print()

    print(
        "Sample resolved paths:"
    )


    for path in data[
        "file_path"
    ].head(3):

        print(
            f"  {path}"
        )


    return data


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def chronological_split(
    data,
):

    train = data[

        data[
            "observation_time"
        ]

        < pd.Timestamp(
            "2012-05-01"
        )

    ].copy()


    validation = data[

        (

            data[
                "observation_time"
            ]

            >= pd.Timestamp(
                "2012-05-01"
            )

        )

        &

        (

            data[
                "observation_time"
            ]

            < pd.Timestamp(
                "2012-07-01"
            )

        )

    ].copy()


    test = data[

        data[
            "observation_time"
        ]

        >= pd.Timestamp(
            "2012-07-01"
        )

    ].copy()


    print()

    print(
        "========== CHRONOLOGICAL SPLIT =========="
    )


    for name, dataframe in [

        ("TRAIN", train),

        ("VALIDATION", validation),

        ("FINAL TEST", test),

    ]:

        print()

        print(
            f"{name}:"
        )

        print(
            f"  Rows: {len(dataframe)}"
        )


        if len(dataframe) > 0:

            print(

                f"  Period: "

                f"{dataframe['observation_time'].min()} "

                f"-> "

                f"{dataframe['observation_time'].max()}"

            )


    return (
        train,
        validation,
        test,
    )


# ============================================================
# VALIDATE FILES
# ============================================================

def validate_files(
    dataframe,
):

    valid_rows = []

    missing_paths = []

    invalid_files = []


    for _, row in dataframe.iterrows():

        file_path = Path(
            str(
                row[
                    "file_path"
                ]
            )
        )


        # ====================================================
        # FIX PATH ONE MORE TIME
        # ====================================================

        file_path = (
            MAGNETOGRAM_DIR
            / file_path.name
        )


        # ====================================================
        # EXISTS
        # ====================================================

        if not file_path.exists():

            if len(missing_paths) < 5:

                missing_paths.append(
                    str(file_path)
                )

            continue


        # ====================================================
        # CHECK NPZ
        # ====================================================

        try:

            with np.load(
                file_path
            ) as data:

                if "img" not in data:

                    if len(invalid_files) < 5:

                        invalid_files.append(

                            (
                                str(file_path),
                                "Missing 'img' key",
                            )

                        )

                    continue


                image = np.asarray(
                    data["img"],
                    dtype=np.float32,
                )


                if image.size == 0:

                    if len(invalid_files) < 5:

                        invalid_files.append(

                            (
                                str(file_path),
                                "Empty image",
                            )

                        )

                    continue


                row = row.copy()

                row[
                    "file_path"
                ] = str(
                    file_path.resolve()
                )


                valid_rows.append(
                    row
                )


        except Exception as error:

            if len(invalid_files) < 5:

                invalid_files.append(

                    (
                        str(file_path),
                        str(error),
                    )

                )


    valid_dataframe = (

        pd.DataFrame(
            valid_rows
        )

        .reset_index(
            drop=True
        )

    )


    print(
        f"Found {len(valid_dataframe)} valid .npz files."
    )


    if missing_paths:

        print()

        print(
            "Sample missing paths:"
        )

        for path in missing_paths:

            print(
                f"  {path}"
            )


    if invalid_files:

        print()

        print(
            "Sample invalid files:"
        )

        for path, reason in invalid_files:

            print(
                f"  {path}"
            )

            print(
                f"    Reason: {reason}"
            )


    return valid_dataframe


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def print_distribution(
    name,
    dataframe,
):

    positives = int(
        (
            dataframe[
                "target"
            ] == 1
        ).sum()
    )


    negatives = int(
        (
            dataframe[
                "target"
            ] == 0
        ).sum()
    )


    print(

        f"{name:10s} -> "

        f"Positive: {positives}, "

        f"Negative: {negatives}"

    )


# ============================================================
# MODEL
# ============================================================

def create_model():

    print(
        "Loading pretrained ResNet18..."
    )


    model = models.resnet18(

        weights=(
            models.ResNet18_Weights.DEFAULT
        )

    )


    input_features = (
        model.fc.in_features
    )


    model.fc = nn.Sequential(

        nn.Dropout(
            p=0.4
        ),

        nn.Linear(
            input_features,
            1,
        ),

    )


    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    loader,
    criterion,
):

    model.eval()

    total_loss = 0.0

    total_samples = 0

    all_labels = []

    all_probs = []


    with torch.no_grad():

        for inputs, labels in loader:

            inputs = inputs.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )


            inputs = torch.nan_to_num(
                inputs,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )


            outputs = model(
                inputs
            ).view(-1)


            outputs = torch.nan_to_num(
                outputs,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            )


            loss = criterion(
                outputs,
                labels,
            )


            total_loss += (

                loss.item()

                * inputs.size(0)

            )


            probabilities = torch.sigmoid(
                outputs
            )


            probabilities = torch.nan_to_num(
                probabilities,
                nan=0.5,
                posinf=1.0,
                neginf=0.0,
            )


            all_probs.extend(

                probabilities
                .cpu()
                .numpy()

            )


            all_labels.extend(

                labels
                .cpu()
                .numpy()

            )


            total_samples += (
                inputs.size(0)
            )


    all_labels = np.asarray(
        all_labels,
        dtype=np.int64,
    )


    all_probs = np.asarray(
        all_probs,
        dtype=np.float64,
    )


    all_probs = np.nan_to_num(
        all_probs,
        nan=0.5,
        posinf=1.0,
        neginf=0.0,
    )


    all_probs = np.clip(
        all_probs,
        0.0,
        1.0,
    )


    if len(
        np.unique(
            all_labels
        )
    ) > 1:

        roc_auc = roc_auc_score(
            all_labels,
            all_probs,
        )


        pr_auc = average_precision_score(
            all_labels,
            all_probs,
        )

    else:

        roc_auc = 0.0

        pr_auc = 0.0


    average_loss = (

        total_loss

        / max(
            total_samples,
            1,
        )

    )


    return (

        average_loss,

        roc_auc,

        pr_auc,

        all_labels,

        all_probs,

    )


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def find_best_threshold(
    actuals,
    probabilities,
):

    thresholds = np.arange(
        0.05,
        0.96,
        0.01,
    )


    best_threshold = 0.5

    best_f1 = -1.0

    best_precision = 0.0

    best_recall = 0.0


    for threshold in thresholds:

        predictions = (

            probabilities
            >= threshold

        ).astype(int)


        precision, recall, f1, _ = (

            precision_recall_fscore_support(

                actuals,

                predictions,

                average="binary",

                zero_division=0,

            )

        )


        if f1 > best_f1:

            best_f1 = f1

            best_threshold = threshold

            best_precision = precision

            best_recall = recall


    return (

        best_threshold,

        best_precision,

        best_recall,

        best_f1,

    )


# ============================================================
# TRAINING
# ============================================================

def train_model(

    model,

    train_loader,

    validation_loader,

    criterion,

    optimizer,

    scheduler,

):

    best_pr_auc = -1.0

    best_epoch = 0

    patience_counter = 0

    history = []


    print()

    print(
        "=" * 60
    )

    print(
        " STARTING RESNET18 TRAINING"
    )

    print(
        "=" * 60
    )


    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        running_loss = 0.0

        samples_seen = 0


        for inputs, labels in train_loader:

            inputs = inputs.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )


            inputs = torch.nan_to_num(
                inputs,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )


            optimizer.zero_grad()


            outputs = model(
                inputs
            ).view(-1)


            outputs = torch.nan_to_num(
                outputs,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            )


            loss = criterion(
                outputs,
                labels,
            )


            if not torch.isfinite(loss):

                continue


            loss.backward()


            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )


            optimizer.step()


            running_loss += (

                loss.item()

                * inputs.size(0)

            )


            samples_seen += (
                inputs.size(0)
            )


        train_loss = (

            running_loss

            / max(
                samples_seen,
                1,
            )

        )


        (

            validation_loss,

            validation_roc,

            validation_pr,

            _,

            _,

        ) = evaluate_model(

            model,

            validation_loader,

            criterion,

        )


        scheduler.step(
            validation_pr
        )


        current_lr = (

            optimizer
            .param_groups[0]
            ["lr"]

        )


        print()

        print(

            f"Epoch {epoch}/{EPOCHS} "

            f"- Train Loss: {train_loss:.4f} "

            f"- Val Loss: {validation_loss:.4f} "

            f"- Val ROC-AUC: {validation_roc:.4f} "

            f"- Val PR-AUC: {validation_pr:.4f} "

            f"- LR: {current_lr:.6f}"

        )


        history.append({

            "epoch": epoch,

            "train_loss": train_loss,

            "validation_loss": validation_loss,

            "validation_roc_auc": validation_roc,

            "validation_pr_auc": validation_pr,

            "learning_rate": current_lr,

        })


        if validation_pr > best_pr_auc:

            best_pr_auc = validation_pr

            best_epoch = epoch

            patience_counter = 0


            torch.save(

                {

                    "epoch": epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "validation_pr_auc":
                        validation_pr,

                    "validation_roc_auc":
                        validation_roc,

                },

                MODEL_PATH,

            )


            print(

                f"  -> New best model! "

                f"PR-AUC: {validation_pr:.4f}"

            )


        else:

            patience_counter += 1


            print(

                f"  No improvement "

                f"({patience_counter}/{PATIENCE})"

            )


            if patience_counter >= PATIENCE:

                print()

                print(
                    "Early stopping triggered."
                )

                break


    history_dataframe = pd.DataFrame(
        history
    )


    history_dataframe.to_csv(

        RESULTS_DIR
        / "training_metrics.csv",

        index=False,

    )


    return (
        best_epoch,
        best_pr_auc,
    )


# ============================================================
# PLOTS
# ============================================================

def create_plots():

    metrics_path = (
        RESULTS_DIR
        / "training_metrics.csv"
    )


    if not metrics_path.exists():

        return


    history = pd.read_csv(
        metrics_path
    )


    # ========================================================
    # LOSS PLOT
    # ========================================================

    plt.figure()


    plt.plot(
        history["epoch"],
        history["train_loss"],
        label="Train Loss",
    )


    plt.plot(
        history["epoch"],
        history["validation_loss"],
        label="Validation Loss",
    )


    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "ResNet18 Training Loss"
    )

    plt.legend()

    plt.tight_layout()


    plt.savefig(

        RESULTS_DIR
        / "training_loss.png"

    )


    plt.close()


    # ========================================================
    # METRICS PLOT
    # ========================================================

    plt.figure()


    plt.plot(
        history["epoch"],
        history["validation_roc_auc"],
        label="ROC-AUC",
    )


    plt.plot(
        history["epoch"],
        history["validation_pr_auc"],
        label="PR-AUC",
    )


    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "ResNet18 Validation Metrics"
    )

    plt.legend()

    plt.tight_layout()


    plt.savefig(

        RESULTS_DIR
        / "validation_metrics.png"

    )


    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print(
        "=" * 60
    )

    print(
        " RESNET18 MAGNETOGRAM TRAINING"
    )

    print(
        "=" * 60
    )


    # ========================================================
    # SEED
    # ========================================================

    set_seed()


    print()

    print(
        f"Using device: {DEVICE}"
    )


    # ========================================================
    # LOAD DATA
    # ========================================================

    data = load_data()


    # ========================================================
    # SPLIT
    # ========================================================

    train_data, validation_data, test_data = (

        chronological_split(
            data
        )

    )


    # ========================================================
    # VALIDATE FILES
    # ========================================================

    print()

    print(
        "========== CHECKING MAGNETOGRAM FILES =========="
    )


    train_data = validate_files(
        train_data
    )


    validation_data = validate_files(
        validation_data
    )


    test_data = validate_files(
        test_data
    )


    print()

    print(
        "========== VALID FILE COUNTS =========="
    )


    print(
        f"Train:      {len(train_data)}"
    )

    print(
        f"Validation: {len(validation_data)}"
    )

    print(
        f"Final test: {len(test_data)}"
    )


    if (

        len(train_data) == 0

        or

        len(validation_data) == 0

        or

        len(test_data) == 0

    ):

        raise ValueError(
            "One or more datasets are empty."
        )


    # ========================================================
    # CLASS DISTRIBUTION
    # ========================================================

    print()

    print(
        "========== CLASS DISTRIBUTION =========="
    )


    print_distribution(
        "TRAIN",
        train_data,
    )


    print_distribution(
        "VALIDATION",
        validation_data,
    )


    print_distribution(
        "FINAL TEST",
        test_data,
    )


    # ========================================================
    # CHECK BOTH CLASSES
    # ========================================================

    for name, dataframe in [

        ("TRAIN", train_data),

        ("VALIDATION", validation_data),

        ("FINAL TEST", test_data),

    ]:

        unique_classes = set(
            dataframe[
                "target"
            ].unique()
        )


        if unique_classes != {0, 1}:

            raise ValueError(

                f"{name} does not contain "

                f"both classes. "

                f"Found: {unique_classes}"

            )


    # ========================================================
    # CLASS WEIGHT
    # ========================================================

    positives = int(
        (
            train_data[
                "target"
            ] == 1
        ).sum()
    )


    negatives = int(
        (
            train_data[
                "target"
            ] == 0
        ).sum()
    )


    pos_weight_value = (

        negatives

        / max(
            positives,
            1,
        )

    )


    pos_weight = torch.tensor(

        [pos_weight_value],

        dtype=torch.float32,

        device=DEVICE,

    )


    print()

    print(
        f"pos_weight: {pos_weight_value:.4f}"
    )


    # ========================================================
    # TRANSFORMS
    # ========================================================

    train_transform = transforms.Compose([

        transforms.RandomHorizontalFlip(),

        transforms.RandomVerticalFlip(),

        transforms.RandomRotation(
            degrees=10
        ),

    ])


    evaluation_transform = None


    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = MagnetogramDataset(

        train_data,

        transform=train_transform,

    )


    validation_dataset = MagnetogramDataset(

        validation_data,

        transform=evaluation_transform,

    )


    test_dataset = MagnetogramDataset(

        test_data,

        transform=evaluation_transform,

    )


    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS,

        pin_memory=(
            DEVICE.type == "cuda"
        ),

    )


    validation_loader = DataLoader(

        validation_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=(
            DEVICE.type == "cuda"
        ),

    )


    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=(
            DEVICE.type == "cuda"
        ),

    )


    # ========================================================
    # MODEL
    # ========================================================

    model = create_model().to(
        DEVICE
    )


    # ========================================================
    # LOSS
    # ========================================================

    criterion = nn.BCEWithLogitsLoss(

        pos_weight=pos_weight

    )


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,

    )


    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler = (

        torch.optim.lr_scheduler.ReduceLROnPlateau(

            optimizer,

            mode="max",

            factor=0.5,

            patience=2,

        )

    )


    # ========================================================
    # TRAIN
    # ========================================================

    best_epoch, best_validation_pr = (

        train_model(

            model,

            train_loader,

            validation_loader,

            criterion,

            optimizer,

            scheduler,

        )

    )


    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    print()

    print(
        "Loading best validation checkpoint..."
    )


    checkpoint = torch.load(

        MODEL_PATH,

        map_location=DEVICE,

    )


    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ]

    )


    # ========================================================
    # VALIDATION
    # ========================================================

    (

        validation_loss,

        validation_roc,

        validation_pr,

        validation_actuals,

        validation_probabilities,

    ) = evaluate_model(

        model,

        validation_loader,

        criterion,

    )


    (

        threshold,

        validation_precision,

        validation_recall,

        validation_f1,

    ) = find_best_threshold(

        validation_actuals,

        validation_probabilities,

    )


    print()

    print(
        "========== VALIDATION RESULTS =========="
    )


    print(
        f"Best threshold: {threshold:.2f}"
    )

    print(
        f"ROC-AUC: {validation_roc:.4f}"
    )

    print(
        f"PR-AUC: {validation_pr:.4f}"
    )

    print(
        f"Precision: {validation_precision:.4f}"
    )

    print(
        f"Recall: {validation_recall:.4f}"
    )

    print(
        f"F1: {validation_f1:.4f}"
    )


    # ========================================================
    # FINAL TEST
    # ========================================================

    print()

    print(
        "=" * 60
    )

    print(
        " FINAL UNSEEN TEST EVALUATION"
    )

    print(
        "=" * 60
    )


    (

        test_loss,

        test_roc,

        test_pr,

        test_actuals,

        test_probabilities,

    ) = evaluate_model(

        model,

        test_loader,

        criterion,

    )


    test_predictions = (

        test_probabilities

        >= threshold

    ).astype(
        int
    )


    (

        test_precision,

        test_recall,

        test_f1,

        _,

    ) = precision_recall_fscore_support(

        test_actuals,

        test_predictions,

        average="binary",

        zero_division=0,

    )


    matrix = confusion_matrix(

        test_actuals,

        test_predictions,

    )


    print()

    print(
        "FINAL TEST PERIOD:"
    )

    print(
        "July 1 - December 31, 2012"
    )

    print()

    print(
        f"ROC-AUC  : {test_roc:.4f}"
    )

    print(
        f"PR-AUC   : {test_pr:.4f}"
    )

    print(
        f"Precision: {test_precision:.4f}"
    )

    print(
        f"Recall   : {test_recall:.4f}"
    )

    print(
        f"F1       : {test_f1:.4f}"
    )

    print()

    print(
        "CONFUSION MATRIX"
    )

    print(
        matrix
    )


    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    predictions = test_data.copy()

    predictions[
        "actual"
    ] = test_actuals

    predictions[
        "probability"
    ] = test_probabilities

    predictions[
        "prediction"
    ] = test_predictions


    predictions.to_csv(

        RESULTS_DIR
        / "final_test_predictions.csv",

        index=False,

    )


    # ========================================================
    # SAVE METRICS
    # ========================================================

    final_metrics = pd.DataFrame([{

        "best_epoch":
            best_epoch,

        "best_validation_pr_auc":
            best_validation_pr,

        "validation_roc_auc":
            validation_roc,

        "validation_pr_auc":
            validation_pr,

        "validation_precision":
            validation_precision,

        "validation_recall":
            validation_recall,

        "validation_f1":
            validation_f1,

        "threshold":
            threshold,

        "test_roc_auc":
            test_roc,

        "test_pr_auc":
            test_pr,

        "test_precision":
            test_precision,

        "test_recall":
            test_recall,

        "test_f1":
            test_f1,

    }])


    final_metrics.to_csv(

        RESULTS_DIR
        / "final_metrics.csv",

        index=False,

    )


    # ========================================================
    # PLOTS
    # ========================================================

    create_plots()


    # ========================================================
    # COMPLETE
    # ========================================================

    print()

    print(
        "=" * 60
    )

    print(
        " RESNET18 TRAINING COMPLETE"
    )

    print(
        "=" * 60
    )


    print()

    print(
        f"Best epoch: {best_epoch}"
    )


    print()

    print(
        "VALIDATION RESULTS"
    )

    print(
        f"ROC-AUC: {validation_roc:.4f}"
    )

    print(
        f"PR-AUC: {validation_pr:.4f}"
    )

    print(
        f"F1: {validation_f1:.4f}"
    )


    print()

    print(
        "FINAL TEST RESULTS"
    )

    print(
        f"ROC-AUC: {test_roc:.4f}"
    )

    print(
        f"PR-AUC: {test_pr:.4f}"
    )

    print(
        f"Precision: {test_precision:.4f}"
    )

    print(
        f"Recall: {test_recall:.4f}"
    )

    print(
        f"F1: {test_f1:.4f}"
    )


    print()

    print(
        "Model saved to:"
    )

    print(
        MODEL_PATH
    )


    print()

    print(
        "Results saved to:"
    )

    print(
        RESULTS_DIR
    )


if __name__ == "__main__":

    main()