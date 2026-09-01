import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# PATHS
# ============================================================

project_root = Path(__file__).parent.parent

data_dir = (
    project_root
    / "data"
    / "processed"
    / "magnetograms"
)

labels_path = (
    data_dir
    / "dataset_labels.csv"
)

model_path = (
    project_root
    / "models"
    / "cnn_magnetogram.pt"
)

results_dir = (
    project_root
    / "results"
    / "cnn"
)

results_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

# TRAIN:
# January - April 2012
#
# VALIDATION:
# May - June 2012
#
# FINAL TEST:
# July - December 2012
#
# The final test set is NOT used for:
# - model selection
# - early stopping
# - threshold selection

TRAIN_END = pd.Timestamp(
    "2012-04-30 23:59:59"
)

VAL_END = pd.Timestamp(
    "2012-06-30 23:59:59"
)


# ============================================================
# DATASET
# ============================================================

class MagnetogramDataset(Dataset):

    def __init__(
        self,
        df,
        data_dir,
        augment=False,
    ):

        self.df = (
            df
            .reset_index(drop=True)
            .copy()
        )

        self.data_dir = Path(
            data_dir
        )

        self.augment = augment

        # ----------------------------------------------------
        # Keep only observations whose .npz file exists
        # ----------------------------------------------------

        valid_indices = []

        for idx, row in self.df.iterrows():

            fname = (
                f"harp_{int(row['HARPNUM'])}_"
                f"{row['observation_time'].strftime('%Y%m%d_%H%M%S')}_"
                f"t{int(row['target_24h'])}.npz"
            )

            if (
                self.data_dir / fname
            ).exists():

                valid_indices.append(idx)

        self.df = (
            self.df
            .iloc[valid_indices]
            .reset_index(drop=True)
        )

        print(
            f"Found {len(self.df)} valid .npz files."
        )


    def __len__(self):

        return len(self.df)


    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        fname = (
            f"harp_{int(row['HARPNUM'])}_"
            f"{row['observation_time'].strftime('%Y%m%d_%H%M%S')}_"
            f"t{int(row['target_24h'])}.npz"
        )

        file_path = (
            self.data_dir
            / fname
        )

        # ----------------------------------------------------
        # Load magnetogram
        # ----------------------------------------------------

        data = np.load(
            file_path
        )["img"]

        data = np.asarray(
            data,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Data augmentation
        #
        # Only horizontal/vertical flips.
        #
        # We avoid arbitrary rotations because the spatial
        # structure of the magnetic field may contain useful
        # information.
        # ----------------------------------------------------

        if self.augment:

            if random.random() < 0.5:

                data = np.fliplr(
                    data
                ).copy()

            if random.random() < 0.5:

                data = np.flipud(
                    data
                ).copy()

        # ----------------------------------------------------
        # Convert [H,W] -> [1,H,W]
        # ----------------------------------------------------

        img_tensor = torch.tensor(
            data,
            dtype=torch.float32
        ).unsqueeze(0)

        label = torch.tensor(
            float(row["target_24h"]),
            dtype=torch.float32
        )

        return (
            img_tensor,
            label
        )


# ============================================================
# MODEL
# ============================================================

def build_model():

    # ResNet18 architecture
    model = models.resnet18(
        weights=None
    )

    # --------------------------------------------------------
    # Original ResNet expects 3 channels.
    # Magnetograms are single-channel.
    # --------------------------------------------------------

    model.conv1 = nn.Conv2d(
        1,
        64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )

    # --------------------------------------------------------
    # Replace classifier
    # --------------------------------------------------------

    num_features = (
        model.fc.in_features
    )

    model.fc = nn.Sequential(

        nn.Dropout(
            p=0.4
        ),

        nn.Linear(
            num_features,
            1
        )
    )

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    loader,
    device,
    criterion=None,
):

    model.eval()

    all_probs = []
    all_labels = []

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for inputs, labels in loader:

            inputs = inputs.to(
                device
            )

            labels = labels.to(
                device
            )

            outputs = model(
                inputs
            ).view(-1)

            if criterion is not None:

                loss = criterion(
                    outputs,
                    labels
                )

                total_loss += (
                    loss.item()
                    * inputs.size(0)
                )

            probabilities = torch.sigmoid(
                outputs
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

    if len(all_labels) == 0:

        return None

    all_labels = np.asarray(
        all_labels
    )

    all_probs = np.asarray(
        all_probs
    )

    # --------------------------------------------------------
    # ROC-AUC and PR-AUC
    # --------------------------------------------------------

    if len(
        np.unique(all_labels)
    ) > 1:

        roc_auc = roc_auc_score(
            all_labels,
            all_probs
        )

        pr_auc = average_precision_score(
            all_labels,
            all_probs
        )

    else:

        roc_auc = 0.0
        pr_auc = 0.0

    avg_loss = (
        total_loss
        / max(total_samples, 1)
    )

    return {

        "loss": avg_loss,

        "roc_auc": roc_auc,

        "pr_auc": pr_auc,

        "labels": all_labels,

        "probabilities": all_probs,

    }


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def find_best_threshold(
    labels,
    probabilities,
):

    thresholds = np.arange(
        0.01,
        0.51,
        0.01
    )

    best_threshold = 0.5
    best_f1 = -1

    rows = []

    for threshold in thresholds:

        predictions = (
            probabilities
            >= threshold
        ).astype(int)
        

        precision = precision_score(
            labels,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            labels,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            labels,
            predictions,
            zero_division=0
        )

        rows.append({

            "threshold": threshold,

            "precision": precision,

            "recall": recall,

            "f1": f1,

        })

        if f1 > best_f1:

            best_f1 = f1

            best_threshold = threshold

    threshold_df = pd.DataFrame(
        rows
    )

    threshold_df.to_csv(
        results_dir
        / "threshold_results.csv",
        index=False
    )

    return (
        best_threshold,
        threshold_df
    )


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train_model():

    print(
        "======================================================="
    )

    print(
        " CLEAN TEMPORAL CNN TRAINING"
    )

    print(
        "======================================================="
    )

    # --------------------------------------------------------
    # Check labels
    # --------------------------------------------------------

    if not labels_path.exists():

        print(
            f"ERROR: Labels file not found:"
        )

        print(
            labels_path
        )

        return

    print(
        "\nLoading labels..."
    )

    df = pd.read_csv(
        labels_path
    )

    df[
        "observation_time"
    ] = pd.to_datetime(
        df["observation_time"],
        errors="coerce"
    )

    # Remove invalid timestamps
    df = df.dropna(
        subset=[
            "observation_time"
        ]
    )

    # Sort chronologically
    df = df.sort_values(
        "observation_time"
    ).reset_index(
        drop=True
    )

    print(
        f"Total label rows: {len(df)}"
    )


    # ========================================================
    # CHRONOLOGICAL SPLIT
    # ========================================================

    train_df = df[
        df["observation_time"]
        <= TRAIN_END
    ].copy()

    val_df = df[
        (
            df["observation_time"]
            > TRAIN_END
        )
        &
        (
            df["observation_time"]
            <= VAL_END
        )
    ].copy()

    test_df = df[
        df["observation_time"]
        > VAL_END
    ].copy()


    print()
    print(
        "========== CHRONOLOGICAL SPLIT =========="
    )

    print(
        f"TRAIN:"
    )

    print(
        f"  Rows: {len(train_df)}"
    )

    print(
        f"  Period: "
        f"{train_df['observation_time'].min()} "
        f"-> "
        f"{train_df['observation_time'].max()}"
    )

    print()
    print(
        f"VALIDATION:"
    )

    print(
        f"  Rows: {len(val_df)}"
    )

    print(
        f"  Period: "
        f"{val_df['observation_time'].min()} "
        f"-> "
        f"{val_df['observation_time'].max()}"
    )

    print()
    print(
        f"FINAL TEST:"
    )

    print(
        f"  Rows: {len(test_df)}"
    )

    print(
        f"  Period: "
        f"{test_df['observation_time'].min()} "
        f"-> "
        f"{test_df['observation_time'].max()}"
    )


    # ========================================================
    # DATASETS
    # ========================================================

    print()
    print(
        "========== CHECKING MAGNETOGRAM FILES =========="
    )

    train_dataset = MagnetogramDataset(
        train_df,
        data_dir,
        augment=True
    )

    val_dataset = MagnetogramDataset(
        val_df,
        data_dir,
        augment=False
    )

    test_dataset = MagnetogramDataset(
        test_df,
        data_dir,
        augment=False
    )

    print()
    print(
        "========== VALID FILE COUNTS =========="
    )

    print(
        f"Train:      {len(train_dataset)}"
    )

    print(
        f"Validation: {len(val_dataset)}"
    )

    print(
        f"Final test: {len(test_dataset)}"
    )

    if (
        len(train_dataset) == 0
        or len(val_dataset) == 0
        or len(test_dataset) == 0
    ):

        print(
            "\nERROR: One of the datasets is empty."
        )

        return


    # ========================================================
    # CLASS DISTRIBUTION
    # ========================================================

    print()
    print(
        "========== CLASS DISTRIBUTION =========="
    )

    train_positive = int(
        train_dataset.df[
            "target_24h"
        ].sum()
    )

    train_negative = (
        len(train_dataset)
        - train_positive
    )

    val_positive = int(
        val_dataset.df[
            "target_24h"
        ].sum()
    )

    val_negative = (
        len(val_dataset)
        - val_positive
    )

    test_positive = int(
        test_dataset.df[
            "target_24h"
        ].sum()
    )

    test_negative = (
        len(test_dataset)
        - test_positive
    )

    print(
        f"TRAIN      -> "
        f"Positive: {train_positive}, "
        f"Negative: {train_negative}"
    )

    print(
        f"VALIDATION -> "
        f"Positive: {val_positive}, "
        f"Negative: {val_negative}"
    )

    print(
        f"FINAL TEST -> "
        f"Positive: {test_positive}, "
        f"Negative: {test_negative}"
    )


    # ========================================================
    # DATA LOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )


    # ========================================================
    # CLASS WEIGHT
    # ========================================================

    pos_weight_value = (
        train_negative
        / max(train_positive, 1)
    )

    print()
    print(
        f"pos_weight: "
        f"{pos_weight_value:.2f}"
    )


    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )


    # ========================================================
    # MODEL
    # ========================================================

    model = build_model().to(
        device
    )


    # ========================================================
    # LOSS
    # ========================================================

    criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                [pos_weight_value],
                dtype=torch.float32,
                device=device
            )
        )
    )


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = optim.AdamW(
        model.parameters(),
        lr=3e-5,
        weight_decay=1e-4
    )


    # ========================================================
    # LEARNING RATE SCHEDULER
    # ========================================================

    scheduler = (
        optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2
        )
    )


    # ========================================================
    # TRAINING SETTINGS
    # ========================================================

    epochs = 15

    patience = 4

    best_pr_auc = -np.inf

    best_epoch = 0

    epochs_without_improvement = 0

    training_history = []


    # ========================================================
    # TRAINING LOOP
    # ========================================================

    print()
    print(
        "======================================================="
    )

    print(
        " STARTING CNN TRAINING"
    )

    print(
        "======================================================="
    )

    for epoch in range(
        epochs
    ):

        model.train()

        running_loss = 0.0

        samples_seen = 0


        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        for inputs, labels in train_loader:

            inputs = inputs.to(
                device
            )

            labels = labels.to(
                device
            )

            optimizer.zero_grad()

            outputs = model(
                inputs
            ).view(-1)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
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
                1
            )
        )


        # ----------------------------------------------------
        # VALIDATION
        #
        # IMPORTANT:
        # We evaluate validation data here.
        # We DO NOT touch final test data.
        # ----------------------------------------------------

        evaluation = evaluate_model(
            model,
            val_loader,
            device,
            criterion
        )

        val_loss = evaluation[
            "loss"
        ]

        val_roc = evaluation[
            "roc_auc"
        ]

        val_pr = evaluation[
            "pr_auc"
        ]


        # Scheduler uses validation PR-AUC
        scheduler.step(
            val_pr
        )

        current_lr = (
            optimizer
            .param_groups[0]
            ["lr"]
        )


        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        training_history.append({

            "epoch":
                epoch + 1,

            "train_loss":
                train_loss,

            "validation_loss":
                val_loss,

            "validation_roc_auc":
                val_roc,

            "validation_pr_auc":
                val_pr,

            "learning_rate":
                current_lr,

        })


        # ----------------------------------------------------
        # Print epoch
        # ----------------------------------------------------

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Train Loss: {train_loss:.4f} "
            f"- Val Loss: {val_loss:.4f} "
            f"- Val ROC-AUC: {val_roc:.4f} "
            f"- Val PR-AUC: {val_pr:.4f}"
        )


        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        if val_pr > best_pr_auc:

            best_pr_auc = val_pr

            best_epoch = (
                epoch + 1
            )

            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                model_path
            )

            print(
                f"  -> New best model! "
                f"PR-AUC: {val_pr:.4f}"
            )

        else:

            epochs_without_improvement += 1


        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= patience
        ):

            print()
            print(
                "Early stopping triggered."
            )

            break


    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    print()
    print(
        "Loading best validation checkpoint..."
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )


    # ========================================================
    # VALIDATION THRESHOLD
    #
    # Threshold is selected ONLY using validation data.
    # ========================================================

    val_eval = evaluate_model(
        model,
        val_loader,
        device
    )
    validation_predictions_df = pd.DataFrame({
    "actual": val_eval["labels"].astype(int),
    "probability": val_eval["probabilities"],
})

    validation_predictions_df.to_csv(
    results_dir / "validation_predictions.csv",
    index=False
)

    print(
    "Validation predictions saved to:"
)

    print(
    results_dir / "validation_predictions.csv"
)

    val_labels = (
        val_eval["labels"]
    )

    val_probabilities = (
        val_eval["probabilities"]
    )

    (
        best_threshold,
        threshold_df
    ) = find_best_threshold(
        val_labels,
        val_probabilities
    )


    val_predictions = (
        val_probabilities
        >= best_threshold
    ).astype(int)


    val_precision = precision_score(
        val_labels,
        val_predictions,
        zero_division=0
    )

    val_recall = recall_score(
        val_labels,
        val_predictions,
        zero_division=0
    )

    val_f1 = f1_score(
        val_labels,
        val_predictions,
        zero_division=0
    )


    print()
    print(
        "========== VALIDATION THRESHOLD =========="
    )

    print(
        f"Best threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Validation Precision: "
        f"{val_precision:.4f}"
    )

    print(
        f"Validation Recall: "
        f"{val_recall:.4f}"
    )

    print(
        f"Validation F1: "
        f"{val_f1:.4f}"
    )


    # ========================================================
    # FINAL UNSEEN TEST
    #
    # THIS IS THE FIRST TIME WE USE THIS DATA.
    # ========================================================

    print()
    print(
        "======================================================="
    )

    print(
        " FINAL UNSEEN TEST EVALUATION"
    )

    print(
        "======================================================="
    )

    final_eval = evaluate_model(
        model,
        test_loader,
        device
    )

    test_labels = (
        final_eval["labels"]
    )

    test_probabilities = (
        final_eval["probabilities"]
    )


    # Apply threshold selected from validation
    test_predictions = (
        test_probabilities
        >= best_threshold
    ).astype(int)
    


    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

    test_roc = roc_auc_score(
        test_labels,
        test_probabilities
    )

    test_pr = average_precision_score(
        test_labels,
        test_probabilities
    )

    test_precision = precision_score(
        test_labels,
        test_predictions,
        zero_division=0
    )

    test_recall = recall_score(
        test_labels,
        test_predictions,
        zero_division=0
    )
    test_accuracy = accuracy_score(
        test_labels,
        test_predictions,
    )
    

    test_f1 = f1_score(
        test_labels,
        test_predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        test_labels,
        test_predictions
    )


    # ========================================================
    # SAVE TRAINING HISTORY
    # ========================================================

    history_df = pd.DataFrame(
        training_history
    )

    history_df.to_csv(
        results_dir
        / "training_metrics.csv",
        index=False
    )


    # ========================================================
    # SAVE FINAL METRICS
    # ========================================================

    metrics_df = pd.DataFrame([{

        # Final test
"test_roc_auc":
    test_roc,

"test_pr_auc":
    test_pr,

"test_accuracy":
    test_accuracy,

"test_precision":
    test_precision,

"test_recall":
    test_recall,

"test_f1":
    test_f1,

    }])


    metrics_df.to_csv(
        results_dir
        / "final_metrics.csv",
        index=False
    )


    # ========================================================
    # SAVE FINAL PREDICTIONS
    # ========================================================

    prediction_df = pd.DataFrame({

        "actual":
            test_labels.astype(int),

        "probability":
            test_probabilities,

        "prediction":
            test_predictions,
        "test_accuracy":
        test_accuracy,

    })

    prediction_df.to_csv(
        results_dir
        / "final_test_predictions.csv",
        index=False
    )


    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print(
        "======================================================="
    )

    print(
        " CNN TRAINING COMPLETE"
    )

    print(
        "======================================================="
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
        f"ROC-AUC  : "
        f"{val_eval['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC   : "
        f"{val_eval['pr_auc']:.4f}"
    )

    print(
        f"F1       : "
        f"{val_f1:.4f}"
    )

    print(
        f"Threshold: "
        f"{best_threshold:.2f}"
    )

    print()
    print(
        "======================================================="
    )

    print(
        " FINAL UNSEEN TEST RESULTS"
    )

    print(
        "======================================================="
    )

    print(
        "Test period: July - December 2012"
    )

    print(
        f"ROC-AUC  : "
        f"{test_roc:.4f}"
    )

    print(
        f"PR-AUC   : "
        f"{test_pr:.4f}"
    )

    print(
        f"Precision: "
        f"{test_precision:.4f}"
    )

    print(
        f"Recall   : "
        f"{test_recall:.4f}"
    )

    print(
        f"F1       : "
        f"{test_f1:.4f}"
    )
    print(
    f"Accuracy : "
    f"{test_accuracy:.4f}"
)

    print()
    print(
        "CONFUSION MATRIX"
    )

    print(cm)

    print()
    print(
        "Best model:"
    )

    print(
        model_path
    )

    print()
    print(
        "Training metrics:"
    )

    print(
        results_dir
        / "training_metrics.csv"
    )

    print()
    print(
        "Final metrics:"
    )

    print(
        results_dir
        / "final_metrics.csv"
    )

    print()
    print(
        "Final predictions:"
    )

    print(
        results_dir
        / "final_test_predictions.csv"
    )

    print()
    print(
        "======================================================="
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    train_model()