import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from sklearn.metrics import roc_auc_score, average_precision_score
from pathlib import Path


# ============================================================
# Paths
# ============================================================

project_root = Path(__file__).parent.parent

data_dir = project_root / "data" / "processed" / "magnetograms"
labels_path = data_dir / "dataset_labels.csv"

model_path = project_root / "models" / "cnn_magnetogram.pt"

# Results folder for CNN outputs
results_dir = project_root / "results" / "cnn"
results_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# Split Date
# Same chronological split as the tabular model
# ============================================================

TRAIN_END = pd.Timestamp("2012-06-30 23:59:59")


# ============================================================
# Dataset
# ============================================================

class MagnetogramDataset(Dataset):

    def __init__(self, df, data_dir, transform=None):

        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.transform = transform

        # Keep only files that actually exist
        valid_indices = []

        for idx, row in self.df.iterrows():

            fname = (
                f"harp_{row['HARPNUM']}_"
                f"{row['observation_time'].strftime('%Y%m%d_%H%M%S')}"
                f"_t{row['target_24h']}.npz"
            )

            if (self.data_dir / fname).exists():
                valid_indices.append(idx)

        self.df = self.df.iloc[valid_indices].reset_index(drop=True)

        print(f"Found {len(self.df)} valid .npz files.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        fname = (
            f"harp_{row['HARPNUM']}_"
            f"{row['observation_time'].strftime('%Y%m%d_%H%M%S')}"
            f"_t{row['target_24h']}.npz"
        )

        # Load magnetogram
        data = np.load(self.data_dir / fname)["img"]

        # Convert to tensor [C, H, W]
        img_tensor = torch.tensor(
            data,
            dtype=torch.float32
        ).unsqueeze(0)

        label = torch.tensor(
            row["target_24h"],
            dtype=torch.float32
        )

        return img_tensor, label


# ============================================================
# CNN Model
# ============================================================

def build_model():

    # ResNet18 without pretrained ImageNet weights
    model = models.resnet18(weights=None)

    # Change first layer from RGB (3 channels)
    # to magnetogram input (1 channel)
    model.conv1 = nn.Conv2d(
        1,
        64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )

    # Binary classification
    num_ftrs = model.fc.in_features

    model.fc = nn.Linear(
        num_ftrs,
        1
    )

    return model


# ============================================================
# Plot Results
# ============================================================

def save_training_plots(
    epochs,
    train_losses,
    val_losses,
    roc_scores,
    pr_scores
):

    # --------------------------------------------------------
    # Training / Validation Loss
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_losses,
        marker="o",
        label="Training Loss"
    )

    plt.plot(
        epochs,
        val_losses,
        marker="o",
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Training vs Validation Loss")

    plt.xticks(epochs)
    plt.legend()
    plt.tight_layout()

    loss_path = results_dir / "training_loss.png"

    plt.savefig(
        loss_path,
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # ROC-AUC / PR-AUC
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        roc_scores,
        marker="o",
        label="ROC-AUC"
    )

    plt.plot(
        epochs,
        pr_scores,
        marker="o",
        label="PR-AUC"
    )

    # Mark best PR-AUC epoch
    best_index = int(np.argmax(pr_scores))
    best_epoch = epochs[best_index]

    plt.axvline(
        best_epoch,
        linestyle="--",
        label=f"Best PR-AUC (Epoch {best_epoch})"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("CNN Validation Performance")

    plt.xticks(epochs)
    plt.ylim(0, 1)

    plt.legend()
    plt.tight_layout()

    metrics_path = results_dir / "validation_metrics.png"

    plt.savefig(
        metrics_path,
        dpi=200
    )

    plt.close()

    print()
    print("Plots saved:")
    print(f"  Loss plot:    {loss_path}")
    print(f"  Metrics plot: {metrics_path}")


# ============================================================
# Training
# ============================================================

def train_model():

    if not labels_path.exists():

        print(
            f"Error: {labels_path} not found. "
            "Run download_magnetograms.py first."
        )

        return

    print("Loading labels...")

    df = pd.read_csv(labels_path)

    df["observation_time"] = pd.to_datetime(
        df["observation_time"]
    )

    df = df.sort_values(
        "observation_time"
    )

    # ========================================================
    # Chronological Split
    # ========================================================

    train_df = df[
        df["observation_time"] <= TRAIN_END
    ]

    test_df = df[
        df["observation_time"] > TRAIN_END
    ]

    print(
        f"Train samples before file check: "
        f"{len(train_df)}"
    )

    print(
        f"Test samples before file check: "
        f"{len(test_df)}"
    )

    # ========================================================
    # Create datasets
    # ========================================================

    train_dataset = MagnetogramDataset(
        train_df,
        data_dir
    )

    test_dataset = MagnetogramDataset(
        test_df,
        data_dir
    )

    print(
        f"Actual train samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Actual test samples: "
        f"{len(test_dataset)}"
    )

    if len(train_dataset) == 0:

        print(
            "No training data available. Exiting."
        )

        return

    if len(test_dataset) == 0:

        print(
            "No test data available. Exiting."
        )

        return

    # ========================================================
    # DataLoaders
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )

    # ========================================================
    # Class imbalance
    # ========================================================

    num_pos = train_dataset.df[
        "target_24h"
    ].sum()

    num_neg = (
        len(train_dataset.df) - num_pos
    )

    pos_weight_val = (
        num_neg / max(num_pos, 1)
    )

    print(
        f"Training positives: {int(num_pos)}"
    )

    print(
        f"Training negatives: {int(num_neg)}"
    )

    print(
        f"pos_weight: {pos_weight_val:.2f}"
    )

    # ========================================================
    # Device
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
    # Model
    # ========================================================

    model = build_model().to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [pos_weight_val],
            dtype=torch.float32
        ).to(device)
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    epochs = 10

    # ========================================================
    # Metric history
    # ========================================================

    epoch_numbers = []

    train_losses = []
    val_losses = []

    roc_scores = []
    pr_scores = []

    # ========================================================
    # Best model tracking
    # ========================================================

    best_pr_auc = -1.0
    best_roc_auc = -1.0
    best_epoch = 0

    # ========================================================
    # Training loop
    # ========================================================

    print("Starting training...")

    for epoch in range(epochs):

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        model.train()

        running_loss = 0.0

        for inputs, labels in train_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs).squeeze()

            # Handle batch size 1
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item() *
                inputs.size(0)
            )

        epoch_loss = (
            running_loss /
            len(train_dataset)
        )

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        model.eval()

        all_preds = []
        all_labels = []

        val_loss = 0.0

        with torch.no_grad():

            for inputs, labels in test_loader:

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs).squeeze()

                # Handle batch size 1
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)

                loss = criterion(
                    outputs,
                    labels
                )

                val_loss += (
                    loss.item() *
                    inputs.size(0)
                )

                probs = torch.sigmoid(
                    outputs
                )

                all_preds.extend(
                    probs.cpu().numpy()
                )

                all_labels.extend(
                    labels.cpu().numpy()
                )

        val_loss = (
            val_loss /
            len(test_dataset)
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        if (
            len(all_labels) > 0
            and len(np.unique(all_labels)) > 1
        ):

            val_roc = roc_auc_score(
                all_labels,
                all_preds
            )

            val_pr = average_precision_score(
                all_labels,
                all_preds
            )

        else:

            val_roc = 0.0
            val_pr = 0.0

        # Save history
        epoch_numbers.append(
            epoch + 1
        )

        train_losses.append(
            epoch_loss
        )

        val_losses.append(
            val_loss
        )

        roc_scores.append(
            val_roc
        )

        pr_scores.append(
            val_pr
        )

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Train Loss: {epoch_loss:.4f} "
            f"- Val Loss: {val_loss:.4f} "
            f"- Val ROC-AUC: {val_roc:.4f} "
            f"- Val PR-AUC: {val_pr:.4f}"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_pr > best_pr_auc:

            best_pr_auc = val_pr
            best_roc_auc = val_roc
            best_epoch = epoch + 1

            model_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            torch.save(
                model.state_dict(),
                model_path
            )

            print(
                f"  -> New best model! "
                f"PR-AUC: {best_pr_auc:.4f}"
            )

    # ========================================================
    # Save metrics CSV
    # ========================================================

    metrics_df = pd.DataFrame({
        "epoch": epoch_numbers,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "roc_auc": roc_scores,
        "pr_auc": pr_scores
    })

    metrics_path = (
        results_dir /
        "training_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False
    )

    # ========================================================
    # Save plots
    # ========================================================

    save_training_plots(
        epoch_numbers,
        train_losses,
        val_losses,
        roc_scores,
        pr_scores
    )

    # ========================================================
    # Final summary
    # ========================================================

    print()
    print("=" * 55)
    print("CNN TRAINING COMPLETE")
    print("=" * 55)

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best ROC-AUC: {best_roc_auc:.4f}"
    )

    print(
        f"Best PR-AUC: {best_pr_auc:.4f}"
    )

    print()
    print(
        f"Best model saved to:"
    )

    print(
        f"  {model_path}"
    )

    print()
    print(
        f"Training metrics saved to:"
    )

    print(
        f"  {metrics_path}"
    )

    print()
    print(
        f"Results directory:"
    )

    print(
        f"  {results_dir}"
    )

    print("=" * 55)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    train_model()