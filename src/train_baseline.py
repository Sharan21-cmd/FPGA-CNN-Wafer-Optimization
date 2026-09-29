import os
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from dataset import WaferDataset, CLASSES
from model import WaferCNN


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/raw/LSWMD.pkl"
MODEL_PATH = "models/baseline_cnn.pth"

IMAGE_SIZE = 48
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# LEGACY PICKLE LOADER
# ============================================================

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("pandas.indexes"):
            module = module.replace(
                "pandas.indexes",
                "pandas.core.indexes",
                1
            )
        return super().find_class(module, name)


def load_wm811k(path):
    print("Loading WM-811K...")

    with open(path, "rb") as f:
        df = CustomUnpickler(
            f,
            encoding="latin1"
        ).load()

    return df


# ============================================================
# LABEL EXTRACTION
# ============================================================

def extract_label(x):
    try:
        return str(x.flat[0])
    except Exception:
        return "unknown"


# ============================================================
# PREPARE DATAFRAME
# ============================================================

def prepare_dataframe(df):

    df = df.copy()

    df["failure"] = df["failureType"].apply(extract_label)
    df["split"] = df["trianTestLabel"].apply(extract_label)

    valid_classes = set(CLASSES)

    df = df[
        df["failure"].isin(valid_classes)
        & df["split"].isin(["Training", "Test"])
    ].copy()

    train_df = df[df["split"] == "Training"].copy()
    test_df = df[df["split"] == "Test"].copy()

    return train_df, test_df


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader):

    model.eval()

    all_labels = []
    all_predictions = []

    start_time = time.perf_counter()

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.numpy())
            all_predictions.extend(
                predictions.cpu().numpy()
            )

    end_time = time.perf_counter()

    labels_np = np.array(all_labels)
    predictions_np = np.array(all_predictions)

    total_samples = len(labels_np)

    total_time = end_time - start_time

    accuracy = accuracy_score(
        labels_np,
        predictions_np
    )

    macro_f1 = f1_score(
        labels_np,
        predictions_np,
        average="macro",
        zero_division=0
    )

    latency_ms_per_image = (
        total_time / total_samples
    ) * 1000

    report = classification_report(
        labels_np,
        predictions_np,
        target_names=CLASSES,
        zero_division=0
    )

    matrix = confusion_matrix(
        labels_np,
        predictions_np
    )

    return (
        accuracy,
        macro_f1,
        latency_ms_per_image,
        report,
        matrix
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n========================================")
    print("      WAfer CNN BASELINE")
    print("========================================")

    print("Device:", DEVICE)
    print("Image size:", IMAGE_SIZE)
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = load_wm811k(DATA_PATH)

    train_df, test_df = prepare_dataframe(df)

    print("\nTraining samples:", len(train_df))
    print("Test samples:", len(test_df))

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = WaferDataset(
        train_df,
        image_size=IMAGE_SIZE
    )

    test_dataset = WaferDataset(
        test_df,
        image_size=IMAGE_SIZE
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = WaferCNN(
        num_classes=len(CLASSES)
    ).to(DEVICE)

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\nModel parameters:", total_parameters)
    print("Trainable parameters:", trainable_parameters)

    # --------------------------------------------------------
    # Loss / optimizer
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\n===== TRAINING =====")

    for epoch in range(EPOCHS):

        start = time.perf_counter()

        loss, accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        elapsed = time.perf_counter() - start

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {loss:.4f} "
            f"Accuracy: {accuracy:.4f} "
            f"Time: {elapsed:.2f}s"
        )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    print("\nModel saved to:")
    print(MODEL_PATH)

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print("\n===== TEST EVALUATION =====")

    (
        accuracy,
        macro_f1,
        latency,
        report,
        matrix
    ) = evaluate(
        model,
        test_loader
    )

    print("\nAccuracy:", f"{accuracy:.6f}")
    print("Macro F1:", f"{macro_f1:.6f}")
    print(
        "Average inference latency:",
        f"{latency:.6f} ms/image"
    )

    print("\n===== CLASSIFICATION REPORT =====")
    print(report)

    print("\n===== CONFUSION MATRIX =====")
    print(matrix)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    os.makedirs("reports", exist_ok=True)

    with open(
        "reports/baseline_results.txt",
        "w"
    ) as f:

        f.write("WAfer CNN BASELINE RESULTS\n")
        f.write("==========================\n\n")

        f.write(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}\n")
        f.write(f"Training samples: {len(train_df)}\n")
        f.write(f"Test samples: {len(test_df)}\n")
        f.write(f"Parameters: {total_parameters}\n\n")

        f.write(
            f"Accuracy: {accuracy:.6f}\n"
        )

        f.write(
            f"Macro F1: {macro_f1:.6f}\n"
        )

        f.write(
            f"Latency ms/image: {latency:.6f}\n\n"
        )

        f.write(
            "CLASSIFICATION REPORT\n"
        )
        f.write(
            "---------------------\n"
        )
        f.write(report)

        f.write(
            "\nCONFUSION MATRIX\n"
        )
        f.write(
            "-----------------\n"
        )
        f.write(
            np.array2string(matrix)
        )

    print(
        "\nResults saved to:"
        "\nreports/baseline_results.txt"
    )


if __name__ == "__main__":
    main()
