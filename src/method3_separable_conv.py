import os
import sys
import time
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report

sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

from dataset import WaferDataset, CLASSES


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "LSWMD.pkl"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "separable_cnn.pth"
)

REPORT_PATH = os.path.join(
    PROJECT_ROOT,
    "reports",
    "method3_separable_conv.txt"
)

IMAGE_SIZE = 48
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

DEVICE = torch.device("cpu")


# ============================================================
# WM-811K LOADER
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

        return CustomUnpickler(
            f,
            encoding="latin1"
        ).load()


# ============================================================
# LABEL EXTRACTION
# ============================================================

def extract_label(x):

    try:
        return str(x.flat[0])
    except Exception:
        return "unknown"


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_dataframe(df, split_name):

    labels = df["failureType"].apply(
        extract_label
    )

    split = df["trianTestLabel"].apply(
        extract_label
    )

    mask = (
        split.eq(split_name)
        & labels.isin(CLASSES)
    )

    result = df.loc[
        mask,
        ["waferMap", "failureType"]
    ].copy()

    result["failure"] = result[
        "failureType"
    ].apply(extract_label)

    return result[
        ["waferMap", "failure"]
    ].reset_index(drop=True)


# ============================================================
# DEPTHWISE-SEPARABLE CONV
# ============================================================

class DepthwiseSeparableConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=True
        )

        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            bias=True
        )

    def forward(self, x):

        x = self.depthwise(x)
        x = self.pointwise(x)

        return x


# ============================================================
# SEPARABLE CNN
# ============================================================

class SeparableWaferCNN(nn.Module):

    def __init__(self, num_classes=9):

        super().__init__()

        self.features = nn.Sequential(

            DepthwiseSeparableConv(
                1,
                16
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),


            DepthwiseSeparableConv(
                16,
                32
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),


            DepthwiseSeparableConv(
                32,
                64
            ),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                64 * 6 * 6,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# MAC COUNT
# ============================================================

def calculate_macs():

    # Input: 48x48

    # --------------------------------------------------------
    # Layer 1
    # --------------------------------------------------------

    h1 = 48
    w1 = 48

    normal1 = (
        h1 * w1 *
        3 * 3 *
        1 * 16
    )

    separable1 = (
        h1 * w1 *
        (
            3 * 3 * 1
            +
            1 * 1 * 1 * 16
        )
    )

    # Pool

    h2 = 24
    w2 = 24

    # --------------------------------------------------------
    # Layer 2
    # --------------------------------------------------------

    normal2 = (
        h2 * w2 *
        3 * 3 *
        16 * 32
    )

    separable2 = (
        h2 * w2 *
        (
            3 * 3 * 16
            +
            1 * 1 * 16 * 32
        )
    )

    # Pool

    h3 = 12
    w3 = 12

    # --------------------------------------------------------
    # Layer 3
    # --------------------------------------------------------

    normal3 = (
        h3 * w3 *
        3 * 3 *
        32 * 64
    )

    separable3 = (
        h3 * w3 *
        (
            3 * 3 * 32
            +
            1 * 1 * 32 * 64
        )
    )

    normal_total = (
        normal1 +
        normal2 +
        normal3
    )

    separable_total = (
        separable1 +
        separable2 +
        separable3
    )

    reduction = (
        1 -
        separable_total /
        normal_total
    ) * 100

    return (
        normal_total,
        separable_total,
        reduction
    )


# ============================================================
# TRAIN
# ============================================================

def train_model(
    model,
    train_loader
):

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    model.train()

    for epoch in range(EPOCHS):

        running_loss = 0.0
        correct = 0
        total = 0

        start = time.perf_counter()

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item() *
                images.size(0)
            )

            predictions = (
                outputs.argmax(dim=1)
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        epoch_time = (
            time.perf_counter() - start
        )

        accuracy = (
            correct / total
        )

        loss_value = (
            running_loss / total
        )

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Loss {loss_value:.4f} | "
            f"Train Acc {accuracy * 100:.2f}% | "
            f"Time {epoch_time:.1f}s"
        )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    loader
):

    model.eval()

    all_targets = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            outputs = model(images)

            predictions = (
                outputs.argmax(dim=1)
            )

            all_targets.extend(
                labels.numpy()
            )

            all_predictions.extend(
                predictions.numpy()
            )

    targets = np.asarray(
        all_targets
    )

    predictions = np.asarray(
        all_predictions
    )

    accuracy = (
        predictions == targets
    ).mean()

    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro"
    )

    return (
        accuracy,
        macro_f1,
        targets,
        predictions
    )


# ============================================================
# CPU LATENCY
# ============================================================

def benchmark_latency(
    model,
    loader,
    warmup_batches=10,
    benchmark_batches=100
):

    model.eval()

    with torch.no_grad():

        for i, (images, _) in enumerate(loader):

            images = images.to(DEVICE)

            _ = model(images)

            if i + 1 >= warmup_batches:
                break

    total_images = 0

    start = time.perf_counter()

    with torch.no_grad():

        for i, (images, _) in enumerate(loader):

            images = images.to(DEVICE)

            _ = model(images)

            total_images += images.size(0)

            if i + 1 >= benchmark_batches:
                break

    elapsed = (
        time.perf_counter() - start
    )

    return (
        elapsed /
        total_images *
        1000
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("METHOD 3 - DEPTHWISE-SEPARABLE CONVOLUTION")
    print("=" * 70)

    print(
        "Replacing standard 3x3 convolutions "
        "with depthwise + pointwise convolutions."
    )

    print()

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = load_wm811k(
        DATA_PATH
    )

    train_df = prepare_dataframe(
        df,
        "Training"
    )

    test_df = prepare_dataframe(
        df,
        "Test"
    )

    print(
        f"Training samples: {len(train_df)}"
    )

    print(
        f"Test samples: {len(test_df)}"
    )

    print()

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
    # Create model
    # --------------------------------------------------------

    print(
        "Creating separable CNN..."
    )

    model = SeparableWaferCNN(
        num_classes=len(CLASSES)
    ).to(DEVICE)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters: {parameter_count:,}"
    )

    print()

    # --------------------------------------------------------
    # MAC analysis
    # --------------------------------------------------------

    (
        normal_macs,
        separable_macs,
        mac_reduction
    ) = calculate_macs()

    print(
        f"Standard convolution MACs: "
        f"{normal_macs:,}"
    )

    print(
        f"Separable convolution MACs: "
        f"{separable_macs:,}"
    )

    print(
        f"Convolution MAC reduction: "
        f"{mac_reduction:.2f}%"
    )

    print()

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print(
        "Training separable CNN..."
    )

    train_model(
        model,
        train_loader
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    print()
    print(
        f"Model saved: {MODEL_PATH}"
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print()
    print(
        "Evaluating separable CNN..."
    )

    (
        accuracy,
        macro_f1,
        targets,
        predictions
    ) = evaluate(
        model,
        test_loader
    )

    print(
        f"Accuracy: "
        f"{accuracy * 100:.4f}%"
    )

    print(
        f"Macro F1: "
        f"{macro_f1 * 100:.4f}%"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        targets,
        predictions,
        target_names=CLASSES,
        digits=4,
        zero_division=0
    )

    print()
    print(report)

    # --------------------------------------------------------
    # Latency
    # --------------------------------------------------------

    print(
        "Benchmarking CPU latency..."
    )

    latency = benchmark_latency(
        model,
        test_loader
    )

    print(
        f"CPU latency: "
        f"{latency:.6f} ms/image"
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(REPORT_PATH),
        exist_ok=True
    )

    with open(
        REPORT_PATH,
        "w"
    ) as f:

        f.write(
            "METHOD 3 - DEPTHWISE-SEPARABLE CONVOLUTION\n"
        )

        f.write(
            "===========================================\n\n"
        )

        f.write(
            f"Input size: 1x{IMAGE_SIZE}x{IMAGE_SIZE}\n"
        )

        f.write(
            f"Training samples: {len(train_df)}\n"
        )

        f.write(
            f"Test samples: {len(test_df)}\n\n"
        )

        f.write(
            f"Parameters: {parameter_count:,}\n"
        )

        f.write(
            f"Standard convolution MACs: "
            f"{normal_macs:,}\n"
        )

        f.write(
            f"Separable convolution MACs: "
            f"{separable_macs:,}\n"
        )

        f.write(
            f"MAC reduction: "
            f"{mac_reduction:.4f}%\n\n"
        )

        f.write(
            f"Accuracy: "
            f"{accuracy * 100:.4f}%\n"
        )

        f.write(
            f"Macro F1: "
            f"{macro_f1 * 100:.4f}%\n"
        )

        f.write(
            f"CPU latency: "
            f"{latency:.6f} ms/image\n\n"
        )

        f.write(
            "Classification Report\n"
        )

        f.write(
            "---------------------\n"
        )

        f.write(report)

    print()
    print(
        f"Report saved: {REPORT_PATH}"
    )

    print()
    print("=" * 70)
    print("METHOD 3 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
