import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from src.model import WaferCNN
from src.dataset import WaferDataset


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "data/raw/LSWMD.pkl"
MODEL_PATH = "models/baseline_cnn.pth"

IMAGE_SIZE = 48
BATCH_SIZE = 128

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
# DATA
# ============================================================

def prepare_data():

    print("Loading WM-811K...")

    df = load_wm811k(DATA_PATH)

    df["split"] = df[
        "trianTestLabel"
    ].apply(extract_label)

    df["failure"] = df[
        "failureType"
    ].apply(extract_label)

    df = df[
        df["split"].isin(
            ["Training", "Test"]
        )
    ].copy()

    test_df = df[
        df["split"] == "Test"
    ].copy()

    print(
        f"Test samples: {len(test_df)}"
    )

    test_dataset = WaferDataset(
        test_df,
        image_size=IMAGE_SIZE
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return test_loader


# ============================================================
# ZERO-SKIPPING CONVOLUTION
# ============================================================

def zero_skip_conv2d(
    x,
    conv
):

    """
    Numerically equivalent convolution.

    Zero-valued input activations are not included
    in the MAC accumulation.

    This is a software simulation of zero skipping.
    """

    weight = conv.weight
    bias = conv.bias

    B, Cin, H, W = x.shape

    Cout = weight.shape[0]
    K = weight.shape[2]

    padding = conv.padding[0]
    stride = conv.stride[0]

    if padding > 0:

        x_padded = torch.nn.functional.pad(
            x,
            (
                padding,
                padding,
                padding,
                padding
            )
        )

    else:

        x_padded = x

    out_H = (
        H +
        2 * padding -
        K
    ) // stride + 1

    out_W = (
        W +
        2 * padding -
        K
    ) // stride + 1

    output = torch.zeros(
        (
            B,
            Cout,
            out_H,
            out_W
        ),
        dtype=x.dtype,
        device=x.device
    )

    total_macs = 0
    executed_macs = 0
    skipped_macs = 0

    # --------------------------------------------------------
    # Explicit convolution
    # --------------------------------------------------------

    for oh in range(out_H):

        h_start = oh * stride

        for ow in range(out_W):

            w_start = ow * stride

            patch = x_padded[
                :,
                :,
                h_start:h_start + K,
                w_start:w_start + K
            ]

            # Shape:
            # B x Cin x K x K

            for b in range(B):

                patch_b = patch[b]

                for cout in range(Cout):

                    weights = weight[cout]

                    products = (
                        patch_b *
                        weights
                    )

                    nonzero_mask = (
                        patch_b != 0
                    )

                    total = products.numel()

                    nonzero_count = (
                        nonzero_mask.sum().item()
                    )

                    total_macs += total
                    executed_macs += nonzero_count
                    skipped_macs += (
                        total -
                        nonzero_count
                    )

                    output[
                        b,
                        cout,
                        oh,
                        ow
                    ] = (
                        products[
                            nonzero_mask
                        ].sum()
                    )

                    if bias is not None:

                        output[
                            b,
                            cout,
                            oh,
                            ow
                        ] += bias[cout]

    stats = {
        "total_macs": total_macs,
        "executed_macs": executed_macs,
        "skipped_macs": skipped_macs
    }

    return output, stats


# ============================================================
# ZERO-SKIPPING MODEL
# ============================================================

class ZeroSkippingCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(
            1,
            16,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            16,
            32,
            kernel_size=3,
            padding=1
        )

        self.conv3 = nn.Conv2d(
            32,
            64,
            kernel_size=3,
            padding=1
        )

        self.fc1 = nn.Linear(
            64 * 6 * 6,
            128
        )

        self.fc2 = nn.Linear(
            128,
            9
        )

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(2)

        self.dropout = nn.Dropout(0.3)

    def forward(
        self,
        x,
        collect_stats=False
    ):

        stats = {
            "total_macs": 0,
            "executed_macs": 0,
            "skipped_macs": 0
        }

        # ----------------------------------------------------
        # Conv 1
        # ----------------------------------------------------

        x, s = zero_skip_conv2d(
            x,
            self.conv1
        )

        for key in stats:
            stats[key] += s[key]

        x = self.relu(x)

        x = self.pool(x)

        # ----------------------------------------------------
        # Conv 2
        # ----------------------------------------------------

        x, s = zero_skip_conv2d(
            x,
            self.conv2
        )

        for key in stats:
            stats[key] += s[key]

        x = self.relu(x)

        x = self.pool(x)

        # ----------------------------------------------------
        # Conv 3
        # ----------------------------------------------------

        x, s = zero_skip_conv2d(
            x,
            self.conv3
        )

        for key in stats:
            stats[key] += s[key]

        x = self.relu(x)

        x = self.pool(x)

        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        x = torch.flatten(
            x,
            1
        )

        x = self.fc1(x)

        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        if collect_stats:

            return x, stats

        return x


# ============================================================
# LOAD WEIGHTS
# ============================================================

def build_model():

    baseline = WaferCNN(
        num_classes=9
    ).to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    baseline.load_state_dict(
        checkpoint
    )

    model = ZeroSkippingCNN().to(DEVICE)

    model.conv1.load_state_dict(
        baseline.features[0].state_dict()
    )

    model.conv2.load_state_dict(
        baseline.features[3].state_dict()
    )

    model.conv3.load_state_dict(
        baseline.features[6].state_dict()
    )

    model.fc1.load_state_dict(
        baseline.classifier[1].state_dict()
    )

    model.fc2.load_state_dict(
        baseline.classifier[4].state_dict()
    )

    model.eval()

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader):

    model.eval()

    predictions = []
    labels_all = []

    total_macs = 0
    executed_macs = 0
    skipped_macs = 0

    start = time.perf_counter()

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE,
                dtype=torch.float32
            )

            outputs, stats = model(
                images,
                collect_stats=True
            )

            preds = outputs.argmax(
                dim=1
            )

            predictions.extend(
                preds.cpu().numpy()
            )

            labels_all.extend(
                labels.numpy()
            )

            total_macs += stats[
                "total_macs"
            ]

            executed_macs += stats[
                "executed_macs"
            ]

            skipped_macs += stats[
                "skipped_macs"
            ]

    end = time.perf_counter()

    total_time = end - start

    accuracy = accuracy_score(
        labels_all,
        predictions
    )

    macro_f1 = f1_score(
        labels_all,
        predictions,
        average="macro"
    )

    latency_ms = (
        total_time /
        len(labels_all)
    ) * 1000

    skip_percentage = (
        skipped_macs /
        total_macs
    ) * 100

    return (
        accuracy,
        macro_f1,
        latency_ms,
        total_macs,
        executed_macs,
        skipped_macs,
        skip_percentage
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("METHOD 2 - ZERO SKIPPING")
    print("=" * 70)

    print(
        "This is a software simulation of zero-skipping."
    )

    print(
        "Latency is CPU software latency."
    )

    loader = prepare_data()

    print("\nLoading model...")

    model = build_model()

    print("\nRunning zero-skipping evaluation...")

    (
        accuracy,
        macro_f1,
        latency,
        total_macs,
        executed_macs,
        skipped_macs,
        skip_percentage
    ) = evaluate(
        model,
        loader
    )

    print("\n" + "=" * 70)
    print("ZERO-SKIPPING RESULTS")
    print("=" * 70)

    print(
        f"Accuracy             : {accuracy:.6f}"
    )

    print(
        f"Macro F1             : {macro_f1:.6f}"
    )

    print(
        f"CPU latency          : {latency:.6f} ms/image"
    )

    print(
        f"Total MAC operations : {total_macs:,}"
    )

    print(
        f"Executed MACs        : {executed_macs:,}"
    )

    print(
        f"Skipped MACs         : {skipped_macs:,}"
    )

    print(
        f"MAC skip percentage  : {skip_percentage:.2f}%"
    )

    os.makedirs(
        "reports",
        exist_ok=True
    )

    report_path = (
        "reports/method2_zero_skipping.txt"
    )

    with open(
        report_path,
        "w"
    ) as f:

        f.write(
            "METHOD 2 - ZERO SKIPPING\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            "Accuracy: "
            f"{accuracy:.6f}\n"
        )

        f.write(
            "Macro F1: "
            f"{macro_f1:.6f}\n"
        )

        f.write(
            "CPU latency (ms/image): "
            f"{latency:.6f}\n"
        )

        f.write(
            "Total MACs: "
            f"{total_macs}\n"
        )

        f.write(
            "Executed MACs: "
            f"{executed_macs}\n"
        )

        f.write(
            "Skipped MACs: "
            f"{skipped_macs}\n"
        )

        f.write(
            "MAC skip percentage: "
            f"{skip_percentage:.2f}%\n"
        )

    print("\nReport saved to:")
    print(report_path)


if __name__ == "__main__":
    main()
