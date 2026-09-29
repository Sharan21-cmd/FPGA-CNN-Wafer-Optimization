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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import WaferCNN
from src.dataset import WaferDataset


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "data/raw/LSWMD.pkl"
MODEL_PATH = "models/baseline_cnn.pth"

IMAGE_SIZE = 48
BATCH_SIZE = 128

BIT_WIDTHS = [8, 7, 6, 5, 4]

DEVICE = torch.device("cpu")


# ============================================================
# LEGACY WM-811K LOADER
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
# DATA PREPARATION
# ============================================================

def prepare_data():

    print("Loading WM-811K...")

    df = load_wm811k(DATA_PATH)

    df["split"] = df["trianTestLabel"].apply(extract_label)
    df["failure"] = df["failureType"].apply(extract_label)

    df = df[
        df["split"].isin(["Training", "Test"])
    ].copy()

    train_df = df[
        df["split"] == "Training"
    ].copy()

    test_df = df[
        df["split"] == "Test"
    ].copy()

    print(f"Training samples: {len(train_df)}")
    print(f"Test samples    : {len(test_df)}")

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
# QUANTIZATION
# ============================================================

def quantize_tensor(x, bits):

    """
    Symmetric signed quantization.

    The quantized tensor is immediately dequantized
    back to floating point so that PyTorch continues to
    execute the CNN normally.

    This is a software numerical-precision experiment,
    NOT a true integer hardware implementation.
    """

    max_abs = x.abs().max()

    if max_abs == 0:
        return x.clone()

    qmax = (2 ** (bits - 1)) - 1

    scale = max_abs / qmax

    q = torch.round(
        x / scale
    ).clamp(
        -qmax,
        qmax
    )

    return q * scale


# ============================================================
# QUANTIZE MODEL
# ============================================================

def quantize_model(model, bits):

    model_q = WaferCNN(
        num_classes=9
    ).to(DEVICE)

    model_q.load_state_dict(
        model.state_dict()
    )

    with torch.no_grad():

        for name, parameter in model_q.named_parameters():

            if "weight" in name:
                parameter.copy_(
                    quantize_tensor(
                        parameter,
                        bits
                    )
                )

    return model_q


# ============================================================
# ACTIVATION QUANTIZATION
# ============================================================

class ActivationQuantizer:

    def __init__(self, bits):
        self.bits = bits

    def __call__(self, module, inputs, output):

        if isinstance(output, torch.Tensor):

            return quantize_tensor(
                output,
                self.bits
            )

        return output


def attach_activation_quantizers(model, bits):

    hooks = []

    for module in model.modules():

        if isinstance(
            module,
            (
                nn.Conv2d,
                nn.Linear
            )
        ):

            hook = module.register_forward_hook(
                ActivationQuantizer(bits)
            )

            hooks.append(hook)

    return hooks


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, test_loader, measure_latency=False):

    model.eval()

    all_predictions = []
    all_labels = []

    total_time = 0.0
    total_images = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(
                DEVICE,
                dtype=torch.float32
            )

            labels = labels.to(DEVICE)

            if measure_latency:

                start = time.perf_counter()

                outputs = model(images)

                end = time.perf_counter()

                total_time += end - start
                total_images += images.size(0)

            else:

                outputs = model(images)

                total_images += images.size(0)

            predictions = outputs.argmax(
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro"
    )

    latency_ms = (
        total_time /
        total_images
    ) * 1000 if measure_latency else 0.0

    return accuracy, macro_f1, latency_ms


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("METHOD 1 - MSB / BIT-WIDTH REDUCTION")
    print("=" * 70)

    print(f"Device: {DEVICE}")

    test_loader = prepare_data()

    print("\nLoading baseline model...")

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

    baseline.eval()

    # --------------------------------------------------------
    # Baseline latency
    # --------------------------------------------------------

    print("\nMeasuring FP32 baseline...")

    baseline_acc, baseline_f1, baseline_latency = evaluate(
        baseline,
        test_loader,
        measure_latency=True
    )

    print("\nBASELINE")
    print("-" * 70)
    print(f"Accuracy       : {baseline_acc:.6f}")
    print(f"Macro F1       : {baseline_f1:.6f}")
    print(f"CPU latency    : {baseline_latency:.6f} ms/image")

    results = []

    results.append({
        "bits": "FP32",
        "accuracy": baseline_acc,
        "macro_f1": baseline_f1,
        "latency_ms": baseline_latency
    })

    # --------------------------------------------------------
    # Reduced precision experiments
    # --------------------------------------------------------

    for bits in BIT_WIDTHS:

        print("\n" + "=" * 70)
        print(f"TESTING {bits}-BIT")
        print("=" * 70)

        model_q = quantize_model(
            baseline,
            bits
        )

        hooks = attach_activation_quantizers(
            model_q,
            bits
        )

        accuracy, macro_f1, latency = evaluate(
            model_q,
            test_loader,
            measure_latency=True
        )

        for hook in hooks:
            hook.remove()

        print(f"\n{bits}-BIT RESULTS")
        print("-" * 70)
        print(f"Accuracy       : {accuracy:.6f}")
        print(f"Macro F1       : {macro_f1:.6f}")
        print(f"CPU latency    : {latency:.6f} ms/image")

        results.append({
            "bits": bits,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "latency_ms": latency
        })

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    os.makedirs(
        "reports",
        exist_ok=True
    )

    report_path = (
        "reports/method1_msb_reduction.txt"
    )

    with open(
        report_path,
        "w"
    ) as f:

        f.write(
            "METHOD 1 - MSB / BIT-WIDTH REDUCTION\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            "Important: latency is CPU software inference latency.\n"
        )

        f.write(
            "It is NOT FPGA hardware latency.\n\n"
        )

        f.write(
            f"{'Precision':<12}"
            f"{'Accuracy':<15}"
            f"{'Macro F1':<15}"
            f"{'Latency(ms)':<15}\n"
        )

        f.write("-" * 60 + "\n")

        for result in results:

            f.write(
                f"{str(result['bits']):<12}"
                f"{result['accuracy']:<15.6f}"
                f"{result['macro_f1']:<15.6f}"
                f"{result['latency_ms']:<15.6f}\n"
            )

    print("\n" + "=" * 70)
    print("FINAL METHOD 1 RESULTS")
    print("=" * 70)

    print(
        f"{'Precision':<12}"
        f"{'Accuracy':<15}"
        f"{'Macro F1':<15}"
        f"{'Latency(ms)':<15}"
    )

    print("-" * 60)

    for result in results:

        print(
            f"{str(result['bits']):<12}"
            f"{result['accuracy']:<15.6f}"
            f"{result['macro_f1']:<15.6f}"
            f"{result['latency_ms']:<15.6f}"
        )

    print("\nReport saved to:")
    print(report_path)


if __name__ == "__main__":
    main()
