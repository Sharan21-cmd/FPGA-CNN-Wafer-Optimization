import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Allow imports from src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import WaferDataset, CLASSES
from model import WaferCNN


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "LSWMD.pkl"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "baseline_cnn.pth"
)

REPORT_PATH = os.path.join(
    PROJECT_ROOT,
    "reports",
    "method2_zero_skipping.txt"
)

IMAGE_SIZE = 48
BATCH_SIZE = 128

DEVICE = torch.device("cpu")


# ============================================================
# LEGACY WM-811K PICKLE LOADER
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
# PREPARE TEST DATA
# ============================================================

def prepare_test_dataframe(df):

    labels = df["failureType"].apply(extract_label)
    split = df["trianTestLabel"].apply(extract_label)

    mask = (
        split.eq("Test")
        & labels.isin(CLASSES)
    )

    test_df = df.loc[
        mask,
        ["waferMap", "failureType"]
    ].copy()

    test_df["failure"] = test_df["failureType"].apply(
        extract_label
    )

    test_df = test_df[
        ["waferMap", "failure"]
    ].reset_index(drop=True)

    return test_df


# ============================================================
# ACTIVATION STATISTICS
# ============================================================

class ActivationCollector:

    def __init__(self):
        self.activations = {}

    def hook(self, name):

        def fn(module, inputs, output):

            # Store only the activation required for
            # zero-skipping analysis.
            self.activations[name] = output.detach()

        return fn


# ============================================================
# MAC ANALYSIS
# ============================================================

def analyze_conv_zero_skipping(
    activation,
    conv,
    layer_name
):

    """
    Calculate theoretical MAC reduction if zero input
    activations were skipped.

    This does NOT perform a custom sparse convolution.

    Therefore:
        - accuracy remains identical to baseline
        - predictions remain identical
        - MAC statistics represent potential skipping
    """

    batch_size = activation.shape[0]

    in_channels = activation.shape[1]
    height = activation.shape[2]
    width = activation.shape[3]

    out_channels = conv.out_channels

    kernel_h = conv.kernel_size[0]
    kernel_w = conv.kernel_size[1]

    stride_h = conv.stride[0]
    stride_w = conv.stride[1]

    padding_h = conv.padding[0]
    padding_w = conv.padding[1]

    dilation_h = conv.dilation[0]
    dilation_w = conv.dilation[1]

    # Output dimensions
    out_h = (
        (height + 2 * padding_h
         - dilation_h * (kernel_h - 1)
         - 1)
        // stride_h
        + 1
    )

    out_w = (
        (width + 2 * padding_w
         - dilation_w * (kernel_w - 1)
         - 1)
        // stride_w
        + 1
    )

    # Total dense MAC count
    total_macs = (
        batch_size
        * out_h
        * out_w
        * in_channels
        * kernel_h
        * kernel_w
        * out_channels
    )

    # --------------------------------------------------------
    # Count exact zero-input MAC opportunities
    # --------------------------------------------------------

    zero_mask = (
        activation == 0
    ).to(torch.float32)

    # Convert each input channel into convolution windows.
    unfolded = torch.nn.functional.unfold(
        zero_mask,
        kernel_size=(kernel_h, kernel_w),
        dilation=(dilation_h, dilation_w),
        padding=(padding_h, padding_w),
        stride=(stride_h, stride_w)
    )

    # Shape:
    # [batch, in_channels * kernel_h * kernel_w, output_positions]

    zero_elements = unfolded.sum().item()

    # Each zero input element would otherwise participate
    # in one multiplication for EVERY output channel.
    skipped_macs = int(
        zero_elements * out_channels
    )

    executed_macs = max(
        int(total_macs - skipped_macs),
        0
    )

    skip_percentage = (
        skipped_macs / total_macs * 100
        if total_macs > 0
        else 0.0
    )

    nonzero_percentage = 100.0 - skip_percentage

    return {
        "layer": layer_name,
        "total_macs": int(total_macs),
        "skipped_macs": skipped_macs,
        "executed_macs": executed_macs,
        "skip_percentage": skip_percentage,
        "nonzero_percentage": nonzero_percentage,
        "activation_elements": int(
            activation.numel()
        ),
        "zero_activation_elements": int(
            (activation == 0).sum().item()
        ),
    }


# ============================================================
# LATENCY BENCHMARK
# ============================================================

def benchmark_latency(
    model,
    loader,
    warmup_batches=10,
    benchmark_batches=100
):

    model.eval()

    # Warm-up
    with torch.no_grad():

        for batch_index, (images, _) in enumerate(loader):

            images = images.to(DEVICE)

            _ = model(images)

            if batch_index + 1 >= warmup_batches:
                break

    # Benchmark
    total_images = 0

    start = time.perf_counter()

    with torch.no_grad():

        for batch_index, (images, _) in enumerate(loader):

            images = images.to(DEVICE)

            _ = model(images)

            total_images += images.size(0)

            if batch_index + 1 >= benchmark_batches:
                break

    elapsed = time.perf_counter() - start

    latency_ms = (
        elapsed / total_images * 1000
    )

    return latency_ms


# ============================================================
# FULL EVALUATION
# ============================================================

def evaluate_model(
    model,
    loader
):

    model.eval()

    correct = 0
    total = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():

        for images, targets in loader:

            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(images)

            predictions = (
                outputs.argmax(dim=1)
            )

            correct += (
                predictions == targets
            ).sum().item()

            total += targets.size(0)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_targets.extend(
                targets.cpu().numpy()
            )

    accuracy = correct / total

    return (
        accuracy,
        np.asarray(all_targets),
        np.asarray(all_predictions)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("METHOD 2 - ZERO SKIPPING")
    print("=" * 70)

    print(
        "Vectorized zero-skipping analysis."
    )

    print(
        "No Python-level convolution loops."
    )

    print(
        "CPU latency is measured using the normal CNN."
    )

    print(
        "Zero-skipping MAC reduction is reported separately."
    )

    print()

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = load_wm811k(DATA_PATH)

    test_df = prepare_test_dataframe(df)

    print(
        f"Test samples: {len(test_df)}"
    )

    print()

    dataset = WaferDataset(
        test_df,
        image_size=IMAGE_SIZE
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # Load baseline model
    # --------------------------------------------------------

    print("Loading model...")

    model = WaferCNN(
        num_classes=len(CLASSES)
    ).to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint)

    model.eval()

    print("Model loaded.")
    print()

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    print("Evaluating baseline predictions...")

    accuracy, targets, predictions = evaluate_model(
        model,
        loader
    )

    print(
        f"Accuracy: {accuracy * 100:.4f}%"
    )

    # --------------------------------------------------------
    # Macro F1
    # --------------------------------------------------------

    from sklearn.metrics import f1_score

    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro"
    )

    print(
        f"Macro F1: {macro_f1 * 100:.4f}%"
    )

    print()

    # --------------------------------------------------------
    # CPU latency
    # --------------------------------------------------------

    print(
        "Benchmarking CPU software latency..."
    )

    latency_ms = benchmark_latency(
        model,
        loader
    )

    print(
        f"CPU latency: {latency_ms:.6f} ms/image"
    )

    print()

    # --------------------------------------------------------
    # Collect activations
    # --------------------------------------------------------

    print(
        "Collecting ReLU activation statistics..."
    )

    collector = ActivationCollector()

    handles = []

    handles.append(
        model.features[1].register_forward_hook(
            collector.hook("features.1")
        )
    )

    handles.append(
        model.features[4].register_forward_hook(
            collector.hook("features.4")
        )
    )

    handles.append(
        model.features[7].register_forward_hook(
            collector.hook("features.7")
        )
    )

    # We analyze batches separately because the full
    # 118k test set cannot be held in memory as activations.

    layer_stats = {
        "features.1": {
            "total_macs": 0,
            "skipped_macs": 0,
            "executed_macs": 0,
            "activation_elements": 0,
            "zero_activation_elements": 0
        },
        "features.4": {
            "total_macs": 0,
            "skipped_macs": 0,
            "executed_macs": 0,
            "activation_elements": 0,
            "zero_activation_elements": 0
        },
        "features.7": {
            "total_macs": 0,
            "skipped_macs": 0,
            "executed_macs": 0,
            "activation_elements": 0,
            "zero_activation_elements": 0
        }
    }

    # --------------------------------------------------------
    # Analyze every test batch
    # --------------------------------------------------------

    with torch.no_grad():

        for batch_index, (images, _) in enumerate(loader):

            images = images.to(DEVICE)

            _ = model(images)

            for layer_name, conv_index in [
                ("features.1", 0),
                ("features.4", 3),
                ("features.7", 6)
            ]:

                activation = collector.activations[
                    layer_name
                ]

                conv = model.features[
                    conv_index
                ]

                stats = analyze_conv_zero_skipping(
                    activation,
                    conv,
                    layer_name
                )

                for key in [
                    "total_macs",
                    "skipped_macs",
                    "executed_macs",
                    "activation_elements",
                    "zero_activation_elements"
                ]:

                    layer_stats[
                        layer_name
                    ][key] += stats[key]

            if (batch_index + 1) % 100 == 0:

                print(
                    f"  analyzed "
                    f"{batch_index + 1} batches..."
                )

    # Remove hooks
    for handle in handles:
        handle.remove()

    # --------------------------------------------------------
    # Calculate final statistics
    # --------------------------------------------------------

    total_macs_all = 0
    skipped_macs_all = 0
    executed_macs_all = 0

    print()
    print("=" * 70)
    print("ZERO-SKIPPING RESULTS")
    print("=" * 70)

    for layer_name in [
        "features.1",
        "features.4",
        "features.7"
    ]:

        stats = layer_stats[layer_name]

        total_macs_all += stats["total_macs"]
        skipped_macs_all += stats["skipped_macs"]
        executed_macs_all += stats["executed_macs"]

        skip_percentage = (
            stats["skipped_macs"]
            / stats["total_macs"]
            * 100
        )

        zero_percentage = (
            stats["zero_activation_elements"]
            / stats["activation_elements"]
            * 100
        )

        print()
        print(layer_name)

        print(
            f"  Activation zero rate : "
            f"{zero_percentage:.4f}%"
        )

        print(
            f"  Total MACs           : "
            f"{stats['total_macs']:,}"
        )

        print(
            f"  Skipped MACs         : "
            f"{stats['skipped_macs']:,}"
        )

        print(
            f"  Executed MACs        : "
            f"{stats['executed_macs']:,}"
        )

        print(
            f"  Potential MAC skip   : "
            f"{skip_percentage:.4f}%"
        )

    overall_skip = (
        skipped_macs_all
        / total_macs_all
        * 100
    )

    print()
    print("-" * 70)

    print(
        f"TOTAL MACs             : "
        f"{total_macs_all:,}"
    )

    print(
        f"TOTAL skipped MACs     : "
        f"{skipped_macs_all:,}"
    )

    print(
        f"TOTAL executed MACs    : "
        f"{executed_macs_all:,}"
    )

    print(
        f"OVERALL MAC skip       : "
        f"{overall_skip:.4f}%"
    )

    print()
    print(
        "NOTE:"
    )

    print(
        "The reported CPU latency is the normal dense "
        "PyTorch CNN latency."
    )

    print(
        "The MAC skip percentage represents the theoretical "
        "arithmetic reduction available to a true "
        "zero-skipping implementation."
    )

    print(
        "It is NOT claimed as a CPU speedup."
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(REPORT_PATH),
        exist_ok=True
    )

    with open(REPORT_PATH, "w") as f:

        f.write(
            "METHOD 2 - ZERO SKIPPING\n"
        )

        f.write(
            "========================\n\n"
        )

        f.write(
            "Test samples: "
            f"{len(test_df)}\n"
        )

        f.write(
            f"Input size: 1x{IMAGE_SIZE}x{IMAGE_SIZE}\n"
        )

        f.write(
            f"Accuracy: {accuracy * 100:.4f}%\n"
        )

        f.write(
            f"Macro F1: {macro_f1 * 100:.4f}%\n"
        )

        f.write(
            f"CPU latency: {latency_ms:.6f} ms/image\n\n"
        )

        for layer_name in [
            "features.1",
            "features.4",
            "features.7"
        ]:

            stats = layer_stats[layer_name]

            skip_percentage = (
                stats["skipped_macs"]
                / stats["total_macs"]
                * 100
            )

            zero_percentage = (
                stats["zero_activation_elements"]
                / stats["activation_elements"]
                * 100
            )

            f.write(
                f"{layer_name}\n"
            )

            f.write(
                f"Activation zero rate: "
                f"{zero_percentage:.4f}%\n"
            )

            f.write(
                f"Total MACs: "
                f"{stats['total_macs']:,}\n"
            )

            f.write(
                f"Skipped MACs: "
                f"{stats['skipped_macs']:,}\n"
            )

            f.write(
                f"Executed MACs: "
                f"{stats['executed_macs']:,}\n"
            )

            f.write(
                f"Potential MAC skip: "
                f"{skip_percentage:.4f}%\n\n"
            )

        f.write(
            f"Overall MAC skip: "
            f"{overall_skip:.4f}%\n\n"
        )

        f.write(
            "CPU latency is dense PyTorch latency. "
            "MAC skip percentage is theoretical potential "
            "for a true zero-skipping implementation.\n"
        )

    print()
    print(
        f"Report saved to: {REPORT_PATH}"
    )

    print()
    print("=" * 70)
    print("METHOD 2 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
