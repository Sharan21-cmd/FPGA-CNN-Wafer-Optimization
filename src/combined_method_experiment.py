import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report

sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

from dataset import WaferDataset, CLASSES
from method3_separable_conv import (
    SeparableWaferCNN,
    load_wm811k,
    prepare_dataframe,
    IMAGE_SIZE,
    DATA_PATH,
)


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "separable_cnn.pth"
)

REPORT_PATH = os.path.join(
    PROJECT_ROOT,
    "reports",
    "combined_method_results.txt"
)

BATCH_SIZE = 128
DEVICE = torch.device("cpu")

# Test several CPU parallelism levels.
CPU_THREADS = [1, 2, 4]


# ============================================================
# 8-BIT QUANTIZATION
# ============================================================

def quantize_symmetric_8bit(x):

    max_abs = x.abs().max()

    if max_abs == 0:
        return x.clone()

    scale = max_abs / 127.0

    q = torch.round(
        x / scale
    ).clamp(-127, 127)

    # Fake quantization:
    # integer representation is simulated,
    # then restored to FP32 for PyTorch execution.
    return q * scale


def quantize_model_weights_8bit(model):

    with torch.no_grad():

        for module in model.modules():

            if isinstance(
                module,
                (nn.Conv2d, nn.Linear)
            ):

                module.weight.data = (
                    quantize_symmetric_8bit(
                        module.weight.data
                    )
                )

                if module.bias is not None:

                    module.bias.data = (
                        quantize_symmetric_8bit(
                            module.bias.data
                        )
                    )


# ============================================================
# 8-BIT ACTIVATION QUANTIZATION
# ============================================================

def fake_quantize_activation(x):

    max_abs = x.detach().abs().max()

    if max_abs == 0:

        return x

    scale = max_abs / 127.0

    q = torch.round(
        x / scale
    ).clamp(-127, 127)

    return q * scale


# ============================================================
# COMBINED MODEL
# ============================================================

class CombinedModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = SeparableWaferCNN(
            num_classes=len(CLASSES)
        )

        self.zero_stats = {
            "total": 0,
            "zeros": 0
        }

    def reset_stats(self):

        self.zero_stats = {
            "total": 0,
            "zeros": 0
        }

    def record_activation(self, x):

        self.zero_stats["total"] += x.numel()

        self.zero_stats["zeros"] += (
            x == 0
        ).sum().item()

    def forward(self, x):

        # Input is exactly represented by
        # the original wafer-map values.
        x = fake_quantize_activation(x)

        # ----------------------------------------------------
        # DSC block 1
        # ----------------------------------------------------

        x = self.model.features[0].depthwise(x)

        x = fake_quantize_activation(x)

        self.record_activation(x)

        x = self.model.features[0].pointwise(x)

        x = fake_quantize_activation(x)

        x = self.model.features[1](x)

        self.record_activation(x)

        x = self.model.features[2](x)

        # ----------------------------------------------------
        # DSC block 2
        # ----------------------------------------------------

        x = self.model.features[3].depthwise(x)

        x = fake_quantize_activation(x)

        self.record_activation(x)

        x = self.model.features[3].pointwise(x)

        x = fake_quantize_activation(x)

        x = self.model.features[4](x)

        self.record_activation(x)

        x = self.model.features[5](x)

        # ----------------------------------------------------
        # DSC block 3
        # ----------------------------------------------------

        x = self.model.features[6].depthwise(x)

        x = fake_quantize_activation(x)

        self.record_activation(x)

        x = self.model.features[6].pointwise(x)

        x = fake_quantize_activation(x)

        x = self.model.features[7](x)

        self.record_activation(x)

        x = self.model.features[8](x)

        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        x = self.model.classifier[0](x)

        x = self.model.classifier[1](x)

        x = fake_quantize_activation(x)

        x = self.model.classifier[2](x)

        x = self.model.classifier[3](x)

        return self.model.classifier[4](x)


# ============================================================
# MAC ANALYSIS
# ============================================================

def calculate_dsc_macs():

    # Same architecture as Method 3.

    standard = 0
    separable = 0

    # Conv 1
    h = 48
    w = 48

    standard += (
        h * w * 3 * 3 * 1 * 16
    )

    separable += (
        h * w *
        (9 * 1 + 1 * 1 * 16)
    )

    # Pool
    h = 24
    w = 24

    # Conv 2
    standard += (
        h * w * 3 * 3 * 16 * 32
    )

    separable += (
        h * w *
        (9 * 16 + 16 * 32)
    )

    # Pool
    h = 12
    w = 12

    # Conv 3
    standard += (
        h * w * 3 * 3 * 32 * 64
    )

    separable += (
        h * w *
        (9 * 32 + 32 * 64)
    )

    return standard, separable


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader):

    model.eval()

    targets = []
    predictions = []

    model.reset_stats()

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            outputs = model(images)

            pred = outputs.argmax(
                dim=1
            )

            targets.extend(
                labels.numpy()
            )

            predictions.extend(
                pred.numpy()
            )

    targets = np.asarray(targets)
    predictions = np.asarray(predictions)

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
# CPU PARALLEL LATENCY
# ============================================================

def benchmark_latency(
    model,
    loader,
    threads
):

    torch.set_num_threads(
        threads
    )

    model.eval()

    # Warm-up
    with torch.no_grad():

        for i, (images, _) in enumerate(loader):

            images = images.to(DEVICE)

            _ = model(images)

            if i >= 9:
                break

    total_images = 0

    start = time.perf_counter()

    with torch.no_grad():

        for i, (images, _) in enumerate(loader):

            images = images.to(DEVICE)

            _ = model(images)

            total_images += images.size(0)

            if i >= 99:
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
    print("COMBINED METHOD")
    print("DSC + 8-BIT + ZERO-SKIPPING + PARALLEL CPU")
    print("=" * 70)

    print()

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = load_wm811k(
        DATA_PATH
    )

    test_df = prepare_dataframe(
        df,
        "Test"
    )

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

    # --------------------------------------------------------
    # Load trained DSC model
    # --------------------------------------------------------

    print()
    print(
        "Loading trained DSC model..."
    )

    combined = CombinedModel().to(DEVICE)

    state = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    combined.model.load_state_dict(
        state
    )

    # --------------------------------------------------------
    # 8-bit weight quantization
    # --------------------------------------------------------

    print(
        "Applying 8-bit weight quantization..."
    )

    quantize_model_weights_8bit(
        combined.model
    )

    combined.eval()

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    parameters = sum(
        p.numel()
        for p in combined.parameters()
    )

    print(
        f"Parameters: {parameters:,}"
    )

    # --------------------------------------------------------
    # MAC analysis
    # --------------------------------------------------------

    standard_macs, dsc_macs = (
        calculate_dsc_macs()
    )

    mac_reduction = (
        1 -
        dsc_macs /
        standard_macs
    ) * 100

    print()
    print(
        f"Standard convolution MACs: "
        f"{standard_macs:,}"
    )

    print(
        f"DSC MACs: "
        f"{dsc_macs:,}"
    )

    print(
        f"DSC MAC reduction: "
        f"{mac_reduction:.2f}%"
    )

    # --------------------------------------------------------
    # Accuracy + Macro F1
    # --------------------------------------------------------

    print()
    print(
        "Evaluating combined model..."
    )

    (
        accuracy,
        macro_f1,
        targets,
        predictions
    ) = evaluate(
        combined,
        test_loader
    )

    print()
    print(
        f"Accuracy: "
        f"{accuracy * 100:.4f}%"
    )

    print(
        f"Macro F1: "
        f"{macro_f1 * 100:.4f}%"
    )

    # --------------------------------------------------------
    # Zero statistics
    # --------------------------------------------------------

    zero_total = (
        combined.zero_stats["total"]
    )

    zero_count = (
        combined.zero_stats["zeros"]
    )

    zero_percentage = (
        zero_count /
        zero_total *
        100
    )

    print()
    print(
        "Zero-activation analysis:"
    )

    print(
        f"Total activation values: "
        f"{zero_total:,}"
    )

    print(
        f"Zero activation values: "
        f"{zero_count:,}"
    )

    print(
        f"Zero activation rate: "
        f"{zero_percentage:.4f}%"
    )

    # --------------------------------------------------------
    # Estimate MAC skipping opportunity
    # --------------------------------------------------------

    potential_skipped = (
        int(
            dsc_macs *
            zero_percentage /
            100
        )
    )

    potential_executed = (
        dsc_macs -
        potential_skipped
    )

    print()
    print(
        f"Potential skipped DSC MACs: "
        f"{potential_skipped:,}"
    )

    print(
        f"Potential executed DSC MACs: "
        f"{potential_executed:,}"
    )

    # --------------------------------------------------------
    # Parallel CPU benchmark
    # --------------------------------------------------------

    print()
    print(
        "Parallel CPU latency benchmark:"
    )

    latency_results = {}

    for threads in CPU_THREADS:

        latency = benchmark_latency(
            combined,
            test_loader,
            threads
        )

        latency_results[
            threads
        ] = latency

        print(
            f"  {threads} CPU thread(s): "
            f"{latency:.6f} ms/image"
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
            "COMBINED METHOD\n"
        )

        f.write(
            "===============\n\n"
        )

        f.write(
            "DSC + 8-bit + Zero-Skipping + "
            "Parallel CPU\n\n"
        )

        f.write(
            f"Test samples: {len(test_df)}\n"
        )

        f.write(
            f"Input size: 1x{IMAGE_SIZE}x{IMAGE_SIZE}\n"
        )

        f.write(
            f"Parameters: {parameters:,}\n\n"
        )

        f.write(
            f"Accuracy: "
            f"{accuracy * 100:.4f}%\n"
        )

        f.write(
            f"Macro F1: "
            f"{macro_f1 * 100:.4f}%\n\n"
        )

        f.write(
            f"Standard convolution MACs: "
            f"{standard_macs:,}\n"
        )

        f.write(
            f"DSC MACs: "
            f"{dsc_macs:,}\n"
        )

        f.write(
            f"DSC MAC reduction: "
            f"{mac_reduction:.4f}%\n\n"
        )

        f.write(
            f"Zero activation rate: "
            f"{zero_percentage:.4f}%\n"
        )

        f.write(
            f"Potential skipped DSC MACs: "
            f"{potential_skipped:,}\n"
        )

        f.write(
            f"Potential executed DSC MACs: "
            f"{potential_executed:,}\n\n"
        )

        f.write(
            "Parallel CPU latency\n"
        )

        f.write(
            "--------------------\n"
        )

        for threads, latency in (
            latency_results.items()
        ):

            f.write(
                f"{threads} thread(s): "
                f"{latency:.6f} ms/image\n"
            )

        f.write(
            "\nClassification Report\n"
        )

        f.write(
            "---------------------\n"
        )

        f.write(report)

        f.write(
            "\n\nIMPORTANT:\n"
        )

        f.write(
            "Zero-skipping percentage represents "
            "potential arithmetic reduction. "
            "It is not claimed as measured CPU "
            "speedup.\n"
        )

        f.write(
            "8-bit quantization is simulated using "
            "fake quantization; PyTorch still executes "
            "FP32 operations internally.\n"
        )

    print()
    print(
        f"Report saved: {REPORT_PATH}"
    )

    print()
    print("=" * 70)
    print("COMBINED EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
