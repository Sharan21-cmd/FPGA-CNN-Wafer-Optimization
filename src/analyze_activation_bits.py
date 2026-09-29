import os
import pickle
import numpy as np
import torch

from torch.utils.data import DataLoader
from dataset import WaferDataset, CLASSES
from model import WaferCNN


DATA_PATH = "data/raw/LSWMD.pkl"
MODEL_PATH = "models/baseline_cnn.pth"

IMAGE_SIZE = 48
BATCH_SIZE = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    print("Loading WM-811K...")

    with open(path, "rb") as f:
        return CustomUnpickler(
            f,
            encoding="latin1"
        ).load()


def extract_label(x):

    try:
        return str(x.flat[0])
    except Exception:
        return "unknown"


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_dataframe(df):

    df = df.copy()

    df["failure"] = df["failureType"].apply(
        extract_label
    )

    df["split"] = df["trianTestLabel"].apply(
        extract_label
    )

    df = df[
        df["failure"].isin(CLASSES)
        & df["split"].isin(["Training", "Test"])
    ].copy()

    test_df = df[
        df["split"] == "Test"
    ].copy()

    return test_df


# ============================================================
# ACTIVATION STATISTICS
# ============================================================

class ActivationRecorder:

    def __init__(self):
        self.stats = {}

    def hook(self, name):

        def capture(module, inputs, output):

            tensor = output.detach().cpu()

            if name not in self.stats:
                self.stats[name] = {
                    "min": float("inf"),
                    "max": float("-inf"),
                    "abs_max": 0.0,
                    "values": 0,
                    "nonzero": 0
                }

            current = self.stats[name]

            current["min"] = min(
                current["min"],
                tensor.min().item()
            )

            current["max"] = max(
                current["max"],
                tensor.max().item()
            )

            current["abs_max"] = max(
                current["abs_max"],
                tensor.abs().max().item()
            )

            current["values"] += tensor.numel()

            current["nonzero"] += torch.count_nonzero(
                tensor
            ).item()

        return capture


# ============================================================
# REQUIRED SIGNED INTEGER BITS
# ============================================================

def required_signed_bits(max_abs):

    if max_abs == 0:
        return 1

    # Find smallest signed integer width
    # that can represent [-max_abs, +max_abs].

    for bits in range(1, 33):

        max_positive = (2 ** (bits - 1)) - 1

        if max_abs <= max_positive:
            return bits

    return 32


# ============================================================
# MODEL HOOKS
# ============================================================

def register_hooks(model, recorder):

    handles = []

    for name, module in model.named_modules():

        if isinstance(
            module,
            (
                torch.nn.Conv2d,
                torch.nn.Linear,
                torch.nn.ReLU,
                torch.nn.MaxPool2d
            )
        ):

            handle = module.register_forward_hook(
                recorder.hook(name)
            )

            handles.append(handle)

    return handles


# ============================================================
# WEIGHT ANALYSIS
# ============================================================

def analyze_weights(model):

    print("\n")
    print("=" * 70)
    print("                    WEIGHT ANALYSIS")
    print("=" * 70)

    print(
        f"{'Layer':25s}"
        f"{'Min':>12s}"
        f"{'Max':>12s}"
        f"{'Abs Max':>12s}"
        f"{'Bits':>8s}"
    )

    print("-" * 70)

    for name, module in model.named_modules():

        if isinstance(
            module,
            (
                torch.nn.Conv2d,
                torch.nn.Linear
            )
        ):

            weights = module.weight.detach().cpu()

            minimum = weights.min().item()
            maximum = weights.max().item()
            abs_max = weights.abs().max().item()

            # Weight values are floating point.
            # This is only the integer-magnitude analysis.
            bits = required_signed_bits(
                abs_max
            )

            print(
                f"{name:25s}"
                f"{minimum:12.6f}"
                f"{maximum:12.6f}"
                f"{abs_max:12.6f}"
                f"{bits:8d}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("              CNN ACTIVATION BIT ANALYSIS")
    print("=" * 70)

    print("Device:", DEVICE)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = load_wm811k(DATA_PATH)

    test_df = prepare_dataframe(df)

    print("\nTest samples:", len(test_df))

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
    # Load model
    # --------------------------------------------------------

    model = WaferCNN(
        num_classes=len(CLASSES)
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.eval()

    print(
        "\nLoaded model:",
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Register activation hooks
    # --------------------------------------------------------

    recorder = ActivationRecorder()

    handles = register_hooks(
        model,
        recorder
    )

    # --------------------------------------------------------
    # Run inference
    # --------------------------------------------------------

    print(
        "\nRunning inference and collecting "
        "activation statistics..."
    )

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            model(images)

    # --------------------------------------------------------
    # Remove hooks
    # --------------------------------------------------------

    for handle in handles:
        handle.remove()

    # --------------------------------------------------------
    # Print activation statistics
    # --------------------------------------------------------

    print("\n")
    print("=" * 95)
    print("                         ACTIVATION ANALYSIS")
    print("=" * 95)

    print(
        f"{'Layer':25s}"
        f"{'Min':>12s}"
        f"{'Max':>12s}"
        f"{'Abs Max':>12s}"
        f"{'Bits':>8s}"
        f"{'Nonzero %':>14s}"
    )

    print("-" * 95)

    for name, stats in recorder.stats.items():

        bits = required_signed_bits(
            stats["abs_max"]
        )

        nonzero_percent = (
            stats["nonzero"]
            / stats["values"]
        ) * 100

        print(
            f"{name:25s}"
            f"{stats['min']:12.4f}"
            f"{stats['max']:12.4f}"
            f"{stats['abs_max']:12.4f}"
            f"{bits:8d}"
            f"{nonzero_percent:13.2f}%"
        )

    # --------------------------------------------------------
    # Weight analysis
    # --------------------------------------------------------

    analyze_weights(model)

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    os.makedirs(
        "reports",
        exist_ok=True
    )

    report_path = (
        "reports/activation_bit_analysis.txt"
    )

    with open(
        report_path,
        "w"
    ) as f:

        f.write(
            "CNN ACTIVATION AND WEIGHT BIT ANALYSIS\n"
        )

        f.write(
            "=======================================\n\n"
        )

        f.write(
            f"Model: {MODEL_PATH}\n"
        )

        f.write(
            f"Input: {IMAGE_SIZE}x{IMAGE_SIZE}\n"
        )

        f.write(
            f"Test samples: {len(test_df)}\n\n"
        )

        f.write(
            "ACTIVATIONS\n"
        )

        f.write(
            "-----------\n"
        )

        for name, stats in recorder.stats.items():

            bits = required_signed_bits(
                stats["abs_max"]
            )

            nonzero_percent = (
                stats["nonzero"]
                / stats["values"]
            ) * 100

            f.write(
                f"{name}: "
                f"min={stats['min']:.6f}, "
                f"max={stats['max']:.6f}, "
                f"abs_max={stats['abs_max']:.6f}, "
                f"required_integer_bits={bits}, "
                f"nonzero={nonzero_percent:.2f}%\n"
            )

        f.write(
            "\nWEIGHTS\n"
        )

        f.write(
            "-------\n"
        )

        for name, module in model.named_modules():

            if isinstance(
                module,
                (
                    torch.nn.Conv2d,
                    torch.nn.Linear
                )
            ):

                weights = module.weight.detach().cpu()

                minimum = weights.min().item()
                maximum = weights.max().item()
                abs_max = weights.abs().max().item()

                bits = required_signed_bits(
                    abs_max
                )

                f.write(
                    f"{name}: "
                    f"min={minimum:.6f}, "
                    f"max={maximum:.6f}, "
                    f"abs_max={abs_max:.6f}, "
                    f"required_integer_bits={bits}\n"
                )

    print("\n")
    print("=" * 70)
    print("Analysis complete.")
    print("Report saved to:")
    print(report_path)
    print("=" * 70)


if __name__ == "__main__":
    main()
